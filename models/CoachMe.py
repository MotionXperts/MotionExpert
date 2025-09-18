from transformers import T5ForConditionalGeneration, AutoConfig
from torch import nn
from visualize_model import model_view, head_view
from .HumanPosePerception import HumanPosePerception
from .Projection import Projection
import torch, os
import torch.distributed as dist

class CoachMe(nn.Module) :
    def __init__(self, cfg) :
        super(CoachMe, self).__init__()
        # Configuration.
        self.cfg = cfg
        self.ref = cfg.TASK.REF
        self.pretrain = cfg.TASK.PRETRAIN
        self.hpp_way = cfg.TASK.HPP_WAY
        self.diff_type = cfg.TASK.DIFF_TYPE
        self.diff_way = cfg.TASK.DIFF_WAY

        # LoRA config.
        if self.pretrain :
            hpp_lora_config, proj_lora_config = None, None
        elif not self.pretrain and cfg.TASK.SPORT == "Skating" :
            hpp_lora_config = {"bias" : "none", "r" : 32, "lora_alpha" : 64, "lora_dropout" : 0.1}
            proj_lora_config = {"bias" : "none", "r" : 32, "lora_alpha" : 64, "lora_dropout" : 0.1}
        elif not self.pretrain and cfg.TASK.SPORT == "Boxing" :
            hpp_lora_config = {"bias" : "none", "r" : 32, "lora_alpha" : 64, "lora_dropout" : 0.1}
            proj_lora_config = {"bias" : "none", "r" : 32, "lora_alpha" : 64, "lora_dropout" : 0.1}

        self.HumanPosePerception = HumanPosePerception(num_class = 1024, in_channel = 6, residual = True,
                                                       dropout = 0.5, t_kernel_size = 9, layout = 'SMPL',
                                                       strategy = 'spatial', hop_size = 3, num_att_graph = 4,
                                                       hpp_way = self.hpp_way, pretrain = self.pretrain,
                                                       lora_config = hpp_lora_config)
        if self.ref or self.diff_type == 'RGB' :
            in_channel = 1024
        else :
            in_channel = self.HumanPosePerception.output_channel

        self.Projection = Projection(self.pretrain, cfg.TASK.PROJ_STRATEGY , in_channel, t5_channel = 768, lora_config = proj_lora_config)

        self.LanguageModel = T5ForConditionalGeneration.from_pretrained('t5-base', config = AutoConfig.from_pretrained('t5-base'))

        self.RGB_lifting = nn.Linear(128, 512)
        # Distributed Training.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the difference feature.
    def get_diff_feat(self, user, standard, diff_way) :
        if diff_way == 'Subtraction' :
            batch_diff = user - standard 

        # diff_way : Padding.
        else :
            batch_diff = torch.zeros(user.shape[0], user.shape[1], user.shape[2], user.shape[3])

        # The dimension of batch_diff is [batchsize, seq_length, vertex(22), channel (512)].
        return batch_diff

    # Get the standard feature.
    def get_std_feat(self, keypoints, seq_len) :
        std_skeleton_coords = keypoints
        # Batch size.
        for i in range(0, keypoints.shape[0]) :
            # Number of frames.
            for j in range(1, seq_len[i]) :
                # joint coordinates (3) + bone coordinates (3).
                for k in range(0, 6) :
                    # Copy the 22 joints of every coordinate.
                    std_skeleton_coords[i][k][j] = keypoints[i][k][0]
        return std_skeleton_coords

    # Get the projection feature.
    def get_proj_feat(self, motion_tokens, difference_tokens, ref, diff_type) :
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if ref and diff_type == 'Skeleton' :
            tokens = torch.cat([motion_tokens, difference_tokens.to(device)], dim = -1)
            tokens, max_indices = self.Projection(tokens)
        elif ref and diff_type == 'RGB' :
            difference_tokens = self.RGB_lifting(difference_tokens)
            difference_tokens = difference_tokens[:, :(motion_tokens).shape[1], :, :]
            tokens = torch.cat([motion_tokens, difference_tokens.to(device)], dim = -1)
            tokens, max_indices = self.Projection(tokens)
        else :
            tokens, max_indices = self.Projection(motion_tokens)
        return tokens, max_indices

    def forward(self,**kwargs) :
        skeleton_coords = kwargs['skeleton_coords'].float()
        frame_mask = kwargs['frame_mask'].float()
        seq_len = kwargs['seq_len']
        standard = kwargs['std_coords'].float()
        decoder_input_ids = kwargs['decoder_input_ids']
        labels = kwargs['labels']
        subtraction = kwargs['subtraction'].float()

        self.HumanPosePerception.train()
        motion_tokens, _, _ = self.HumanPosePerception(skeleton_coords)
        motion_tokens = motion_tokens.float()

        if self.ref :
            with torch.no_grad() :
                self.HumanPosePerception.eval()
                if self.pretrain :
                    standard = self.get_std_feat(skeleton_coords, seq_len)
                    standard = standard.float()
                standard_tokens, _ , _ = self.HumanPosePerception(standard)
                standard_tokens = standard_tokens.float()
            difference_tokens = self.get_diff_feat(motion_tokens, standard_tokens, self.ref)
            difference_tokens = difference_tokens.float()
            tokens, max_indices = self.get_proj_feat(motion_tokens, difference_tokens, self.ref, self.diff_type)

        elif self.diff_type == 'RGB' :
            # The dimension of difference_tokens is [batch size, seq length, 1, 128]
            difference_tokens = subtraction.unsqueeze(2).expand(-1, -1, 22, -1)
            # The dimension of difference_tokens becomes [batch size, seq length, 22, 128]
            difference_tokens = difference_tokens.float()
            tokens, max_indices = self.get_proj_feat(motion_tokens, difference_tokens, self.ref, self.diff_type)
        else :
            tokens, max_indices = self.get_proj_feat(motion_tokens, None, self.ref, self.diff_type)
        tokens = tokens.float()

        return self.LanguageModel(inputs_embeds = tokens.contiguous(),
                                  attention_mask = frame_mask,
                                  decoder_input_ids = decoder_input_ids,
                                  labels = labels.contiguous())

    def generate(self,**kwargs) :
        skeleton_coords = kwargs['skeleton_coords'].float()
        frame_mask = kwargs['frame_mask'].float()
        seq_len = kwargs['seq_len']
        standard = kwargs['std_coords'].float()
        decoder_input_ids = kwargs['decoder_input_ids']
        tokenizer = kwargs['tokenizer']
        subtraction = kwargs['subtraction'].float()
        with torch.no_grad() :
            self.HumanPosePerception.eval()
            motion_tokens, attention_node, attention_graph = self.HumanPosePerception(skeleton_coords)
            motion_tokens = motion_tokens.float()

            if self.ref :
                if self.pretrain :
                    standard = self.get_std_feat(skeleton_coords, seq_len)
                    standard = standard.float()

                standard_tokens, _ , _ = self.HumanPosePerception(standard)
                standard_tokens = standard_tokens.float()

                difference_tokens = self.get_diff_feat(motion_tokens, standard_tokens, self.diff_way).float()
                difference_tokens = difference_tokens.float()

                tokens, max_indices = self.get_proj_feat(motion_tokens.float(), difference_tokens, self.ref, self.diff_type)
            elif self.diff_type== 'RGB' :
                # The dimension of difference_tokens is [batch size, seq length, 1, 128].
                difference_tokens = subtraction.unsqueeze(2).expand(-1, -1, 22, -1)
                difference_tokens = difference_tokens.float()
                # The dimension of difference_tokens is [batch size, seq length, 22, 128].
                tokens, max_indices = self.get_proj_feat(motion_tokens, difference_tokens, self.ref, self.diff_type)
            else :
                tokens, max_indices = self.get_proj_feat(motion_tokens, None, self.ref, self.diff_type)
            tokens = tokens.float()

        dosample = False
        # Set do_sample as True for demo.
        if self.cfg.EVAL.ckpt != "None" :
            dosample = True

        generated_ids = self.LanguageModel.generate(inputs_embeds = tokens,
                                                    attention_mask = frame_mask,
                                                    decoder_input_ids = decoder_input_ids,
                                                    max_length = 160,
                                                    num_beams = 3,
                                                    repetition_penalty = 5.0,
                                                    length_penalty = 3.0,
                                                    return_dict_in_generate = True,
                                                    output_attentions = True,
                                                    # Set do_sample as True for demo.
                                                    temperature = 2.0,
                                                    do_sample = dosample,
                                                    early_stopping = True)
        # Distributed Training.
        if not self.pretrain and dist.get_rank() == 0 :
            decoded_text = tokenizer.convert_ids_to_tokens(generated_ids.sequences[0])
            out = self.LanguageModel(inputs_embeds = tokens[0].unsqueeze(0),
                                     decoder_input_ids = generated_ids.sequences[0].unsqueeze(0),
                                     output_attentions = True,
                                     return_dict = True)

            # In order to use the model_view function during inference of the T5 model, it is necessary
            # to first assign out.encoder_attentions, out.cross_attentions and out.decoder_attentions
            # to encoder_attentions, cross_attentions and decoder_attentions, respectivly. There are
            # some references to these tricky operations.
            # Reference : https://discuss.huggingface.co/t/error-when-trying-to-visualize-attention-in-t5-model/35350/2
            # Reference : https://huggingface.co/docs/transformers/main_classes/text_generation
            encoder_attentions = out.encoder_attentions
            cross_attentions = out.cross_attentions
            decoder_attentions = out.decoder_attentions
            inputs = {"encoder_attention" : encoder_attentions,
                      "decoder_attention" : decoder_attentions,
                      "cross_attention" : cross_attentions,
                      "encoder_tokens" : len(tokens[0]),
                      "decoder_tokens" : decoded_text,
                      "html_action" : 'return'}

            html_object = model_view(**inputs)
            html_object_head = head_view(**inputs)

            if not os.path.exists(kwargs['result_dir'] + "/HTML/epoch" + str(kwargs['epoch'])):
                os.makedirs(kwargs['result_dir'] + "/HTML/epoch" + str(kwargs['epoch']))

            # Take kwargs['name'][0] as name because its batch size is one in inference dataset
            file_path = kwargs['result_dir'] + "/HTML/epoch" + str(kwargs['epoch']) + "/" + kwargs['video_name'][0]
            with open(file_path + "_model_view.html", 'w') as file :
                file.write(html_object.data)
            with open(file_path + "_head_view.html", 'w') as file :
                file.write(html_object_head.data)

        return generated_ids.sequences, attention_node, attention_graph, max_indices