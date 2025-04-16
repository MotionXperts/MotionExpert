from transformers import T5ForConditionalGeneration, AutoConfig
from torch import nn
from visualize_model import model_view, head_view
from .STAGCN import STA_GCN
from .Transformation import Transformation
import torch, os
import torch.distributed as dist

class SimpleT5Model(nn.Module) :
    def __init__(self, cfg) :
        super(SimpleT5Model, self).__init__()
        config = AutoConfig.from_pretrained('t5-base')
        if cfg.TASK.SPORT == "Skating" :
            stagcn_lora_config = {"bias" : "none",
                                  "r" : 32,
                                  "lora_alpha" : 64,
                                  "lora_dropout" : 0.1}
            transformation_lora_config = {"bias" : "none",
                                          "r" : 32,
                                          "lora_alpha" : 64,
                                          "lora_dropout" : 0.1}
        elif cfg.TASK.SPORT == "Boxing" :
            stagcn_lora_config = {"bias" : "none",
                                  "r" : 64,
                                  "lora_alpha" : 128,
                                  "lora_dropout" : 0.5}
            transformation_lora_config = {"bias" : "none",
                                          "r" : 64,
                                          "lora_alpha" : 128,
                                          "lora_dropout" : 0.5}
        self.cfg = cfg
        self.stagcn = STA_GCN(num_class = 1024, in_channels = 6, residual = True, dropout = 0.5, t_kernel_size = 9, layout = 'SMPL',
                              strategy = 'spatial', hop_size = 3, num_att_A = 4, PRETRAIN_SETTING = self.cfg.TASK.PRETRAIN_SETTING,
                              PRETRAIN = cfg.TASK.PRETRAIN, lora_config = stagcn_lora_config)

        if self.cfg.TASK.PRETRAIN_DIFFERENCE or self.cfg.TASK.DIFFERENCE_TYPE == 'RGB' :
            in_channel = 1024
        else :
            in_channel = self.stagcn.output_channel

        self.transformation = Transformation(cfg, in_channel, t5_channel = 768, lora_config = transformation_lora_config)

        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-base', config = config)
        '''
        lora_config = LoraConfig(r = 32,
                                 lora_alpha = 32,
                                 target_modules = ["q", "v"],
                                 lora_dropout = 0.1,
                                 bias = "none",
                                 task_type = TaskType.SEQ_2_SEQ_LM)
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-base', config = config)
        self.t5 = get_peft_model(self.t5, lora_config)
        '''
        self.RGB_lifting = nn.Linear(128, 512)
        # Distributed Training.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_difference_feature(self, user, standard, DIFFERENCE_SETTING) :
        if DIFFERENCE_SETTING == 'Subtraction' :
            batch_diff = user-standard 

        # DIFFERENCE_SETTING : Padding.
        else :
            batch_diff = torch.zeros(user.shape[0], user.shape[1], user.shape[2], user.shape[3])

        # The dimension of batch_diff is [batchsize, seq_length, vertex(22), channel (512)].
        return batch_diff

    def get_standard_feature(self, keypoints, seq_len, pretrain, standard) :
        standard_input_embedding = keypoints
        # Batch size.
        for i in range(0, keypoints.shape[0]) :
            # Number of frames.
            for j in range(1, seq_len[i]) :
                # joints coordinates (3) + bones coordinates (3).
                for k in range(0, 6) :
                    # Copy the 22 joints of every coordinate.
                    standard_input_embedding[i][k][j] = keypoints[i][k][0]
        return standard_input_embedding

    def get_transformation_feature(self, stagcn_embedding, difference_embedding, PRETRAIN_DIFFERENCE) :
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if PRETRAIN_DIFFERENCE :
            concatenate_embedding = torch.cat([stagcn_embedding, difference_embedding.to(device)], dim = -1)
            transform_embedding, max_indices = self.transformation(concatenate_embedding)
        elif self.cfg.TASK.DIFFERENCE_TYPE == 'RGB' :
            difference_embedding = self.RGB_lifting(difference_embedding)
            difference_embedding = difference_embedding[:, :(stagcn_embedding).shape[1], :, :]
            concatenate_embedding = torch.cat([stagcn_embedding, difference_embedding.to(device)], dim = -1)
            transform_embedding, max_indices = self.transformation(concatenate_embedding)
        else :
            transform_embedding, max_indices = self.transformation(stagcn_embedding)
        return transform_embedding, max_indices

    def forward(self,**kwargs) :
        video_name = kwargs['video_name']
        input_embedding = kwargs['input_embedding']
        input_embedding_mask = kwargs['input_embedding_mask']
        standard = kwargs['standard']
        seq_len = kwargs['seq_len']
        decoder_input_ids = kwargs['decoder_input_ids']
        labels = kwargs['labels']
        tokenizer = kwargs['tokenizer']
        subtraction = kwargs['subtraction']
        self.stagcn.train()
        stagcn_embedding, _, _ = self.stagcn(input_embedding)
        if self.cfg.TASK.PRETRAIN_DIFFERENCE :
            with torch.no_grad() :
                self.stagcn.eval()
                if self.cfg.TASK.PRETRAIN :
                    standard_input_embedding = self.get_standard_feature(input_embedding, seq_len, self.cfg.TASK.PRETRAIN, None)
                    standard_embedding, _ , _ = self.stagcn(standard_input_embedding)
                else :
                    standard = standard.permute(0, 2, 1, 3)
                    standard_embedding, _ , _ = self.stagcn(standard)

            difference_embedding = self.get_difference_feature(stagcn_embedding, standard_embedding, self.cfg.TASK.DIFFERENCE_SETTING)
            transform_embedding, max_indices = self.get_transformation_feature(stagcn_embedding,
                                                                               difference_embedding,
                                                                               self.cfg.TASK.PRETRAIN_DIFFERENCE)

        elif self.cfg.TASK.DIFFERENCE_TYPE == 'RGB' :
            # The dimension of difference_embedding is [batch size, seq length, 1, 128]
            difference_embedding = subtraction
            # The dimension of difference_embedding becomes [batch size, seq length, 22, 128]
            difference_embedding = subtraction.unsqueeze(2).expand(-1, -1, 22, -1)
            transform_embedding, max_indices = self.get_transformation_feature(stagcn_embedding,
                                                                               difference_embedding,
                                                                               self.cfg.TASK.PRETRAIN_DIFFERENCE)
        else :
            transform_embedding, max_indices = self.get_transformation_feature(stagcn_embedding,
                                                                               None,
                                                                               self.cfg.TASK.PRETRAIN_DIFFERENCE)

        logits = self.t5(inputs_embeds = transform_embedding.contiguous(),
                         attention_mask = input_embedding_mask,
                         decoder_input_ids = decoder_input_ids).logits
        argmax = torch.argmax(logits, dim=-1)
        decoded_text = tokenizer.decode(argmax[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return self.t5(inputs_embeds = transform_embedding.contiguous(),
                       attention_mask = input_embedding_mask,
                       decoder_input_ids = decoder_input_ids,
                       labels = labels.contiguous())

    def generate(self,**kwargs) :
        video_name = kwargs['video_name']
        input_embedding = kwargs['input_embedding']
        input_embedding_mask = kwargs['input_embedding_mask']
        standard = kwargs['standard']
        seq_len = kwargs['seq_len']
        decoder_input_ids = kwargs['decoder_input_ids']
        tokenizer = kwargs['tokenizer']
        subtraction = kwargs['subtraction']
        with torch.no_grad() :
            self.stagcn.eval()
            stagcn_embedding, attention_node, attention_matrix = self.stagcn(input_embedding)
            if self.cfg.TASK.PRETRAIN_DIFFERENCE :
                if self.cfg.TASK.PRETRAIN :
                    standard_input_embedding = self.get_standard_feature(input_embedding, seq_len, self.cfg.TASK.PRETRAIN, None)
                    standard_embedding, _ , _ = self.stagcn(standard_input_embedding)
                else :
                    standard = standard.permute(0, 2, 1, 3)
                    standard_embedding, _ , _ = self.stagcn(standard)

                difference_embedding = self.get_difference_feature(stagcn_embedding, standard_embedding, self.cfg.TASK.DIFFERENCE_SETTING)
                transform_embedding, max_indices = self.get_transformation_feature(stagcn_embedding,
                                                                                   difference_embedding,
                                                                                   self.cfg.TASK.PRETRAIN_DIFFERENCE)
            elif self.cfg.TASK.DIFFERENCE_TYPE== 'RGB' :
                # The dimension of difference_embedding is [batch size, seq length, 1, 128].
                difference_embedding = subtraction
                # The dimension of difference_embedding becomes [batch size, seq length, 22, 128].
                difference_embedding = subtraction.unsqueeze(2).expand(-1, -1, 22, -1)
                transform_embedding, max_indices = self.get_transformation_feature(stagcn_embedding,
                                                                                   difference_embedding,
                                                                                   self.cfg.TASK.PRETRAIN_DIFFERENCE)
            else :
                transform_embedding, max_indices = self.get_transformation_feature(stagcn_embedding,
                                                                                   None,
                                                                                   self.cfg.TASK.PRETRAIN_DIFFERENCE)
        generated_ids = self.t5.generate(inputs_embeds = transform_embedding,
                                         attention_mask = input_embedding_mask,
                                         decoder_input_ids = decoder_input_ids,
                                         max_length = 160,
                                         num_beams = 3,
                                         repetition_penalty = 2.5,
                                         length_penalty = 1.0,
                                         return_dict_in_generate = True,
                                         output_attentions = True,
                                         # Set do_sample as True for demo.
                                         do_sample = False,
                                         early_stopping = True)
        # Distributed Training.
        if(not self.cfg.TASK.PRETRAIN) and dist.get_rank() == 0 :
            decoded_text = tokenizer.convert_ids_to_tokens(generated_ids.sequences[0])
            out = self.t5(inputs_embeds = transform_embedding[0].unsqueeze(0),
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
                      "encoder_tokens" : len(transform_embedding[0]),
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

        return generated_ids.sequences, attention_node, attention_matrix, max_indices