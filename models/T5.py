from transformers import T5ForConditionalGeneration, AutoConfig
from torch import nn
from visualize_model import model_view, head_view, neuron_view
from models.STAGCN import STA_GCN
from models.transformation import Transformation
from VideoAlignment.model.transformer.transformer import CARL
import torch,os
import torch.distributed as dist
from alignment.alignment import align
# from utils import time_elapsed

class SimpleT5Model(nn.Module):
    def __init__(self,cfg):
        super(SimpleT5Model, self).__init__()
        config = AutoConfig.from_pretrained('t5-base')

        self.cfg = cfg

        self.stagcn = STA_GCN(num_class=1024, 
                                in_channels=6, 
                                residual=True, 
                                dropout=0.5, 
                                num_person=1, 
                                t_kernel_size=9,
                                layout='SMPL',
                                strategy='spatial',
                                hop_size=3,num_att_A=4 )
        in_channel= self.stagcn.output_channel
        if hasattr(self.cfg,"BRANCH") and self.cfg.BRANCH ==2 :
            self.align_module = CARL(cfg.alignment_cfg)
            in_channel += self.align_module.cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
            for param in self.align_module.parameters():
                param.requires_grad = False
        self.transformation = Transformation(cfg,in_channel = in_channel,t5_channel=768)
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-base', config=config) 
    '''
    # - _get_alignment_feature()
    # @ query : (1, T, 22, 512) standard
    # @ key : (batch_size, T, 22, 512) 
    # @ return : batchsize, vertex(22), seq_length, channel (512)
    '''
    # @time_elapsed
    def _get_alignment_feature(self, query, standard,seq_len , names,pretrain = False):
        max_len = max(seq_len)
        # @time_elapsed
        def interpolate_sequence(sequence):
            step = torch.div(max_len, sequence.size(0), rounding_mode='floor')
            new_sequence = torch.zeros(max_len, sequence.size(1))
            for i in range(sequence.size(0)-1):
                new_index = int(i * step)
                new_sequence[new_index, :] = sequence[i, :]
                if i < sequence.size(1) - 1:
                    for j in range(1, int(step)):
                        ratio = j / step
                        interpolated_vector = sequence[i, :] + ratio * (sequence[i+1, :] - sequence[i, :])
                        new_sequence[new_index + j, :] = interpolated_vector
            return new_sequence

        # find the nearest interval between query and key, the interval should be the same across all keypoints, so avg keypoint here.
        avg_query = nn.AvgPool2d((22,1))(query).squeeze(2) ## b , T , 512
        avg_key = nn.AvgPool2d((22,1))(standard).squeeze(2)     ## 1 , T , 512
        
        batch_allignment = None
        for i in range(len(query)):
            video_alignment = None
            current_len = seq_len[i]
            if pretrain == True:
                start_frame = align(avg_query[i,:current_len], avg_key[i,:current_len] , names[i])
            else:
                start_frame = align(avg_query[i,:current_len], avg_key[0] , names[i])

            for j in range(0,22):
                if pretrain == True:
                    subtraction = standard[i,start_frame:start_frame+current_len,j,:] - query[i,:current_len,j,:]
                else:
                    subtraction = standard[i,start_frame:start_frame+current_len,j,:] - query[0,:current_len,j,:] ## Tu x 1 x C
                    
                ## TODO: should add back the original skeletons here
                subtraction = interpolate_sequence(subtraction) ## seq_len x 1 x C
                subtraction = subtraction.unsqueeze(0) ## 1 x seq_len x 1 x C
                if j == 0: video_alignment = subtraction
                else : video_alignment = torch.cat([video_alignment,subtraction],dim=0) # 1,C 1,C  22,C
            
            video_alignment = video_alignment.unsqueeze(0)
            if i == 0: batch_allignment = video_alignment
            else : batch_allignment = torch.cat([batch_allignment,video_alignment],dim=0) # T,22,C 

        # B x 22 x T x C
        batch_allignment = batch_allignment.permute(0,2,1,3)
        return batch_allignment.to(query.device)
    
    def forward(self, names,keypoints,video_mask,standard,seq_len,decoder_input_ids,labels, videos = None,standard_video = None):
        embeddings, _, _= self.stagcn(keypoints)
        if hasattr(self.cfg,"BRANCH") and self.cfg.BRANCH !=0: 
            
            ## BRAHCH1 CONFIG: use STAGCN's embedding to align 2 sequences, fusion, then feed to T5
            if self.cfg.BRANCH == 1: 
                with torch.no_grad():
                    self.stagcn.eval()
                    if self.cfg.TASK.PRETRAIN :
                        standard = keypoints
                        for i in range(0,keypoints.shape[0]):
                            for j in range(1,seq_len[i]):
                                for k in range(0,6):
                                    standard[i][k][j] = keypoints[i][k][0]
                    standard_embedding, _ , _ = self.stagcn(standard)
                aligned_embedding = self._get_alignment_feature(embeddings, standard_embedding,seq_len,names)
            
            ## BRANCH2 CONFIG: use RGB to align input and standard vids, expand to Tu x 22 x embedding_size, fuse with STAGCN's output, then feed to T5
            elif self.cfg.BRANCH == 2: 
                with torch.no_grad():
                    self.align_module.eval()
                    ## B x Tu x self.align_module.cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
                    query_embeddings =  self.align_module(videos,video_masks=video_mask,split='eval') 
                    ## 1 x Ts x self.align_module.cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
                    standard_embedding = self.align_module(standard_video[0].unsqueeze(0),split='eval') 
                concatnated = []
                s_emb = standard_embedding.squeeze(0)
                
                for i,(q_emb) in enumerate((query_embeddings)):
                    ## dummy check if attention mask is correct
                    assert seq_len[i] == torch.sum(video_mask[i]), f"Found seq len equals to {seq_len[i]} but attention mask sums up to {torch.sum(video_mask[i])}" 
                    if seq_len[i] > s_emb.shape[0]:
                        subtraction = s_emb - q_emb[:s_emb.shape[0],:] ## Tu x 22 x C
                    else:
                        start_frame = align(q_emb[:seq_len[i]], s_emb, names)
                        subtraction = s_emb[start_frame:start_frame+seq_len[i],:] - q_emb[:seq_len[i],:] ## Tu x 22 x C
                    subtraction = subtraction.unsqueeze(1).expand(-1,22,-1) ## B x T x 22 x self.align_module.cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
                    ## pad subtraction to match the length of embeddings
                    subtraction = torch.cat([subtraction,torch.zeros(embeddings[i].shape[0]-subtraction.shape[0],22,128).to(subtraction.device)],dim=0)
                    concatnated.append(torch.concat([embeddings[i],subtraction],dim=-1)) ## Tu x 22 x (512+self.align_module.cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE)
                
                aligned_embedding = torch.stack(concatnated,dim=0) # B x Tu x 22 x (512+self.align_module.cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE)
            
            assert aligned_embedding.shape[:-1] == embeddings.shape[:-1], f"Aligned embedding shape {aligned_embedding.shape[:-1]} should be equal to embeddings shape {embeddings.shape[:-1]} except for the last dimension, check if you correctly did padding "
            embeddings = aligned_embedding
        embeddings = self.transformation(embeddings) ## B x T x 768(T5 input shape)
        return self.t5(inputs_embeds=embeddings.contiguous(), attention_mask=video_mask, decoder_input_ids=decoder_input_ids, labels=labels.contiguous())        
    
    def generate(self,**kwargs):
        seq_len = kwargs['seq_len']
        standard = kwargs['standard']
        tokenizer = kwargs['tokenizer']
        input_embeds = kwargs['input_embeds']
        attention_mask = kwargs['attention_mask']
        decoder_input_ids = kwargs['decoder_input_ids']
        names = kwargs['name'] if 'name' in kwargs else None
        embeddings, attention_node , attention_matrix=self.stagcn(input_embeds)
        beam_size = 1
        if hasattr(self.cfg,"BRANCH") and self.cfg.BRANCH !=0: 
                 
            ## BRAHCH1 CONFIG: use STAGCN's embedding to align 2 sequences, fusion, then feed to T5
            if self.cfg.BRANCH == 1: 
                with torch.no_grad():
                    self.stagcn.eval()
                    if self.cfg.TASK.PRETRAIN :
                        standard = input_embeds
                        for i in range(0,input_embeds.shape[0]):
                            for j in range(1,seq_len[i]):
                                for k in range(0,6):
                                    standard[i][k][j] = input_embeds[i][k][0]
                    standard_embedding, _ , _ = self.stagcn(standard)
                aligned_embedding = self._get_alignment_feature(embeddings, standard_embedding,seq_len,names,True)

            ## BRANCH2 CONFIG: use RGB to align input and standard vids, expand to Tu x 22 x embedding_size, fuse with STAGCN's output, then feed to T5
            elif self.cfg.BRANCH == 2: 
                videos = kwargs['videos']                       # B x T x C x H x W
                standard_video = kwargs['standard_video']       # B x T x C x H x W (We only need the first one)
                with torch.no_grad():
                    self.align_module.eval()
                    query_embeddings =  self.align_module(videos,video_masks=attention_mask,split='eval') ## B x Tu x 512
                    standard_embedding = self.align_module(standard_video[0].unsqueeze(0),split='eval') ## 1 x Ts x 512
                concatnated = []
                s_emb = standard_embedding.squeeze(0)
                for i,(q_emb) in enumerate((query_embeddings)):
                    ## dummy check if attention mask is correct
                    assert seq_len[i] == torch.sum(attention_mask[i]), f"Found seq len equals to {seq_len[i]} but attention mask sums up to {torch.sum(attention_mask[i])}" 
                    if seq_len[i] > s_emb.shape[0]:
                        subtraction = s_emb - q_emb[:s_emb.shape[0],:] ## Tu x 22 x C
                    else:
                        start_frame = align(q_emb[:seq_len[i]], s_emb, names)
                        subtraction = s_emb[start_frame:start_frame+seq_len[i],:] - q_emb[:seq_len[i],:] ## Tu x 22 x C
                    subtraction = subtraction.unsqueeze(1).expand(-1,22,-1) ## B x T x 22 x 512
                    ## pad subtraction to match the length of embeddings
                    subtraction = torch.cat([subtraction,torch.zeros(embeddings[i].shape[0]-subtraction.shape[0],22,128).to(subtraction.device)],dim=0)
                    concatnated.append(torch.concat([embeddings[i],subtraction],dim=-1)) ## Tu x 22 x (512+512)
                aligned_embedding = torch.stack(concatnated,dim=0)
            assert aligned_embedding.shape[:-1] == embeddings.shape[:-1], f"Aligned embedding shape {aligned_embedding.shape[:-1]} should be equal to embeddings shape {embeddings.shape[:-1]} except for the last dimension, check if you correctly did padding "
            embeddings = aligned_embedding
        embeddings = self.transformation(embeddings).long()
        generated_ids = self.t5.generate( inputs_embeds=embeddings, 
                                          decoder_input_ids=decoder_input_ids, 
                                          max_length=50,
                                          num_beams=beam_size, 
                                          repetition_penalty=3.5,
                                          length_penalty=1.0,
                                          # temperature=1.5,
                                          return_dict_in_generate=True,
                                          output_attentions=True,   
                                          # Set do_sample=True if you want to demo
                                          do_sample=False,           
                                          early_stopping=True)

        if(not self.cfg.TASK.PRETRAIN) and dist.get_rank() == 0 and (hasattr(self.cfg,'BRANCH') and self.cfg.BRANCH ==1):
            decoded_text = tokenizer.convert_ids_to_tokens(generated_ids.sequences[0])
            ## not using attention mask if we use nodes for the first dimension.
            out = self.t5(inputs_embeds=embeddings[0].unsqueeze(0), decoder_input_ids=generated_ids.sequences[0].unsqueeze(0), 
                            output_attentions=True, return_dict=True)
            '''
            # encoder_attentions is batch_size x num_heads x seq_length x seq_length
            # The first seq_lentgh is the Tx22.
            # 0,22,44,66... 取平均 1,23,45,67... 取平均
            '''
            encoder_attentions = out.encoder_attentions
            cross_attentions = out.cross_attentions

            if (self.cfg.TRANSFORMATION.REDUCTION_POLICY == 'ORIGIN' ):
                encoder_attentions = encoder_attentions.resahpe(out.encoder_attentions.shape[0], out.encoder_attentions.shape[1], -1,22, out.encoder_attentions.shape[3])
                encoder_attentions = encoder_attentions.mean(dim=2)
                
                cross_attentions = cross_attentions.resahpe(out.cross_attentions.shape[0], out.cross_attentions.shape[1], -1,22, out.cross_attentions.shape[3])
                cross_attentions = cross_attentions.mean(dim=2)

            decoder_attentions = out.decoder_attentions
            html_object = model_view(
                                        encoder_attention=encoder_attentions,
                                        decoder_attention=decoder_attentions,
                                        cross_attention=cross_attentions,
                                        encoder_tokens= len(embeddings[0]),
                                        decoder_tokens=decoded_text,
                                        html_action='return'
                                        )

            html_object_head = head_view(
                                        encoder_attention=encoder_attentions,
                                        decoder_attention=decoder_attentions,
                                        cross_attention=cross_attentions,
                                        encoder_tokens= len(embeddings[0]),
                                        decoder_tokens=decoded_text,
                                        html_action='return'
                                        )

            if not os.path.exists(kwargs['result_dir'] + "/HTML/epoch" + str(kwargs['epoch'])):
                os.makedirs(kwargs['result_dir'] + "/HTML/epoch" + str(kwargs['epoch']))

            # @name : kwargs['name'][0] since its batch size is one in inference dataset 
            with open(kwargs['result_dir'] + "/HTML/epoch" + str(kwargs['epoch']) + "/"+ kwargs['name'][0] + "_model_view.html", 'w') as file:
                file.write(html_object.data)
            with open(kwargs['result_dir'] + "/HTML/epoch" + str(kwargs['epoch']) + "/"+ kwargs['name'][0] + "_head_view.html", 'w') as file:
                file.write(html_object_head.data)

        return generated_ids.sequences, attention_node , attention_matrix