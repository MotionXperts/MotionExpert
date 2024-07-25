from transformers import T5ForConditionalGeneration, AutoConfig
from torch import nn
from visualize_model import model_view, head_view, neuron_view
from .STAGCN import STA_GCN
from .Transformation import Transformation
from VideoAlignment.model.transformer.transformer import CARL
import torch,os
import torch.distributed as dist
from alignment.alignment import align
import sys
from torch.nn.utils.rnn import pad_sequence
class SimpleT5Model(nn.Module):
    def __init__(self,cfg):
        super(SimpleT5Model, self).__init__()
        config = AutoConfig.from_pretrained('t5-base')

        self.cfg    = cfg
        self.stagcn = STA_GCN( num_class=1024, in_channels=6, residual=True, dropout=0.5, num_person=1, t_kernel_size=9, layout='SMPL', strategy='spatial', hop_size=3, num_att_A=4 )
        
        in_channel= self.stagcn.output_channel
        '''
        if hasattr(self.cfg,"BRANCH") and self.cfg.BRANCH == 2 :
            self.align_module = CARL(cfg.alignment_cfg)
            in_channel += self.align_module.cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
            for param in self.align_module.parameters():
                param.requires_grad = False
        '''
        self.transformation = Transformation(cfg,in_channel = 1024,t5_channel=768)
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-base', config=config) 
    
    '''
    # - get_alignment_feature()
    # @ standard : (1, T, 22, 512) 
    # @ user : (batch_size, T, 22, 512) 
    # @ return : batchsize, vertex(22), seq_length, channel (512)
    '''
    def get_start_and_end(self, user, standard, seq_len , names):      
        # Find the nearest interval between user and standard, the interval should be the same across all keypoints, so avg keypoint here.
        avg_user    = nn.AvgPool2d((22,1))(user).squeeze(2)      ## b , T , 512
        avg_std     = nn.AvgPool2d((22,1))(standard).squeeze(2)  ## 1 , T , 512
        batch_start = []
        batch_end = []
        for i in range(len(user)):
         
            cur_len = seq_len[i]
     
            start_frame = align(avg_user[i,:cur_len], avg_std[0] , names[i])
            batch_start.append(start_frame)
            
            standard_length = standard[0].size(0)

            if (cur_len <= standard_length):
                batch_end.append(start_frame + cur_len)
            else:         
                # It is correct           
                # batch_end.append(start_frame + standard_length)
                # FIXME: WRONG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                batch_end.append(start_frame + cur_len)
        
        return batch_start, batch_end
    
    def get_difference_feature(self, user, standard, seq_len, pretrain,std_start,std_end):      
        batch_diff = []
        for i in range(len(user)):
            video_diff = None          
            # Calculate the difference between user and standard
            if pretrain == True:    
                standard_ID = i
                start_frame = 0
                end_frame = seq_len[i]
                video_diff = user[i,:seq_len[i],:,:] - standard[standard_ID,start_frame:end_frame, :, :]  
            else :                  
                standard_ID = 0
                start_frame = std_start[i]
                end_frame = std_end[i]
                if (seq_len[i] <= standard[0].size(0)):
                    # It is correct 
                    # video_diff = user[i,:,:,:] - standard[standard_ID, start_frame:end_frame,:,:]
                    # FIXME: WRONG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    video_diff = user[i,:,:,:] 
                else: 
                    # It is correct 
                    # video_diff = user[i,start_frame:end_frame,:,:] - standard[standard_ID, :,:,:]
                    # FIXME: WRONG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    video_diff = user[i,:,:,:] 
                    
            batch_diff.append(video_diff)
   
        batch_diff = pad_sequence(batch_diff,batch_first=True,padding_value=0) 
        return batch_diff

    def get_standard_feature(self,keypoints,seq_len):
        standard = keypoints
        for i in range(0,keypoints.shape[0]):
            for j in range(1,seq_len[i]):
                for k in range(0,6):
                    standard[i][k][j] = keypoints[i][k][0]
        return standard
    
    def get_transforma_feature(self, stagcn_embedding, difference_embedding):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        concatenate_embedding = torch.cat([stagcn_embedding,difference_embedding.to(device)],dim=-1)
        transform_embedding = self.transformation(concatenate_embedding)
        return transform_embedding

    def forward(self,**kwargs):
        video_name           = kwargs['video_name']
        input_embedding      = kwargs['input_embedding']
        input_embedding_mask = kwargs['input_embedding_mask']
        standard             = kwargs['standard']
        seq_len              = kwargs['seq_len']
        decoder_input_ids    = kwargs['decoder_input_ids']
        labels               = kwargs['labels']
        self.stagcn.train()
        stagcn_embedding, _, _ = self.stagcn(input_embedding)
        ## BRANCH 0 CONFIG: STAGCN
        if hasattr(self.cfg,"BRANCH") and self.cfg.BRANCH != 0: 
            ## BRAHCH 1 CONFIG: STAGCN + Alignment
            if self.cfg.BRANCH == 1: 
                with torch.no_grad():
                    self.stagcn.eval()
                    if self.cfg.TASK.PRETRAIN :standard = self.get_standard_feature(input_embedding,seq_len)
                    standard_embedding, _ , _ = self.stagcn(standard)
                

            if self.cfg.TASK.PRETRAIN :
                difference_embedding = self.get_difference_feature(stagcn_embedding, standard_embedding, seq_len, self.cfg.TASK.PRETRAIN,std_start = None,std_end = None)
                assert difference_embedding.shape[:-1] == stagcn_embedding.shape[:-1], f"Difference embedding shape {difference_embedding.shape[:-1]} should be equal to embeddings shape {difference_embedding.shape[:-1]} except for the last dimension, check if you correctly did padding "
            else : 
                start_frame, end_frame = self.get_start_and_end(stagcn_embedding, standard_embedding, seq_len, video_name)
                difference_embedding = self.get_difference_feature(stagcn_embedding, standard_embedding, seq_len, self.cfg.TASK.PRETRAIN,std_start = start_frame,std_end = end_frame)
            ## BRANCH 2 CONFIG: use RGB to align input and standard vids, expand to Tu x 22 x embedding_size, fuse with STAGCN's output, then feed to T5
            '''
            elif self.cfg.BRANCH == 2: 
                concatenation=[]
                for (b,s) in zip(embeddings,subtraction):
                    s = s.unsqueeze(1).expand(-1,22,-1)
                    s = torch.cat([s,torch.zeros(b.shape[0]-s.shape[0],22,128).to(subtraction.device)],dim=0)
                    concatenation.append(torch.concat([b,s],dim=-1))
                aligned_embedding = torch.stack(concatenation,dim=0) ## B x T x 22 x (512+512)
            '''
        transform_embedding = self.get_transforma_feature(stagcn_embedding,difference_embedding)
        
        return self.t5(inputs_embeds=transform_embedding.contiguous(), attention_mask=input_embedding_mask, decoder_input_ids=decoder_input_ids, labels=labels.contiguous())        
    
    def generate(self,**kwargs):
        video_name = kwargs['video_name']
        input_embedding = kwargs['input_embedding']
        input_embedding_mask = kwargs['input_embedding_mask']
        standard = kwargs['standard']
        seq_len = kwargs['seq_len']
        decoder_input_ids = kwargs['decoder_input_ids']
        tokenizer = kwargs['tokenizer']
        '''
        #videos = kwargs['videos'] if 'videos' in kwargs else None
        #standard_video = kwargs['standard_video'] if 'standard_video' in kwargs else None
        #subtraction = kwargs['subtraction'] if 'subtraction' in kwargs else None
        '''
        stagcn_embedding, attention_node, attention_matrix = self.stagcn(input_embedding)
        if hasattr(self.cfg,"BRANCH") and self.cfg.BRANCH !=0: 
            ## BRAHCH 1 CONFIG: STAGCN
            if self.cfg.BRANCH == 1: 
                with torch.no_grad():
                    self.stagcn.eval()
                    if self.cfg.TASK.PRETRAIN : standard = self.get_standard_feature(input_embedding,seq_len)
                    standard_embedding, _ , _ = self.stagcn(standard)

                if self.cfg.TASK.PRETRAIN :
                    difference_embedding = self.get_difference_feature(stagcn_embedding, standard_embedding, seq_len, self.cfg.TASK.PRETRAIN,std_start = None,std_end = None)
                    assert difference_embedding.shape[:-1] == stagcn_embedding.shape[:-1], f"Difference embedding shape {difference_embedding.shape[:-1]} should be equal to embeddings shape {stagcn_embedding.shape[:-1]} except for the last dimension, check if you correctly did padding "
                else : 
                    start_frame, end_frame = self.get_start_and_end(stagcn_embedding, standard_embedding, seq_len, video_name)
                    difference_embedding = self.get_difference_feature(stagcn_embedding, standard_embedding, seq_len, self.cfg.TASK.PRETRAIN,std_start = start_frame,std_end = end_frame)
            ## BRANCH2 CONFIG: use RGB to align input and standard vids, expand to Tu x 22 x embedding_size, fuse with STAGCN's output, then feed to T5 
            '''
            elif self.cfg.BRANCH == 2: 
                concatenation=[]
                for b,s in zip(embeddings,subtraction):
                    s = s.unsqueeze(1).expand(-1,22,-1)
                    s = torch.cat([s,torch.zeros(b.shape[0]-s.shape[0],22,128).to(subtraction.device)],dim=0)
                    concatenation.append(torch.concat([b,s],dim=-1))
                aligned_embedding = torch.stack(concatenation,dim=0) ## B x T x 22 x (512+512)
            '''
        transform_embedding = self.get_transforma_feature(stagcn_embedding,difference_embedding)
        generated_ids = self.t5.generate( inputs_embeds             = transform_embedding, 
                                          attention_mask            = input_embedding_mask,
                                          decoder_input_ids         = decoder_input_ids, 
                                          max_length                = 50,
                                          num_beams                 = 3, 
                                          repetition_penalty        = 2.5,
                                          length_penalty            = 1.0,
                                          return_dict_in_generate   = True,
                                          output_attentions         = True,   
                                          # Set do_sample           = True if you want to demo
                                          do_sample                 = False,           
                                          early_stopping            = True)
        if(not self.cfg.TASK.PRETRAIN) and dist.get_rank() == 0 and (hasattr(self.cfg,'BRANCH') and self.cfg.BRANCH ==1):
            decoded_text = tokenizer.convert_ids_to_tokens(generated_ids.sequences[0])
            out = self.t5(  inputs_embeds       = transform_embedding[0].unsqueeze(0), 
                            decoder_input_ids   = generated_ids.sequences[0].unsqueeze(0), 
                            output_attentions   = True, 
                            return_dict         = True)

            encoder_attentions  = out.encoder_attentions
            cross_attentions    = out.cross_attentions
            decoder_attentions  = out.decoder_attentions
            inputs = {  "encoder_attention" : encoder_attentions,
                        "decoder_attention" : decoder_attentions,
                        "cross_attention"   : cross_attentions,
                        "encoder_tokens"    : len(transform_embedding[0]),
                        "decoder_tokens"    : decoded_text,
                        "html_action"       : 'return'}

            html_object         = model_view(**inputs)
            html_object_head    = head_view(**inputs)

            if not os.path.exists(kwargs['result_dir'] + "/HTML/epoch" + str(kwargs['epoch'])):
                os.makedirs(kwargs['result_dir'] + "/HTML/epoch" + str(kwargs['epoch']))

            ''' @name : kwargs['name'][0] since its batch size is one in inference dataset '''
            with open(kwargs['result_dir'] + "/HTML/epoch" + str(kwargs['epoch']) + "/"+ kwargs['video_name'][0] + "_model_view.html", 'w') as file:
                file.write(html_object.data)
            with open(kwargs['result_dir'] + "/HTML/epoch" + str(kwargs['epoch']) + "/"+ kwargs['video_name'][0] + "_head_view.html", 'w') as file:
                file.write(html_object_head.data)

        return generated_ids.sequences, attention_node , attention_matrix