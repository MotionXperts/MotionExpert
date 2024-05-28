from models.T5 import SimpleT5Model
from models.transformation import Transformation
import torch.nn
from VideoAlignment.model.transformer.transformer import CARL

class AlignedT5(nn.Module):
    def __init__(self,cfg):
        super(AlignedT5, self).__init__()
        model = SimpleT5Model()
        self.stagcn = model.stagcn[:-1]
        self.align_module = CARL(cfg)

        in_channel = (self.stagcn[-1].out_channels + self.align_module.cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE)
        self.transformation = Transformation(in_channel=in_channel)
        self.T5 = model.t5
        for param in self.align_module.parameters():
            param.requires_grad = False

    def forward(self,skeleton_feature,rgb_feature,video_mask,decoder_input_ids,labels):
        self.align_module.eval()
        with torch.no_grad():
            aligned_embbeding = self.align_module(rgb_feature,split="val")
        skeleton_embedding = self.stagcn(skeleton_feature)
        
        ## fusion
        fusion = torch.cat([aligned_embbeding,skeleton_embedding],dim=-1) ## B , T , in_channel
        transformed = self.transformation(fusion)
        return self.T5(inputs_embeds=transformed, attention_mask=video_mask, decoder_input_ids=decoder_input_ids, labels=labels)
    def generate(self,**kwargs):
        decoder_input_ids = kwargs['decoder_input_ids']
        skeleton_feature = kwargs['skeleton_feature']
        rgb_feature = kwargs['rgb_feature']

        aligned_embedding = self.align_module(rgb_feature,split="val")
        embedding, attention_node , attention_matrix=self.stagcn(skeleton_feature)
        
        fusion = torch.cat([aligned_embedding,embedding],dim=-1)
        transfromed = self.transformation(fusion)

        beam_size = 2
        embedding = transfromed.long()
        generated_ids = self.t5.generate( inputs_embeds=transfromed, 
                                          decoder_input_ids=decoder_input_ids, 
                                          max_length=50,
                                          num_beams=beam_size, 
                                          repetition_penalty=3.5,
                                          length_penalty=1.0,
                                          temperature=1.5,
                                          return_dict_in_generate=True,
                                          output_attentions=True,   
                                          do_sample=True,    
                                          early_stopping=True)

        return generated_ids.sequences, attention_node , attention_matrix
        