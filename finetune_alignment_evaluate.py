import torch
import torch.distributed as dist
from utils.parser import parse_args
from cider import readJSON, readPickle, getGTCaptions, BLEUScore, CIDERScore
from dataloaders import construct_dataloader
from models.T5 import SimpleT5Model
from transformers import AutoTokenizer, AdamW
from torch.utils.tensorboard import SummaryWriter
from models import load_checkpoint
import os,json
os.environ['TOKENIZERS_PARALLELISM'] = "false"
from tqdm import tqdm
import numpy as np
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

def eval(args,eval_dataloader, model, tokenizer,epoch,summary_writer,sanity_check=False,store=None,name_list = None,logger=None):       
    assert logger is not None, "Please provide logger object"
    model.eval()
    loss_list = [] 
    att_node_results = {}
    att_A_results = {}  
    prompt =  "Motion Instruction : "
    with torch.no_grad():
        if dist.get_rank() == 0:
            eval_dataloader = tqdm(eval_dataloader, total=len(eval_dataloader), desc='Evaluating')
        for index,batch in enumerate(eval_dataloader):
            video_names = batch['video_name']
            video_mask_batch = batch['video_mask'].to(model.device) ## is it needed?
            skeleton_feature = batch['keypoints'].to(model.device)
            rgb_feature = batch['video'].to(model.device)
            
            label_batch = batch['label']
            decoder_input_ids = tokenizer([prompt],
                                            return_tensors="pt", 
                                            padding=True, 
                                            truncation=True, 
                                            max_length=50,
                                            add_special_tokens=False)['input_ids']

            decoder_input_ids = decoder_input_ids.repeat(skeleton_feature.shape[0], 1).to(skeleton_feature.device)
            tgt_batch = tokenizer(label_batch, return_tensors="pt", padding="max_length", truncation=True, max_length=50)['input_ids'].to(src_batch.device)
            tgt_input = tgt_batch[:, :-1]
            tgt_label = tgt_batch[:, 1:]
            input = {   "decoder_input_ids": decoder_input_ids,
                        "name": video_names,
                        "skeleton_feature":skeleton_feature,
                        "rgb_feature":rgb_feature}
            with torch.cuda.amp.autocast():
                generated_ids , att_node , att_A = model.module.generate(**input)
                loss = model.module(skeleton_feature=skeleton_feature.contiguous(), 
                            rgb_feature=rgb_feature.contiguous(),
                            video_mask= video_mask_batch.contiguous(),
                            decoder_input_ids=tgt_input.contiguous(),         # text
                            labels=tgt_label.contiguous()).loss 

            loss[torch.isnan(loss)] = 0
            dist.all_reduce(loss, async_op=False)
            reduced_loss = loss / dist.get_world_size()
            loss_list.append(reduced_loss.detach().cpu())
            for name, gen_id,label in zip(video_names, generated_ids,label_batch):
                
                decoded_text = tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True).split(prompt)
                if len(decoded_text) > 1:
                    decoded_text = decoded_text[1].strip()
                else:
                    decoded_text = ""
                store.set(name,decoded_text)
            for name, att_node in zip(video_names, att_node):
                att_node_results[name] = att_node.cpu().numpy().tolist()
                # store.set(f'att_node_{name}',att_node.cpu().numpy().tolist()) ## store.set only accepts strings 
            for name, att_A in zip(video_names, att_A):
                att_A_results[name] = att_A.cpu().numpy().tolist()
                # store.set(f'att_A_{name}',att_A.cpu().numpy().tolist())

            if dist.get_rank() == 0:
                eval_dataloader.set_postfix({
                    'loss': np.mean(loss_list),
                })

            if sanity_check and index > 4:
                return
                
    if dist.get_rank() == 0:
        summary_writer.add_scalar('eval/loss', np.mean(loss_list), epoch)

        results = {}
        
        ## iterate over name_list and get values from store
        for name in name_list:
            results[name] = store.get(name).decode('utf-8')

        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        with open(args.result_dir+'/results_epoch'+str(epoch)+'.json', 'w') as f:
            json.dump(results, f,indent = 1)
        with open(args.result_dir+'/att_node_results_epoch'+str(epoch)+'.json', 'w') as f:
            json.dump(att_node_results, f)
        with open(args.result_dir+'/att_A_results_epoch'+str(epoch)+'.json', 'w') as f:
            json.dump(att_A_results, f)

        predictions = readJSON(args.result_dir+'/results_epoch'+str(epoch)+'.json')
        annotations = readPickle(args.test_data)
        
        gts = getGTCaptions(annotations)
        new_gts = {}
        for name in results:
            new_gts[name] = gts[name]
            # summary_writer.add_text('eval/pred', name +": " + results[name], (epoch+1)*index) ## no index here, think of workarounds
            # summary_writer.add_text('eval/label', name + ": " +  new_gts[name], (epoch+1)*index)
        gts = new_gts
        # Check predictions content is correct
        assert type(predictions) is dict
        assert set(predictions.keys()) == set(gts.keys())
        assert all([type(pred) is str for pred in predictions.values()])
        # CIDErScore
        cider_score = CIDERScore()(predictions, gts)
        bleu_score = BLEUScore()(predictions, gts)
        logger.info(f"Epoch {epoch}: Loss {np.mean(loss_list)}")
        logger.info(f"Epoch {epoch}: CIDEr: {cider_score}")
        logger.info(f"Epoch {epoch}: BLEU: {bleu_score}")

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                                filename=os.path.join(args.out_dir,'stdout.log'))
    lr=1e-3
    tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)
    model = SimpleT5Model()
    import pickle

    ## maintain a name list in main process
    with open(args.test_data, 'rb') as f:
        data = pickle.load(f)
    name_list = []
    for d in data:
        name_list.append(d['video_name'])

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    model = model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                        output_device=args.local_rank)
    optimizer = AdamW(model.parameters(), lr=lr)
    summary_writer = SummaryWriter(os.path.join(args.out_dir, 'train_logs'))

    if dist.get_rank() == 0:
        store = dist.TCPStore("127.0.0.1", 1234, dist.get_world_size(), True, timedelta(seconds=30))
    else:
        store = dist.TCPStore("127.0.0.1", 1234, dist.get_world_size(), False, timedelta(seconds=30))
    val_dataloader = construct_dataloader('val',args.test_data,args.finetune,args.bs)
    

    epoch = load_checkpoint(args,model,optimizer,args.ckpt)
    summary_writer = SummaryWriter()
    eval(args,val_dataloader, model, tokenizer,epoch,summary_writer,sanity_check=False,
                    store=store,name_list=name_list,logger=logger)
    dist.destroy_process_group()