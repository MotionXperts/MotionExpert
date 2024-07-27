import os,json,sys
## add videoalignment to sys path
sys.path.append(os.path.join(os.getcwd(),'VideoAlignment'))
import torch
import torch.distributed as dist
from pytorch_lightning.utilities.seed import seed_everything
from utils.parser import parse_args,load_config
from cider import readJSON, readPickle, getGTCaptions, BLEUScore, CIDERScore
from dataloaders import construct_dataloader
from models.T5 import SimpleT5Model
from transformers import AutoTokenizer, AdamW
from torch.utils.tensorboard import SummaryWriter
from models import load_checkpoint
os.environ['TOKENIZERS_PARALLELISM'] = "false"
from tqdm import tqdm
import numpy as np
from datetime import timedelta
import logging
import pickle
from bert_score import score
from nlgmetricverse import NLGMetricverse,load_metric

logger = logging.getLogger(__name__)

def eval(cfg,eval_dataloader, model,epoch,summary_writer,sanity_check=False,store=None,name_list = None,logger=None):       
    
    assert logger is not None, "Please provide logger object"
    Tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)
    model.eval()
    model = model.cuda()
    loss_list = [] 
    att_node_results = {}
    att_A_results = {}  
    prompt =  "Motion Instruction : " if cfg.TASK.PRETRAIN else "Motion Description : "
    with torch.no_grad():
        # Distributed Training
        if dist.get_rank() == 0:
            eval_dataloader = tqdm(eval_dataloader, total=len(eval_dataloader), desc='Evaluating')
        for index,batch in enumerate(eval_dataloader):
            (video_name,src_batch,keypoints_mask_batch,standard,seq_len,label_batch) = batch
            # If evaluating multiple checkpoints, dont do inference but directly load the result jsons
            if cfg.args.eval_multi: 
                break
            decoder_input_ids = Tokenizer(  [prompt],
                                            return_tensors="pt", 
                                            padding=True, 
                                            truncation=True, 
                                            max_length=50,
                                            add_special_tokens=False)['input_ids']
            decoder_input_ids = decoder_input_ids.repeat(src_batch.shape[0], 1).to(src_batch.device)
            tgt_batch = Tokenizer(label_batch, return_tensors="pt", padding="max_length", truncation=True, max_length=50)['input_ids'].to(src_batch.device)
            tgt_input = tgt_batch[:, :-1]
            tgt_label = tgt_batch[:, 1:]
            inputs = {  "video_name"            : video_name,
                        "input_embedding"       : src_batch.to(model.device),
                        "input_embedding_mask"  : keypoints_mask_batch.to(model.device),
                        "standard"              : standard.to(model.device),
                        "seq_len"               : seq_len.to(model.device),
                        "decoder_input_ids"     : decoder_input_ids.to(model.device),
                        "tokenizer"             : Tokenizer,
                        # For visualizing attention
                        "result_dir"            : cfg.LOGDIR,
                        "epoch"                 : epoch }
            
            with torch.cuda.amp.autocast():
                seed_everything(42) 

                generated_ids , att_node , att_A = model.module.generate(**inputs)

                if (hasattr(cfg,'BRANCH') and cfg.BRANCH == 1) or (cfg.TRANSFORMATION.REDUCTION_POLICY == 'TIME_POOL'): 
                    inputs['labels'] = tgt_label.to(model.device)                              
                    inputs['decoder_input_ids'] = tgt_input.to(model.device)
                    loss = model(**inputs).loss
            
            loss[torch.isnan(loss)] = 0
            # Distributed Training
            dist.all_reduce(loss, async_op=False)
            reduced_loss = loss / dist.get_world_size()
            loss_list.append(reduced_loss.detach().cpu())

            for name, gen_id,label in zip(video_name, generated_ids,label_batch):
                decoded_text = Tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True).split(prompt)
                if len(decoded_text) > 1:
                    decoded_text = decoded_text[1].strip()
                else:
                    decoded_text = ""

                # Distributed Training
                store.set(name,decoded_text)
            for name, att_node in zip(video_name, att_node):
                att_node_results[name] = att_node.cpu().numpy().tolist()
            for name, att_A in zip(video_name, att_A):
                att_A_results[name] = att_A.cpu().numpy().tolist()

            if dist.get_rank() == 0:
                eval_dataloader.set_postfix({'loss': np.mean(loss_list),})
            if sanity_check and index > 4:
                return
            
    # Distributed Training
    if dist.get_rank() == 0:
        summary_writer.add_scalar('eval/loss', np.mean(loss_list), epoch)

        results = {}
        for name in name_list:
            # Distributed Training
            results[name] = store.get(name).decode('utf-8')
        
        if not cfg.args.eval_multi:
            with open(cfg.JSONDIR+'/results_epoch'+str(epoch)+'.json', 'w') as f:
                json.dump(results, f,indent = 1)
            with open(cfg.JSONDIR+'/att_node_results_epoch'+str(epoch)+'.json', 'w') as f:
                json.dump(att_node_results, f)
            with open(cfg.JSONDIR+'/att_A_results_epoch'+str(epoch)+'.json', 'w') as f:
                json.dump(att_A_results, f)
        
        if cfg.args.eval_multi:
            predictions = readJSON(cfg.JSONDIR+'/results_epoch'+str(epoch-1)+'.json')
        else:
            predictions = readJSON(cfg.JSONDIR+'/results_epoch'+str(epoch)+'.json')
        
        annotations = readPickle(cfg.DATA.TEST)
        
        gts = getGTCaptions(annotations)
        new_gts = {}
        for name in results:
            new_gts[name] = gts[name]
            # summary_writer.add_text('eval/pred', name +": " + results[name], (epoch+1)*index) ## no index here, think of workarounds
            # summary_writer.add_text('eval/label', name + ": " +  new_gts[name], (epoch+1)*index)
        gts = new_gts
        # Check predictions content is correct
        assert type(predictions) is dict, f"Predictions should be a dictionary but got {type(predictions)}"
        print("Predictions keys: ",predictions.keys())
        print("GTS keys: ",gts.keys())
        assert len(predictions.keys()) == len(gts.keys()), f"Predictions keys len should be same as gts keys len, but got {len(predictions.keys())} and {len(gts.keys())}" 
        assert all([type(pred) is str for pred in predictions.values()])

        # Calculate scores
        metrics = [
            load_metric("bleu",resulting_name="bleu_1",compute_kwargs={"max_order":1}),
            load_metric("bleu",resulting_name="bleu_4",compute_kwargs={"max_order":4}),
            load_metric("rouge"),
            load_metric("cider"),
        ]
        Evaluator = NLGMetricverse(metrics)
        # Need to convert predictions and gts to list to fit with bert_score
        # Make sure predictions and gts are in the same order
        predictions = dict(sorted(predictions.items()))
        # Del standard in gts since there is no standard in predictions
        if 'standard' in gts: del gts['standard']
        gts = dict(sorted(gts.items()))

        predictions = list(predictions.values())
        gts = list(gts.values())
        scores = Evaluator(predictions=predictions,references=gts)
        results = {}
        results["bleu_1"]   = scores["bleu_1"]['score']
        results["bleu_4"]   = scores["bleu_4"]['score']
        results["rouge"]    = scores["rouge"]['rougeL']
        results["cider"]    = scores["cider"]['score']

        P,R,F1 = score(predictions,gts,lang="en",verbose=False,idf=True,rescale_with_baseline=True)
        results["bertscore"] = F1.mean().item()
        logger.info(f"Epoch {epoch}: Loss {np.mean(loss_list)}")
        for key in results:
            logger.info(f"Epoch {epoch}: {key}: {results[key]}")
      
def main():
    args = parse_args()
    cfg = load_config(args)
    cfg.args = args

    # Dummy check to avoid overwriting
    cfg_path = os.path.join(cfg.LOGDIR,'config.yaml').replace('./',f'{os.getcwd()}/')
    assert cfg_path == args.cfg_file, f"config file path should be {cfg_path} but got {args.cfg_file}"

    if not cfg.TASK.PRETRAIN:
        assert hasattr(cfg,'BRANCH'), "BRANCH should be defined in config for finetuning."
        cfg.alignment_cfg = load_config(cfg.ALIGNMENT)


    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=os.path.join(cfg.LOGDIR,'stdout.log'))
    model = SimpleT5Model(cfg)
    
    # Maintain a name list in main process
    with open(cfg.DATA.TEST, 'rb') as f:
        data = pickle.load(f)
    name_list = []
    for d in data:
        if d['video_name'] != 'standard':
            name_list.append(d['video_name'])

    # Distributed Training
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.localrank)

    model = model.cuda()

    # Distributed Training
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(  model, device_ids=[args.localrank],
                                                        output_device=args.localrank)
    optimizer = AdamW(model.parameters(), lr=float(cfg.OPTIMIZER.LR))
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'train_logs'))

    # Distributed Training
    if dist.get_rank() == 0:
        store = dist.TCPStore("127.0.0.1", 1238, dist.get_world_size(), True, timedelta(seconds=30))
    else:
        store = dist.TCPStore("127.0.0.1", 1238, dist.get_world_size(), False, timedelta(seconds=30))
    val_dataloader = construct_dataloader('val',cfg.DATA.TEST,cfg.TASK.PRETRAIN,5,alignment_cfg=cfg.alignment_cfg)
    summary_writer = SummaryWriter()
    
    if args.eval_multi:
        from glob import glob
        from natsort import natsorted
        if dist.get_rank()==0:
            logger.info("Evaluating multiple checkpoints")
        checkpoints = natsorted(glob(os.path.join(cfg.LOGDIR,"checkpoints","*.pth")))
    else:
        checkpoints = [args.ckpt]
    for ckpt in checkpoints:
        epoch = load_checkpoint(cfg,model,optimizer,ckpt)
        eval(cfg,val_dataloader, model,epoch,summary_writer,store=store,name_list=name_list,logger=logger)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    