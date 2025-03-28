import os, json, sys, torch, numpy as np, logging, pickle, dotenv
import torch.distributed as dist
from pytorch_lightning import seed_everything
from utils.parser import parse_args, load_config
from utils.data_information import convert
from cider import readJSON, readPickle, getGTCaptions, BLEUScore, CIDERScore
from dataloaders import construct_dataloader
from models.T5 import SimpleT5Model
from transformers import AutoTokenizer, AdamW
from torch.utils.tensorboard import SummaryWriter
from models import load_checkpoint
os.environ['TOKENIZERS_PARALLELISM'] = "false"
from tqdm import tqdm
from datetime import timedelta
from bert_score import score
from utils.retrieve_most_similar_label import compute_similar_score
from nlgmetricverse import NLGMetricverse, load_metric
from glob import glob
from natsort import natsorted

logging.getLogger().setLevel(logging.WARNING)
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

def eval(cfg, eval_dataloader, model, epoch, summary_writer, sanity_check = False, store = None,
         name_list = None, logger = None, eval_name = "", pkl_file = None) :
    Tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast = True)
    model.eval()
    model = model.cuda()
    seed_everything(42)
    loss_list = [] 
    att_node_results, att_A_results, max_index_results = {}, {}, {}
    prompt = "Motion Description : " if cfg.TASK.PRETRAIN else "Motion Instruction : "
    with torch.no_grad() :
        # Distributed Training.
        if dist.get_rank() == 0 :
            eval_dataloader = tqdm(eval_dataloader, total = len(eval_dataloader), desc = 'Evaluating')
        for index, batch in enumerate(eval_dataloader) :
            (video_name, src_batch, keypoints_mask_batch, standard, seq_len, label_batch, subtraction, labels_batch) = batch

            decoder_input_ids = Tokenizer([prompt],
                                          return_tensors = "pt",
                                          padding = True,
                                          truncation = True,
                                          max_length = 160,
                                          add_special_tokens = False)['input_ids']
            decoder_input_ids = decoder_input_ids.repeat(src_batch.shape[0], 1).to(src_batch.device)
            tgt_batch = Tokenizer(label_batch,
                                  return_tensors = "pt",
                                  padding = "max_length",
                                  truncation = True,
                                  max_length = 160)['input_ids'].to(src_batch.device)
            tgt_input = tgt_batch[ :, : -1 ]
            tgt_label = tgt_batch[ :, 1 : ]
            inputs = {"video_name" : video_name,
                      "input_embedding" : src_batch.to(model.device),
                      "input_embedding_mask" : keypoints_mask_batch.to(model.device),
                      "standard" : standard.to(model.device),
                      "seq_len" : seq_len.to(model.device),
                      "decoder_input_ids" : decoder_input_ids.to(model.device),
                      "subtraction" : subtraction.to(model.device),
                      "tokenizer" : Tokenizer,
                      "labels" : tgt_label.to(model.device),
                      # For visualizing attention.
                      "result_dir" : cfg.LOGDIR,
                      "epoch" : epoch}

            generated_ids, att_node, att_A, max_index = model.module.generate(**inputs)
            # print("Genrated text : ", Tokenizer.decode(generated_ids[0], skip_special_tokens = True, clean_up_tokenization_spaces = True))
            inputs['decoder_input_ids'] = tgt_input.to(model.device)
            loss = model(**inputs).loss

            loss[torch.isnan(loss)] = 0
            # Distributed Training.
            dist.all_reduce(loss, async_op = False)
            reduced_loss = loss / dist.get_world_size()
            loss_list.append(reduced_loss.detach().cpu())
            for name, gen_id, label in zip(video_name, generated_ids, label_batch) :
                if isinstance(gen_id, torch.Tensor) :
                    gen_id = gen_id.tolist()
                decoded_text = Tokenizer.decode(gen_id, skip_special_tokens = True, clean_up_tokenization_spaces = True).split(prompt)
                if len(decoded_text) > 1 :
                    decoded_text = decoded_text[1].strip()
                else :
                    decoded_text = ""

                # Distributed Training.
                store.set(name, decoded_text)
            for name, att_node in zip(video_name, att_node) :
                att_node_results[name] = att_node.cpu().numpy().tolist()
            for name, att_A in zip(video_name, att_A) :
                att_A_results[name] = att_A.cpu().numpy().tolist()
            for name, max_index in zip(video_name, max_index) :
                max_index_results[name] = max_index.cpu().numpy().tolist()
            if dist.get_rank() == 0 :
                eval_dataloader.set_postfix({'loss' : np.mean(loss_list)})
            if sanity_check and index > 4 :
                return

    # Distributed Training.
    if dist.get_rank() == 0 :
        summary_writer.add_scalar('eval/loss', np.mean(loss_list), epoch)

        results = {}
        for name in name_list :
            # Distributed Training.
            try : results[name] = store.get(name).decode('utf-8')
            except : continue

        print("Saving results")
        filename = str(epoch) + '.json'
        result_json = cfg.JSONDIR + '/results_epoch' + filename
        with open(result_json, 'w') as f :
            json.dump(results, f, indent = 1)
            print(f"Results saved in {result_json}")
        with open(cfg.JSONDIR + '/att_node_results_epoch' + filename, 'w') as f :
            json.dump(att_node_results, f)
        with open(cfg.JSONDIR + '/att_A_results_epoch' + filename, 'w') as f :
            json.dump(att_A_results, f)
        with open(cfg.JSONDIR + '/max_index_epoch' + filename, 'w') as f :
            json.dump(max_index_results, f, indent = 4)

        predictions = readJSON(cfg.JSONDIR + '/results_epoch' + str(epoch) + '.json')
        annotations = readPickle(cfg.DATA.TEST) if pkl_file is None else readPickle(pkl_file)

        # GPT chooses the most similar label from the ground truth.
        if cfg.args.gpt_sim :
            key = os.getenv("OPENAI_KEY")
            annotations, abandoned = compute_similar_score(cfg, predictions, key, eval_name, epoch)
            for ab in abandoned :
                print(f"Abandoned : {ab}")
                del predictions[ab]
        if cfg.args.no_calc_score :
            print("\033[91m {} \033[00m".format("Result saved in ", result_json, ". Skipping score calculation."))
        print("Length of annotations : ", len(annotations))
        gts = getGTCaptions(cfg, annotations)
        print("Length of gts : ", len(gts))
        new_gts = {}
        print("Length of results : ", len(results))
        for name in results :
            new_gts[name] = gts[name]

        gts = new_gts

        # Calculate scores.
        metrics = [load_metric("bleu", resulting_name = "bleu_1", compute_kwargs = {"max_order" : 1}),
                   load_metric("bleu", resulting_name = "bleu_4", compute_kwargs = {"max_order" : 4}),
                   load_metric("rouge"),
                   load_metric("cider")]
        Evaluator = NLGMetricverse(metrics)
        # Convert predictions and gts to list to fit with bert_score.
        # Make sure predictions and gts are in the same order.
        predictions = dict(sorted(predictions.items()))

        gts = dict(sorted(gts.items()))

        predictions = list(predictions.values())
        gts = list(gts.values())
        if cfg.TASK.SPORT == 'Skating' :
            scores = Evaluator(predictions = predictions, references = gts, reduce_fn = "mean")
        elif cfg.TASK.SPORT == 'Boxing' :
            scores = Evaluator(predictions = predictions, references = gts, reduce_fn = "max")
        results = {}
        results["bleu_1"] = scores["bleu_1"]['score']
        results["bleu_4"] = scores["bleu_4"]['score']
        results["rouge"] = scores["rouge"]['rougeL']
        results["cider"] = scores["cider"]['score']

        P, R, F1 = score(predictions, gts, lang = "en", verbose = False, idf = True, rescale_with_baseline = True)
        if cfg.TASK.SPORT == 'Skating' :
            results["bertscore"] = F1.mean().item()
        elif cfg.TASK.SPORT == 'Boxing' :
            results["bertscore"] = F1.mean().item()
        logger.info(f"Epoch {epoch} : Loss {np.mean(loss_list)}")
        for key in results :
            logger.info(f"Epoch {epoch} : {key} : {results[key]}")

def main() :
    args = parse_args()
    cfg = load_config(args)
    cfg_path = os.path.join(cfg.LOGDIR, 'config.yaml').replace('./', f'{os.getcwd()}/')

    logging.basicConfig(level = logging.INFO,
                        format = '%(asctime)s %(filename)s %(lineno)d: %(message)s',
                        datefmt = '%Y-%m-%d %H:%M:%S',
                        filename = os.path.join(cfg.LOGDIR, 'stdout.log'))

    model = SimpleT5Model(cfg)

    seed_everything(42)

    pickle_file = cfg.DATA.TEST
    if cfg.args.eval_name == 'test' :
        pickle_file = cfg.DATA.TEST
        eval_name = 'test'
    elif cfg.args.eval_name == 'train' :
        pickle_file = cfg.DATA.TRAIN
        eval_name = 'train'
    elif cfg.args.eval_name == 'untrimmed' :
        pickle_file = cfg.DATA.UNTRIMMED
        eval_name = 'untrimmed'
    elif cfg.args.eval_name == 'segment' :
        pickle_file = cfg.DATA.SEGMENT
        eval_name = 'segment'
    else:
        raise ValueError("Invalid eval_name. eval_name should be set.")
    json_path = cfg.LOGDIR + "/" + eval_name + '.json'
    print("Eval_name : ", eval_name)
    convert(pickle_file, json_path)

    # Maintain a name list in main process
    with open(pickle_file, 'rb') as f :
        data = pickle.load(f)

    name_list = []
    for d in data :
        name_list.append(d['video_name'])

    dist.init_process_group(backend='nccl', init_method='env://')
    id = dist.get_rank()
    device = id % torch.cuda.device_count()

    # Distributed Training.
    torch.cuda.set_device(id)
    model = model.cuda()

    # Distributed Training.
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [device], output_device = device)
    optimizer = AdamW(model.parameters(), lr = float(cfg.OPTIMIZER.LR))
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'train_logs'))

    # Distributed Training.
    if dist.get_rank() == 0 :
        store = dist.TCPStore("127.0.0.1", 5052, dist.get_world_size(), True, timedelta(seconds = 30))
    else :
        store = dist.TCPStore("127.0.0.1", 5052, dist.get_world_size(), False, timedelta(seconds = 30))
    val_dataloader = construct_dataloader('test', cfg, pickle_file)
    summary_writer = SummaryWriter()
    
    checkpoints = [args.ckpt]
    for ckpt in checkpoints :
        epoch = load_checkpoint(cfg, model, optimizer, ckpt)
        eval(cfg, val_dataloader, model, epoch, summary_writer, store = store, name_list = name_list, logger = logger,
             eval_name = eval_name, pkl_file = pickle_file)
    dist.destroy_process_group()

if __name__ == "__main__" :
    main()