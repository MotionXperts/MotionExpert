import os, json, sys, torch, numpy as np, logging, pickle, dotenv
import torch.distributed as dist
from pytorch_lightning import seed_everything
from utils.parser import parse_args, load_config
from utils.data_information import load_video_name
from utils.cider import readJSON, readPickle, getGTCaptions, BLEUScore, CIDERScore
from dataloaders import construct_dataloader
from models.CoachMe import CoachMe
from transformers import AutoTokenizer
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

def eval(cfg, eval_dataloader, model, epoch, summary_writer, sanity_check = False, store = None, video_name_list = None,
         logger = None, eval_name = "", test_pkl_file = None) :
    Tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast = True)
    model.eval()
    model = model.cuda()
    seed_everything(42)
    loss_list = [] 
    att_node_results, att_graph_results, max_index_results = {}, {}, {}
    prompt = "Motion Description : " if cfg.TASK.PRETRAIN else "Motion Instruction : "
    with torch.no_grad() :
        # Distributed Training.
        if dist.get_rank() == 0 :
            eval_dataloader = tqdm(eval_dataloader, total = len(eval_dataloader), desc = 'Evaluating')
        for index, batch in enumerate(eval_dataloader) :
            (video_name, skeleton_coords, seq_len, frame_mask, label_batch, labels_batch, std_coords, subtraction) = batch

            decoder_input_ids = Tokenizer([prompt],
                                          return_tensors = "pt",
                                          padding = True,
                                          truncation = True,
                                          max_length = 160,
                                          add_special_tokens = False)['input_ids']
            decoder_input_ids = decoder_input_ids.repeat(skeleton_coords.shape[0], 1).to(skeleton_coords.device)

            inputs = {"video_name" : video_name,
                      "skeleton_coords" : skeleton_coords.to(model.device),
                      "frame_mask" : frame_mask.to(model.device),
                      "seq_len" : seq_len,
                      "std_coords" : std_coords.to(model.device),
                      "decoder_input_ids" : decoder_input_ids.to(model.device),
                      "subtraction" : subtraction.to(model.device),
                      "tokenizer" : Tokenizer,
                      # For visualizing attention.
                      "result_dir" : cfg.LOGDIR,
                      "epoch" : epoch}

            generated_ids, att_node, att_graph, max_index = model.module.generate(**inputs)

            if cfg.EVAL.score :
                tgt_batch = Tokenizer(label_batch,
                                      return_tensors = "pt",
                                      padding = "max_length",
                                      truncation = True,
                                      max_length = 160)['input_ids'].to(skeleton_coords.device)
                tgt_input = tgt_batch[ :, : -1 ]
                tgt_label = tgt_batch[:, 1:]
                inputs['decoder_input_ids'] = tgt_input.to(model.device)
                inputs['labels'] = tgt_label.to(model.device)
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
            for name, att_graph in zip(video_name, att_graph) :
                att_graph_results[name] = att_graph.cpu().numpy().tolist()
            for name, max_index in zip(video_name, max_index) :
                max_index_results[name] = max_index.cpu().numpy().tolist()
            if dist.get_rank() == 0 :
                eval_dataloader.set_postfix({'loss' : np.mean(loss_list)})

    # Distributed Training.
    if dist.get_rank() == 0 :
        if cfg.EVAL.score :
            summary_writer.add_scalar('eval/loss', np.mean(loss_list), epoch)

        results = {}
        for name in video_name_list :
            # Distributed Training.
            try : results[name] = store.get(name).decode('utf-8')
            except : continue

        # Clean up generated text by replacing all Unicode right single quotation marks (\u2019) with
        # standard ASCII apostrophes (').
        for data in results :
            instruction = results[data]
            results[data] = instruction.replace('\u2019', "'")

        print("Saving results")
        filename = str(epoch) + '.json'
        if epoch == "demo" :
            result_json = os.path.join(cfg.JSONDIR, cfg.EVAL.UID + '_' + filename)
        else :
            result_json = cfg.JSONDIR + '/results_epoch' + filename
        with open(result_json, 'w') as f :
            json.dump(results, f, indent = 4)
            print(f"Results saved in {result_json}")
        with open(cfg.JSONDIR + '/att_node_results_epoch' + filename, 'w') as f :
            json.dump(att_node_results, f, indent = 4)
        with open(cfg.JSONDIR + '/att_graph_results_epoch' + filename, 'w') as f :
            json.dump(att_graph_results, f, indent = 4)
        with open(cfg.JSONDIR + '/max_index_epoch' + filename, 'w') as f :
            json.dump(max_index_results, f, indent = 4)

        if cfg.EVAL.score :
            predictions_dict = readJSON(cfg.JSONDIR + '/results_epoch' + str(epoch) + '.json')
            annotations = readPickle(test_pkl_file)
            ground_truth_dict = getGTCaptions(cfg, annotations)
            print("Length of predictions_dict : ", len(predictions_dict))
            print("Length of ground_truth_dict : ", len(ground_truth_dict))
            # Calculate scores.
            metrics = [load_metric("bleu", resulting_name = "bleu_1", compute_kwargs = {"max_order" : 1}),
                    load_metric("bleu", resulting_name = "bleu_4", compute_kwargs = {"max_order" : 4}),
                    load_metric("rouge"),
                    load_metric("cider")]
            Evaluator = NLGMetricverse(metrics)
            # Convert predictions and gts to list to fit with bert_score.
            # Make sure predictions and gts are in the same order.
            predictions = dict(sorted(predictions_dict.items()))
            ground_truth = dict(sorted(ground_truth_dict.items()))
            predictions = list(predictions.values())
            ground_truth = list(ground_truth.values())
            scores = Evaluator(predictions = predictions, references = ground_truth, reduce_fn = "mean")

            results = {}
            results["bleu_1"] = scores["bleu_1"]['score']
            results["bleu_4"] = scores["bleu_4"]['score']
            results["rouge"] = scores["rouge"]['rougeL']
            results["cider"] = scores["cider"]['score']

            P, R, F1 = score(predictions, ground_truth, lang = "en", verbose = False, idf = True, rescale_with_baseline = True)
            results["bertscore"] = F1.mean().item()
            logger.info(f"Epoch {epoch} : Loss {np.mean(loss_list)}")
            for key in results :
                logger.info(f"Epoch {epoch} : {key} : {results[key]}")

def main() :
    args = parse_args()
    cfg = load_config(args)

    logging.basicConfig(level = logging.INFO,
                        format = '%(asctime)s %(filename)s %(lineno)d: %(message)s',
                        datefmt = '%Y-%m-%d %H:%M:%S',
                        filename = os.path.join(cfg.LOGDIR, 'stdout.log'))

    model = CoachMe(cfg).to(torch.float32)

    seed_everything(42)

    test_pkl_file = cfg.DATA.TEST
    video_name_list = load_video_name(test_pkl_file)

    # Distributed Training.
    dist.init_process_group(backend='nccl', init_method='env://')
    id = dist.get_rank()
    device = id % torch.cuda.device_count()

    # Distributed Training.
    torch.cuda.set_device(id)
    model = model.cuda()

    # Distributed Training.
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [device], output_device = device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = float(cfg.OPTIMIZER.LR))
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'train_logs'))

    # Distributed Training.
    if dist.get_rank() == 0 :
        store = dist.TCPStore("127.0.0.1", 5052, dist.get_world_size(), True, timedelta(seconds = 30))
    else :
        store = dist.TCPStore("127.0.0.1", 5052, dist.get_world_size(), False, timedelta(seconds = 30))
    val_dataloader = construct_dataloader('test', cfg, test_pkl_file)
    summary_writer = SummaryWriter()
    
    checkpoints = [cfg.EVAL.ckpt]
    for ckpt in checkpoints :
        model = model.to(torch.float32)
        epoch = load_checkpoint(cfg, model, optimizer, ckpt)
        model = model.to(torch.float32)
        model.eval()
        eval(cfg, val_dataloader, model, epoch, summary_writer, store = store, video_name_list = video_name_list, logger = logger,
             eval_name = cfg.SETTING, test_pkl_file = test_pkl_file)
    dist.destroy_process_group()

if __name__ == "__main__" :
    main()