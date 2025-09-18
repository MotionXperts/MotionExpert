import os, torch, numpy as np, pickle, logging, warnings, traceback, warnings
# Visualize the gt indices
import json
import torch.cuda
import random
os.environ['NUMEXPR_MAX_THREADS'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = "false"
warnings.filterwarnings("ignore", category = UserWarning)
warnings.filterwarnings("ignore", category = FutureWarning)
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from utils.parser import parse_args, load_config
from utils.data_information import load_video_name
from tqdm import tqdm
from pytorch_lightning import seed_everything
from evaluation import eval

from torch.utils.tensorboard import SummaryWriter
# Distributed Training
import torch.distributed as dist
from dataloaders import construct_dataloader
from models.CoachMe import CoachMe
from models import save_checkpoint, load_checkpoint
from datetime import timedelta

logger = logging.getLogger(__name__)

def train(cfg, train_dataloader, model, optimizer, scheduler, scaler, summary_writer, epoch, logger) :
    model.train()
    loss_list = []
    Tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast = True)
    if dist.get_rank() == 0 :
        train_dataloader = tqdm(train_dataloader, total = len(train_dataloader), desc = 'Training')

    # Visualize the gt indices
    if cfg.DEBUG == True :
        video_best_gt_indices = {}
        video_original_gt_indices = {}

    for index, batch in enumerate(train_dataloader) :
        (video_name, skeleton_coords, seq_len, frame_mask, label_batch, labels_batch, std_coords, subtraction) = batch

        # Clears all gradients stored in the model’s parameters.
        model.zero_grad()
        # Clears gradients for all parameters that the optimizer is tracking.
        optimizer.zero_grad()

        # Convert the ground truth (descriptions or instructions) into token IDs.
        if cfg.LOSS == "RandomGT" :
            num_gt = len(labels_batch[0])
            rand_idx = random.randint(0, num_gt - 1)
            instructions = []
            for i in range(0, len(labels_batch)) :
                instructions.append(labels_batch[i][rand_idx])
        else :
            instructions = label_batch
        tgt_batch = Tokenizer(instructions, return_tensors = "pt", padding = "max_length",
                              truncation = True, max_length = 160)['input_ids'].to(skeleton_coords.device)
        tgt_input = tgt_batch[:, :-1]
        tgt_label = tgt_batch[:, 1:]

        inputs = { "video_name" : video_name,
                   "skeleton_coords" : skeleton_coords.to(model.device),
                   "frame_mask" : frame_mask.to(model.device),
                   "seq_len" : seq_len,
                   "std_coords" : std_coords.to(model.device),
                   "decoder_input_ids" : tgt_input.to(model.device),
                   "labels" : tgt_label.to(model.device),
                   "subtraction" : subtraction.to(model.device),
                   "tokenizer" : Tokenizer }

        # Forwards the data through the model.
        outputs = model(**inputs)

        # Visualize the gt indices
        logits = outputs.logits
        batch_size, seq_len, vocab_size = logits.shape
        best_losses, best_gts = [], []
        best_gt_indices = []
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index = -100)
        if cfg.LOSS == "ClosestSimGT":
            # ClosestSimGT :
            # From multiple ground truths, select the one that is most similar to the prediction,
            # and compute the cross-entropy loss. The goal is to converge towards the most similar
            # ground truth.

            for i in range(batch_size) :
                min_loss = float('inf')
                best_gt = None
                best_idx = -1
                num_gt = len(labels_batch[i])
                for j in range(num_gt) :
                    gt_label = labels_batch[i][j]
                    gt_label = Tokenizer([gt_label], return_tensors = "pt", padding = "max_length",
                                         truncation = True, max_length = 160)['input_ids'].to(skeleton_coords.device)
                    gt_label = gt_label[:, 1:].to(logits.device)
                    with torch.no_grad():
                        avg_loss = loss_fn(logits[i].view(-1, vocab_size), gt_label.view(-1))
                    if avg_loss < min_loss :
                        min_loss = avg_loss
                        best_gt = gt_label
                        best_idx = j
                best_losses.append(min_loss)
                best_gts.append(best_gt)
                best_gt_indices.append(best_idx)
                if cfg.DEBUG == True :
                    vid = video_name[i]
                    video_best_gt_indices[vid] = best_idx

            final_labels = torch.stack(best_gts)
            loss = loss_fn(logits.view(-1, vocab_size), final_labels.view(-1))
        else :
            # PerGT :
            # For each ground truth, compute the cross-entropy loss.
            if cfg.DEBUG == True :
                tgt_label = tgt_label.to(logits.device)
                for i in range(batch_size) :
                    min_loss = float('inf')
                    best_gt = None
                    best_idx = -1
                    orignal_idx = -1
                    num_gt = len(labels_batch[i])
                    for j in range(num_gt) :
                        gt_label = labels_batch[i][j]
                        gt_label = Tokenizer([gt_label], return_tensors = "pt", padding = "max_length",
                                            truncation = True, max_length = 160)["input_ids"].to(skeleton_coords.device)
                        gt_label = gt_label[:, 1:].to(logits.device)
                        with torch.no_grad():
                            avg_loss = loss_fn(logits[i].view(-1, vocab_size), gt_label.view(-1))
                        if avg_loss < min_loss :
                            min_loss = avg_loss
                            best_gt = gt_label
                            best_idx = j

                        if torch.equal(gt_label[0], tgt_label[i]):
                            orignal_idx = j
                    best_losses.append(min_loss)
                    best_gts.append(best_gt)
                    best_gt_indices.append(best_idx)
                    vid = video_name[i]
                    if vid not in video_best_gt_indices:
                        video_best_gt_indices[vid] = []
                    if vid not in video_original_gt_indices:
                        video_original_gt_indices[vid] = []

                    video_best_gt_indices[vid].append(best_idx)
                    video_original_gt_indices[vid].append(orignal_idx)
            # Calculate loss for every ground truth
            loss = outputs.loss

        # Computes gradients using backpropagation.
        loss.backward()
        # Updates the model's parameters.
        optimizer.step()
        # Adjusts the learning rate.
        scheduler.step()

        # Distributed Training.
        loss[torch.isnan(loss)] = 0
        dist.all_reduce(loss, async_op = False)
        reduced_loss = loss / dist.get_world_size()
        loss_list.append(reduced_loss.detach().cpu())
        if dist.get_rank() == 0 :
            train_dataloader.set_postfix({'loss' : np.mean(loss_list),
                                          'lr' : scheduler.optimizer.param_groups[0]['lr']})

        # Visualize the gt indices
        if cfg.DEBUG == True :
            save_dir = cfg.JSONDIR + 'best_gt_indices'
            save_ori_dir = cfg.JSONDIR + 'ori_gt_indices'
            os.makedirs(save_dir, exist_ok = True)
            os.makedirs(save_ori_dir, exist_ok = True)
            sorted_indices = dict(sorted(video_best_gt_indices.items(), key=lambda item: item[0]))
            sorted_ori_indices = dict(sorted(video_original_gt_indices.items(), key=lambda item: item[0]))
            # Save the best ground truth indices for each video.
            with open(os.path.join(save_dir, f'best_gt_indices_{epoch}.json'), 'w') as f :
                json.dump(sorted_indices, f, indent=4)
            if cfg.LOSS == "PerGT" :
                with open(os.path.join(save_ori_dir, f'ori_gt_indices_{epoch}.json'), 'w') as f :
                    json.dump(sorted_ori_indices, f, indent=4)

    if dist.get_rank() == 0 :
        summary_writer.add_scalar('train/loss', np.mean(loss_list), epoch)
        logger.info(f"Epoch {epoch} : Loss {np.mean(loss_list)}")

def main():
    args = parse_args()
    cfg = load_config(args)
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(filename)s %(lineno)d: %(message)s',
                        datefmt = '%Y-%m-%d %H:%M:%S', filename = os.path.join(cfg.LOGDIR, 'stdout.log'))

    model = CoachMe(cfg).to(torch.float32)

    test_pkl_file = cfg.DATA.TEST
    video_name_list = load_video_name(test_pkl_file)

    # Distributed Training.
    dist.init_process_group(backend = 'nccl', init_method = 'env://')
    if dist.get_rank() == 0 :
        store = dist.TCPStore("127.0.0.1", 8082, dist.get_world_size(), True, timedelta(seconds = 30))
    else :
        store = dist.TCPStore("127.0.0.1", 8082, dist.get_world_size(), False, timedelta(seconds = 30))

    seed_everything(42)

    # Distributed Training.
    id = dist.get_rank()
    device = id % torch.cuda.device_count()
    model = model.to(device)
    torch.cuda.set_device(id)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [device], output_device = device,
                                                      find_unused_parameters = True)

    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr = float(cfg.OPTIMIZER.LR))
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'train_logs'))

    train_dataloader = construct_dataloader('train', cfg, cfg.DATA.TRAIN)
    test_dataloader = construct_dataloader('test', cfg, cfg.DATA.TEST)

    max_epoch = cfg.OPTIMIZER.MAX_EPOCH
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = cfg.OPTIMIZER.WARMUP_STEPS,
                                                num_training_steps = max_epoch * len(train_dataloader))
    model = model.to(torch.float32)
    start_epoch = load_checkpoint(cfg, model, optimizer)
    model = model.to(torch.float32)
    if cfg.TASK.PRETRAIN == True :
        cycle = 1
    else :
        cycle = 5
    try :
        print(f"start_epoch : {start_epoch}, max_epoch : {max_epoch}")
        for epoch in range(start_epoch, max_epoch):
            # Distributed Training
            if dist.get_rank() == 0:
                logger.info(f"Training epoch {epoch}")

            train_dataloader.sampler.set_epoch(epoch)
            train(cfg, train_dataloader, model, optimizer, scheduler, scaler, summary_writer, epoch, logger)
            if (epoch + 1) % cycle == 0 :

                # Distributed Training
                if dist.get_rank() == 0 :
                    os.makedirs(cfg.CKPTDIR, exist_ok = True)
                    model.eval()
                    save_checkpoint(cfg, model, optimizer, epoch + 1)

                try :
                    eval(cfg, test_dataloader, model, epoch + 1, summary_writer, False, store = store,
                         video_name_list = video_name_list, logger = logger, test_pkl_file = test_pkl_file)
                except Exception as e :
                    print(traceback.format_exc())
                    print(f"Error {e} \n in evaluation at epoch {epoch}, continuing training.")

    except Exception as e :
        print(traceback.format_exc())
        print(f"{e} occured, saving model before quitting.")
    finally :
        if dist.get_rank() == 0 and epoch != start_epoch :
            save_checkpoint(cfg, model, optimizer, epoch + 1)
        dist.destroy_process_group()

if __name__ == '__main__' :
    main()