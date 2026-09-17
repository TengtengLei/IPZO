########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
import datetime
import json
from src.model import GPT, GPTConfig
from src.trainer import Trainer, TrainerConfig
from src.utils import Dataset
import torch
import numpy as np
from src.spikingjelly.clock_driven import functional
from src.binidx import MMapIndexedDataset
from accelerate import accelerator
import argparse
import os

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_train', type=str, default='enwik8')
    parser.add_argument('--data_valid', type=str, default='valid.txt') 
    parser.add_argument('--data_test', type=str, default='test.txt')
    parser.add_argument('--ctx_len', type=int, default=1024, help='token length')
    parser.add_argument('--n_layer', type=int, default=24, help='number of layer')
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr_init', type=float, default=6e-4)
    parser.add_argument('--lr_final', type=float, default=1e-5)
    parser.add_argument('--n_epoch', type=int, default=1000, help='epoch')
    parser.add_argument('--epoch_length', type=int, default=10000, help='length of each epoch')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model')
    parser.add_argument('--mode', type=str, default='pretrain', choices=['pretrain', 'finetune'], help='train mode')
    parser.add_argument('--save_path', type=str, default='trained_models', help='model save path')
    parser.add_argument('--epoch_save_freq', type=int, default=10, help='model save frequency')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--optimizer_type', type=str, default='adam', choices=['adam', 'mezo'], help='train optimizer')
    parser.add_argument('--mezo_random_mode', type=str, default='PGUXoR', choices=['PGUXoR', 'PGUReuse', 'Randint', 'Randn'], help='PGU Type')
    parser.add_argument('--mezo_learning_rate', type=float, default=1e-4, help='learning rate for mezo')
    parser.add_argument('--mezo_epsilon', type=float, default=1e-3, help='mezo epsilon')
    parser.add_argument('--mezo_perturb_nums', type=int, default=1, help='perturb nums')
    parser.add_argument('--mezo_seed', type=str, default='0x123456789', help='PGU seed')
    parser.add_argument('--mezo_lr_schedule', type=str, default='constant',
                        choices=['constant', 'linear', 'cosine'], help='MeZO LR schedule')
    parser.add_argument('--mezo_lr_final', type=float, default=0.0,
                        help='final LR for linear/cosine schedule (start = --mezo_learning_rate)')
    parser.add_argument('--freeze_ratio', type=float, default=0.0,
                        help='fraction of blocks to freeze from the bottom (e.g. 0.5 freezes first half)')
    parser.add_argument('--data_format', type=str, default='char', choices=['char', 'bpe'],
                        help="char = build a char vocab from raw text (SpikeGPT default, small vocab, "
                             "emb/head get truncated); bpe = tokenize with the 20B NeoX tokenizer "
                             "(vocab 50277, matches pretrained emb/head -> full weight transfer)")
    return parser.parse_args()

args = parse_args()


### Step 1: set training data ##########################################################################
optimizer_type = args.optimizer_type
datafile_train = args.data_train # txt file or binidx file
datafile_valid = args.data_valid
datafile_test = args.data_test
datafile_encoding = 'utf-8'
# datafile_encoding = 'utf-16le'

### Step 2: set model size #############################################################################

ctx_len = args.ctx_len        # ===> increase T_MAX in model.py if your ctx_len > 1024
n_layer = args.n_layer
n_embd = args.n_embd

# 'RWKV' (better for char-level English) or 'RWKV-ffnPre' (better in some cases)
model_type = 'RWKV'

### Step 3: set batch size #############################################################################

# ===> batch_size must be divisible by B_GROUP_FORWARD and B_GROUP_BACKWARD in model.py
# For example, if your batch_size = 20, you can set B_GROUP_FORWARD = 4, B_GROUP_BACKWARD = 2
# If you see "CUDA out of memory", reduce it. Use GPU-Z to find the highest value for your VRAM.
batch_size = args.batch_size

### Step 4: set learning rate, training mini-epochs #######################################################

lr_init = args.lr_init
lr_final = args.lr_final
# the mini-epoch is very short and of fixed length (ctx_len * epoch_length_fixed tokens)
n_epoch = args.n_epoch
# 0 = never, 1 = every mini-epoch, 2 = every two mini-epochs, etc.
epoch_save_frequency = args.epoch_save_freq
epoch_save_path = args.save_path
epoch_length_fixed = args.epoch_length

mezo_random_mode=args.mezo_random_mode
mezo_learning_rate=args.mezo_learning_rate
mezo_epsilon=args.mezo_epsilon
mezo_perturb_nums=args.mezo_perturb_nums
mezo_seed=args.mezo_seed

########################################################################################################

import src.utils
src.utils.set_seed(args.seed) # remember to change seed if you load a model

np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)

warmup_tokens = 0
grad_norm_clip = 1.0
betas = (0.9, 0.99)
eps = 4e-9
num_workers = 0

########################################################################################################
# Load data
########################################################################################################

print('loading data... ' + datafile_train + f'  (data_format={args.data_format})')
if args.data_format == 'bpe':
    # BPE path (paper-faithful): tokenize raw text with the 20B NeoX tokenizer.
    # vocab = 50277 == pretrained -> emb/head load fully, no truncation.
    # Dataset's numpy branch reads the vocab size from the VOCAB_SIZE env var.
    import numpy as _np
    from transformers import PreTrainedTokenizerFast
    os.environ['VOCAB_SIZE'] = '50277'
    _tok = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')
    def _load_ids(fn):
        if fn.endswith('.npy'):
            # pre-tokenized id stream (e.g. combined WT-103 + WT-2), load directly
            return _np.load(fn)
        return _np.array(_tok(open(fn, "r", encoding=datafile_encoding).read())['input_ids'], dtype=_np.int64)
    train_dataset = Dataset(_load_ids(datafile_train), ctx_len, epoch_length_fixed)
    valid_dataset = Dataset(_load_ids(datafile_valid), ctx_len, epoch_length_fixed)
    test_dataset  = Dataset(_load_ids(datafile_test),  ctx_len, epoch_length_fixed)
else:
    # char path (SpikeGPT default): builds a char vocab from the raw text
    train_dataset = Dataset(open(datafile_train, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed)
    valid_dataset = Dataset(open(datafile_valid, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed)
    test_dataset  = Dataset(open(datafile_test,  "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed)

########################################################################################################
# Train model
########################################################################################################
if __name__ == '__main__':

    model = GPT(GPTConfig(train_dataset.vocab_size, train_dataset.ctx_len, model_type=model_type,
                          n_layer=n_layer, n_embd=n_embd)).cuda()

# load a trained model. remember to change random seed
# m2 = torch.load('medium/trained-30L-768E-936.pth',map_location=torch.device('cpu'))
# model.load_state_dict(m2)

    if args.mode == 'finetune':
        if not args.pretrained_model:
            raise ValueError("A pretrained model should be provided)")

        if not os.path.exists(args.pretrained_model):
            raise FileNotFoundError(f"Pretrained model does not exist")

        print(f"Finetune: load pretrained model {args.pretrained_model}")
        pretrained_dict = torch.load(args.pretrained_model, map_location='cuda')
        model_dict = model.state_dict()

        filtered_pretrained = {}
        for key, value in pretrained_dict.items():
            if key in model_dict:
                if model_dict[key].shape == value.shape:
                    filtered_pretrained[key] = value
                else:
                    if 'emb.weight' in key or 'head.weight' in key:
                        min_dim = min(model_dict[key].size(0), value.size(0))
                        print(f"Change {key} size: {value.shape} -> {model_dict[key].shape}")
                        filtered_pretrained[key] = value[:min_dim]  # 只复制共通的部分
                    else:
                        print(f"Skip unmatched layer {key}: {value.shape} vs {model_dict[key].shape}")
            else:
                print(f"Skip non-existing layer: {key}")

        # 加载过滤后的权重
        model_dict.update(filtered_pretrained)
        model.load_state_dict(model_dict)

        # model.load_state_dict(pretrained_dict)
    else:
        print("Pretrain: training from scratch")

    # Freeze bottom fraction of blocks (and emb) so MeZO only updates the rest
    if args.freeze_ratio > 0.0:
        num_freeze = int(n_layer * args.freeze_ratio)
        for param in model.emb.parameters():
            param.requires_grad = False
        for i in range(num_freeze):
            for param in model.blocks[i].parameters():
                param.requires_grad = False
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Freeze ratio {args.freeze_ratio}: frozen blocks 0~{num_freeze-1}, "
              f"trainable params {trainable}/{total} ({100*trainable/total:.1f}%)")

    # valid_dataset = None
    # test_dataset = None

    print('model', model_type, 'epoch', n_epoch, 'batchsz', batch_size, 'betas',
             betas, 'eps', eps, 'ctx', ctx_len, 'layer', n_layer, 'embd', n_embd, )
    tconf = TrainerConfig(model_type=model_type,
                          optimizer_type=optimizer_type,
                          max_epochs=n_epoch, 
                          batch_size=batch_size,
                          learning_rate=lr_init, 
                          lr_decay=True, 
                          lr_final=lr_final, 
                          betas=betas, 
                          eps=eps, 
                          grad_norm_clip=grad_norm_clip,
                          warmup_tokens=warmup_tokens, 
                          final_tokens=n_epoch*len(train_dataset)*ctx_len, 
                          num_workers=num_workers, 
                          epoch_save_frequency=epoch_save_frequency, 
                          epoch_save_path=epoch_save_path,
                          mezo_random_mode=mezo_random_mode,
                          mezo_learning_rate=mezo_learning_rate,
                          mezo_epsilon=mezo_epsilon,
                          mezo_perturb_nums=mezo_perturb_nums,
                          mezo_seed=mezo_seed,
                          mezo_lr_schedule=args.mezo_lr_schedule,
                          mezo_lr_final=args.mezo_lr_final)
    trainer = Trainer(model, train_dataset, valid_dataset, test_dataset, tconf)

    trainer.train()

    # final timestamped save disabled: trainer already keeps a single model_best.pth (avoids piling up 554MB files)
    # timestamp = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    # model_save_name = f'{args.mode}-{n_layer}L-{n_embd}E-{timestamp}.pth'
    # model_save_path = os.path.join(epoch_save_path, model_save_name)
    # torch.save(model.state_dict(), model_save_path)
    # print(f"Model saved as: {model_save_path}")

