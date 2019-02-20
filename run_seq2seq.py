"""Train a sequence-to-sequence model on the Reddit dataset."""

import argparse
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from os_ds import OpenSubtitlesDataset
from ubuntu import UbuntuCorpus
from models import Seq2Seq, random_sample, MismatchClassifier, MismatchSeq2Seq
from nlputils import convert_npy_to_str
from tensorboard_logger import configure, log_value
from tqdm import tqdm

from tgalert import TelegramAlert
alert = TelegramAlert()

parser = argparse.ArgumentParser(description='Run seq2seq model on Opensubtitles conversations')
parser.add_argument('--source', default=None, help='Directory to look for data files')
parser.add_argument('--model_path', default=None, help='File path where model checkpoint is saved')
parser.add_argument('--temp', help='Where to save generated dataset and vocab files')
parser.add_argument('--regen', default=False, action='store_true', help='Renerate vocabulary')
parser.add_argument('--device', default='cuda:0', help='Cuda device (or cpu) for tensor operations')
parser.add_argument('--epochs', default=1, action='store', type=int, help='Number of epochs to run model for')
parser.add_argument('--restore', default=False, action='store_true', help='Set to restore model from save')
parser.add_argument('--mismatch', default=False, action='store_true', help='If true, use mismatch classifier for training encoder')
parser.add_argument('--num_print', default='1', help='Number of batches to print')
parser.add_argument('--run', default=None, help='Path to save run values for viewing with Tensorboard')
# parser.add_argument('--dataset', default='ubuntu', help='Choose either opensubtitles or ubuntu dataset to train')
parser.add_argument('--val', default=None, help='Validation set to use for model evaluation')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

if args.run is not None and args.epochs > 0:
    configure(os.path.join(args.run, str(datetime.now())), flush_secs=5)

max_history = 10
max_len = 20
max_examples = None
max_vocab_examples = None
max_vocab = 50000
num_epochs = args.epochs
num_print = int(args.num_print)
d_mismatch_emb = 200
d_mismatch_enc = 200
d_emb = 200
d_enc = 300
d_dec = 400
lr = .0001

########### DATASET AND MODEL CREATION #################################################################################

print('Using Ubuntu dialogue corpus')
ds = UbuntuCorpus(args.source, args.temp, max_vocab, max_len, max_history,
                  mismatch=args.mismatch, max_examples=max_examples, regen=args.regen)

print('Printing statistics for training set')
ds.print_statistics()

# use validation set if provided
valds = None
if args.val is not None:
    print('Using validation set for evaluation')
    val_temp_dir = os.path.join(args.temp, 'validation')
    valds = UbuntuCorpus(args.val, val_temp_dir, max_vocab, max_len, max_history, regen=False,
                      vocab=ds.vocab)
    print('Printing statistics for validation set')
    valds.print_statistics()
else:
    print('Using training set for evaluation')

if args.regen: alert.write('run_seq2seq: Building datasets complete')

print('Num examples: %s' % len(ds))
print('Vocab length: %s' % len(ds.vocab))

if not args.mismatch:
    model = Seq2Seq(d_emb, d_enc, len(ds.vocab), d_dec, max_len, bos_idx=ds.vocab[ds.bos],
                    context_size=d_mismatch_enc)
else:
    model = MismatchSeq2Seq(d_emb, d_enc, len(ds.vocab), d_dec, max_len, bos_idx=ds.vocab[ds.bos])

if args.restore and args.model_path is not None:
    print('Restoring model from save')
    model.load_state_dict(torch.load(args.model_path))

######### TRAINING #####################################################################################################

model.to(device)
model.train()

print('Training')

# train model architecture
ce = nn.CrossEntropyLoss(ignore_index=0)
bce = nn.BCEWithLogitsLoss()
optim = optim.Adam(model.parameters(), lr=lr)

# if provided, determine validation loss throughout training
valiter = None
if args.val is not None:
    valiter = iter(DataLoader(valds, batch_size=32, num_workers=0))
for epoch_idx in range(num_epochs):
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=1)
    bar = tqdm(dl)  # visualize progress bar
    for i, data in enumerate(bar):
        data = [d.to(device) for d in data]

        if args.mismatch:
            history, response, chosen_response, match = data
        else:
            history, response = data

        logits = model(history, labels=response)

        loss = ce(logits.view(-1, logits.shape[-1]), response.view(-1))
        optim.zero_grad()
        loss.backward()

        # if mismatch component exists, train mismatch simultaneously
        if args.mismatch:
            pred = model.mismatch(history, chosen_response)
            mismatch_loss = bce(pred.view(-1), match.view(-1).float())
            mismatch_loss.backward()

        optim.step()

        if args.run is not None: log_value('train_loss', loss.item(), epoch_idx * len(dl) + i)

        if args.run is not None and valiter is not None and i % 10 == 9:
            with torch.no_grad():
                # this code grabs a validation batch, or resets the val data loader once it reaches the end
                try:
                    history, response = next(valiter)
                except StopIteration:
                    valiter = iter(DataLoader(valds, batch_size=32, num_workers=0))
                    history, response = next(valiter)
                history = history.to(device)
                response = response.to(device)
                logits = model(history, labels=response)
                loss = ce(logits.view(-1, logits.shape[-1]), response.view(-1))
                log_value('val_loss', loss.item(), epoch_idx * len(dl) + i)

        if (i % 1000 == 999 or i == len(dl) - 1) and args.model_path is not None:
            torch.save(model.state_dict(), args.model_path)

if args.epochs > 0: alert.write('run_seq2seq: Training complete')

######### EVALUATION ###################################################################################################

with torch.no_grad():
    print('Printing generated examples')
    dl = DataLoader(valds, batch_size=5, num_workers=0)
    model.eval()
    # print examples
    for i, data in enumerate(dl):
        data = [d.to(device) for d in data]
        history, response = data
        if i > num_print:
            break
        logits, preds = model(history, sample_func=random_sample)
        for j in range(preds.shape[0]):
            np_pred = preds[j].cpu().numpy()
            np_context = history[j].cpu().numpy()
            np_target = response[j].cpu().numpy()
            context = convert_npy_to_str(np_context, valds.vocab, valds.eos)
            pred = convert_npy_to_str(np_pred, valds.vocab, valds.eos)
            target = convert_npy_to_str(np_target, valds.vocab, valds.eos)
            print('History: %s' % context)
            print('Prediction: %s' % pred)
            print('Target: %s' % target)
            print()

    print('Evaluating perplexity')
    dl = DataLoader(valds, batch_size=32, num_workers=0)
    total_batches = min(len(dl), 1500)
    entropies = []
    bar = tqdm(dl, total=total_batches)
    for i, data in enumerate(bar):

        if i >= total_batches:
            bar.close()
            break

        data = [d.to(device) for d in data]
        history, response = data
        logits = model(history, labels=response)
        loss = ce(logits.view(-1, logits.shape[-1]), response.view(-1))
        entropies.append(loss)

    entropy = torch.mean(torch.stack(entropies, dim=0))
    print('Cross entropy: %s' % entropy.item())
    perplexity = torch.exp(entropy)
    print('Validation perplexity: %s' % perplexity.item())




