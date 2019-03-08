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
from utils import load_train_args, print_numpy_examples

from tgalert import TelegramAlert

parser = argparse.ArgumentParser(description='Run seq2seq model on Opensubtitles conversations')
load_train_args(parser)
args = parser.parse_args()

alert = TelegramAlert(disable=args.tgdisable)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

if args.run is not None and args.epochs > 0:
    configure(os.path.join(args.run, str(datetime.now())), flush_secs=5)

max_history = 170
max_len = 30
max_examples = None
max_vocab_examples = None
max_vocab = 50000
num_epochs = args.epochs
num_print = int(args.num_print)
d_emb = 200
d_enc = 400
d_dec = 400
lr = .0001

########### DATASET AND MODEL CREATION #################################################################################

print('Using Ubuntu dialogue corpus')
ds = UbuntuCorpus(args.source, args.temp, max_vocab, max_len, max_history, max_examples=max_examples, regen=args.regen)

print('Printing statistics for training set')
ds.print_statistics()

# use validation set if provided
valds = None
if args.val is not None:
    print('Using validation set for evaluation')
    if args.tempval is None:
        args.tempval = os.path.join(args.temp, 'validation')
    valds = UbuntuCorpus(args.val, args.tempval, max_vocab, max_len, max_history, regen=args.regenval,
                      vocab=ds.vocab, max_examples=max_examples)
    print('Printing statistics for validation set')
    valds.print_statistics()
else:
    print('Using training set for evaluation')

if args.regen: alert.write('run_seq2seq: Building datasets complete')

print('Num examples: %s' % len(ds))
print('Vocab length: %s' % len(ds.vocab))

model = Seq2Seq(d_emb, len(ds.vocab), d_dec, max_len, bos_idx=ds.vocab[ds.bos])

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

        history, response = data

        logits = model(history, labels=response)

        loss = ce(logits.view(-1, logits.shape[-1]), response.view(-1))
        optim.zero_grad()
        loss.backward()

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

if args.samples_file is not None:
    args.samples_file = open(str(args.samples_file), 'w')

with torch.no_grad():
    batch_size_print = 10
    print('Printing generated examples')
    dl = DataLoader(valds, batch_size=batch_size_print, num_workers=1)
    model.eval()
    # print examples
    for i, data in enumerate(dl):
        data = [d.to(device) for d in data]
        history, response = data
        if i > num_print:
            break
        preds = torch.zeros([batch_size_print, max_len]).long().to(device)
        logits, preds = model.complete(preds, sample_func=random_sample)
        print_numpy_examples(valds.vocab, valds.eos, history, response, preds, samples_file=args.samples_file)

    if args.samples_file is not None: args.samples_file.close()

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




