"""Train a language model augmented with neural turing machine, on the Ubuntu dataset."""

import argparse
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ubuntu import UbuntuCorpus
from models import random_sample
from ntm_models import NTMAugmentedDecoder
from tensorboard_logger import configure, log_value
from tqdm import tqdm
from tgalert import TelegramAlert
from utils import load_train_args, print_numpy_examples, gather_logits, move_prob_from_s_to_eos, replace_eos_slashs, \
    gather_response

parser = argparse.ArgumentParser(description='Run ntm model on Opensubtitles conversations')
load_train_args(parser)
args = parser.parse_args()

alert = TelegramAlert(disable=args.tgdisable)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

if args.run is not None and args.epochs > 0:
    configure(os.path.join(args.run, str(datetime.now())), flush_secs=5)

history_len = 170
max_len = 30
max_examples = None if args.max_examples is None else int(args.max_examples)
max_vocab_examples = None
max_vocab = 50000
num_epochs = args.epochs
num_print = int(args.num_print)
d_emb = 200
d_dec = 400
lr = .0001

########### DATASET AND MODEL CREATION #################################################################################

print('Using Ubuntu dialogue corpus')
ds = UbuntuCorpus(args.source, args.temp, max_vocab, max_len, history_len,
                  concat_feature=True, max_examples=max_examples, regen=args.regen)

print('Printing statistics for training set')
ds.print_statistics()

# use validation set if provided
valds = None
if args.val is not None:
    print('Using validation set for evaluation')
    if args.tempval is None:
        args.tempval = os.path.join(args.temp, 'validation')
    valds = UbuntuCorpus(args.val, args.tempval, max_vocab, max_len, history_len, concat_feature=True, regen=args.regenval,
                      vocab=ds.vocab, max_examples=max_examples)
    print('Printing statistics for validation set')
    valds.print_statistics()
else:
    print('Using training set for evaluation')
    valds = ds

if args.regen: alert.write('run_lm: Building datasets complete')

print('Num examples: %s' % len(ds))
print('Vocab length: %s' % len(ds.vocab))

model = NTMAugmentedDecoder(len(ds.vocab), d_emb, d_dec, history_len + max_len, bos_idx=ds.vocab[ds.bos])
#model = Decoder(len(ds.vocab), d_emb, d_dec, history_len + max_len, d_context=0, bos_idx=ds.vocab[ds.bos])

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
    valiter = iter(DataLoader(valds, batch_size=32, num_workers=1))
for epoch_idx in range(num_epochs):
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=1)
    bar = tqdm(dl)  # visualize progress bar
    for i, data in enumerate(bar):
        data = [d.to(device) for d in data]

        _, _, convo = data

        logits = model(labels=convo)

        loss = ce(logits.view(-1, logits.shape[-1]), convo.view(-1))
        optim.zero_grad()
        loss.backward()

        optim.step()

        if args.run is not None: log_value('train_loss', loss.item(), epoch_idx * len(dl) + i)

        if args.run is not None and valiter is not None and i % 10 == 9:
            with torch.no_grad():
                # this code grabs a validation batch, or resets the val data loader once it reaches the end
                try:
                    history, response, convo = next(valiter)
                except StopIteration:
                    valiter = iter(DataLoader(valds, batch_size=32, num_workers=0))
                    history, response, convo = next(valiter)
                convo = convo.to(device)
                logits = model(labels=convo)
                loss = ce(logits.view(-1, logits.shape[-1]), convo.view(-1))
                log_value('val_loss', loss.item(), epoch_idx * len(dl) + i)

        if (i % 1000 == 999 or i == len(dl) - 1) and args.model_path is not None:
            torch.save(model.state_dict(), args.model_path)

if args.epochs > 0: alert.write('run_lm: Training complete')

######### EVALUATION ###################################################################################################

if args.samples_file is not None:
    args.samples_file = open(str(args.samples_file), 'w')

with torch.no_grad():
    print('Printing generated examples')
    dl = DataLoader(valds, batch_size=10, num_workers=1)
    model.eval()
    # print examples
    for i, data in enumerate(dl):
        data = [d.to(device) for d in data]
        history, response, convo = data
        if i > num_print:
            break

        # here we remove eos from the history and replace it with </s> (end of utterance)
        history_no_eos = replace_eos_slashs(history, ds.vocab)
        # we will ask the model to fill in the rest of the conversation (i.e. the response)
        response_space = torch.zeros([history_no_eos.shape[0], max_len]).long().to(device)
        # we create space to fill in the response
        convo_incomplete = torch.cat([history_no_eos, response_space], dim=1)
        logits, convo_preds = model.complete(convo_incomplete, sample_func=random_sample)
        history_lens = (history_no_eos != 0).long().sum(dim=1)
        response_preds = gather_response(convo_preds, history_lens, max_len)
        response_preds = replace_eos_slashs(response_preds, ds.vocab, reverse=True)
        # pred represents the whole completed conversation
        # now we need to remove the response from the conversation
        print_numpy_examples(valds.vocab, valds.eos, history, response, response_preds, convo_preds, samples_file=args.samples_file)

    if args.samples_file is not None: args.samples_file.close()

    print('Evaluating perplexity')
    dl = DataLoader(valds, batch_size=32, num_workers=1)
    total_batches = min(len(dl), 1500)
    entropies = []
    bar = tqdm(dl, total=total_batches)
    for i, data in enumerate(bar):

        if i >= total_batches:
            bar.close()
            break

        data = [d.to(device) for d in data]
        history, response, convo = data
        history_no_eos = replace_eos_slashs(history, ds.vocab)
        history_lens = (history_no_eos != 0).long().sum(dim=1)

        convo_logits = model(labels=convo)
        logits = gather_logits(convo_logits, history_lens, max_len)
        logits = move_prob_from_s_to_eos(logits, ds.vocab)
        loss = ce(logits.view(-1, logits.shape[-1]), response.view(-1))
        entropies.append(loss)

    entropy = torch.mean(torch.stack(entropies, dim=0))
    print('Cross entropy: %s' % entropy.item())
    perplexity = torch.exp(entropy)
    print('Perplexity: %s' % perplexity.item())




