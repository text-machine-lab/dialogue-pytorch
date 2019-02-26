"""Train a language model on the Ubuntu dataset."""

import argparse
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from os_ds import OpenSubtitlesDataset
from ubuntu import UbuntuCorpus
from models import Seq2Seq, random_sample, Decoder
from nlputils import convert_npy_to_str
from tensorboard_logger import configure, log_value
from tqdm import tqdm
from tgalert import TelegramAlert

parser = argparse.ArgumentParser(description='Run seq2seq model on Opensubtitles conversations')
parser.add_argument('--source', default=None, help='Directory to look for data files')
parser.add_argument('--model_path', default=None, help='File path where model checkpoint is saved')
parser.add_argument('--temp', help='Where to save generated dataset and vocab files')
parser.add_argument('--regen', default=False, action='store_true', help='Renerate vocabulary')
parser.add_argument('--device', default='cuda:0', help='Cuda device (or cpu) for tensor operations')
parser.add_argument('--epochs', default=1, action='store', type=int, help='Number of epochs to run model for')
parser.add_argument('--restore', default=False, action='store_true', help='Set to restore model from save')
parser.add_argument('--num_print', default='1', help='Number of batches to print')
parser.add_argument('--run', default=None, help='Path to save run values for viewing with Tensorboard')
parser.add_argument('--tgdisable', default=False, action='store_true', help='If true, suppress Telegram alerts')
parser.add_argument('--max_examples', default=None, help='Number of examples to use when training on dataset')
# parser.add_argument('--dataset', default='ubuntu', help='Choose either opensubtitles or ubuntu dataset to train')
parser.add_argument('--val', default=None, help='Validation set to use for model evaluation')
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
    val_temp_dir = os.path.join(args.temp, 'validation')
    valds = UbuntuCorpus(args.val, val_temp_dir, max_vocab, max_len, history_len, concat_feature=True, regen=args.regen,
                      vocab=ds.vocab)
    print('Printing statistics for validation set')
    valds.print_statistics()
else:
    print('Using training set for evaluation')

if args.regen: alert.write('run_lm: Building datasets complete')

print('Num examples: %s' % len(ds))
print('Vocab length: %s' % len(ds.vocab))

model = Decoder(len(ds.vocab), d_emb, d_dec, history_len + max_len, d_context=0, bos_idx=ds.vocab[ds.bos])

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

        _, _, convo = data

        logits = model(None, labels=convo)

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
                logits = model(None, labels=convo)
                loss = ce(logits.view(-1, logits.shape[-1]), convo.view(-1))
                log_value('val_loss', loss.item(), epoch_idx * len(dl) + i)

        if (i % 1000 == 999 or i == len(dl) - 1) and args.model_path is not None:
            torch.save(model.state_dict(), args.model_path)

if args.epochs > 0: alert.write('run_lm: Training complete')

######### EVALUATION ###################################################################################################

def approx_equal(x, y, e=1e-3):
    return torch.lt(torch.abs(x-y), e).all()

def adjust_lm_logits(logits, vocab):
    """Take probability mass from </s> symbol and place it on the <eos> symbol."""
    # compute probabilities from logits
    probs = F.softmax(logits, dim=-1)  # b x t x v
    # transfer mass from <\s> to <eos>
    slashsprobs = probs[:, :, vocab['</s>']]  # b x t
    probs[:, :, vocab['<eos>']] = probs[:, :, vocab['<eos>']] + slashsprobs  # b x t
    probs[:, :, vocab['</s>']] = probs[:, :, vocab['</s>']] - slashsprobs  # b x t
    ones = torch.ones(probs.shape[0], probs.shape[1]).to(logits.device)
    dist_sum = probs.sum(dim=-1)
    assert approx_equal(ones, dist_sum)
    # take the log to get logits back
    backlogits = torch.log(probs)
    assert backlogits.ne(logits).all()
    # return logits
    return backlogits

def replace_eos_slashs(utterances, vocab, reverse=False):
    """
    For each utterance, finds all <eos> tokens and replaces them
    with </s>. Tested.
    :param utterances: (batch_size, max_len) tensor with indices
    :param vocab: Vocab object containing bi-directional mapping tokens <--> indices
    :param reverse: if reverse, replace slashs with eos
    :return:
    """
    tk_eos = vocab['<eos>']
    tk_s = vocab['</s>']

    if reverse:
        tmp = tk_eos
        tk_eos = tk_s
        tk_s = tmp

    eos_tokens = (utterances == tk_eos).long()  # 1 if token is eos
    s_tokens = eos_tokens * tk_s  # map where 0 is normal token and tk_s has location of eos with index of /s
    return utterances * (1-eos_tokens) + s_tokens  # set all eos to zero, then add /s token map


def gather_response(convo, split_indices, max_len):
    """
    Extract all tokens in convo that appear after index in split_indices (batched). Return
    tensor containing these tokens extracted.

    :param convo: (batch_size, convo_len) tensor containing token indices
    :param split_indices: (batch_size,) tensor, for each example giving the index of the first token to extract
    :param max_len: maximum number of 1's in a row of the mask, maximum extracted response
    :return: (batch_size, max_len) tensor containing extracted indices
    """
    batch_size = convo.shape[0]
    convo_len = convo.shape[1]
    result = torch.zeros([batch_size, max_len]).long()
    for i in range(batch_size):
        response_len = min(convo_len - split_indices[i], max_len)
        result[i, :response_len] = convo[i, split_indices[i]:split_indices[i] + response_len]
    return result

with torch.no_grad():
    print('Printing generated examples')
    dl = DataLoader(valds, batch_size=5, num_workers=0)
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
        logits, preds = model.complete(convo_incomplete, sample_func=random_sample)
        history_lens = (history_no_eos != 0).long().sum(dim=1)
        response_preds = gather_response(preds, history_lens, max_len)
        response_preds = replace_eos_slashs(response_preds, ds.vocab, reverse=True)
        # pred represents the whole completed conversation
        # now we need to remove the response from the conversation


        for j in range(preds.shape[0]):
            np_pred = response_preds[j].cpu().numpy()
            np_complete_convo = preds[j].cpu().numpy()
            np_context = history[j].cpu().numpy()
            np_target = response[j].cpu().numpy()
            context = convert_npy_to_str(np_context, valds.vocab, valds.eos)
            complete_convo = convert_npy_to_str(np_complete_convo, valds.vocab, valds.eos)
            pred = convert_npy_to_str(np_pred, valds.vocab, valds.eos)
            target = convert_npy_to_str(np_target, valds.vocab, valds.eos)
            print('History: %s' % context)
            print('Completed convo: %s' % complete_convo)
            print('Prediction: %s' % pred)
            print('Target: %s' % target)
            print()

    print('Evaluating perplexity')
    dl = DataLoader(valds, batch_size=3, num_workers=0)
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

        logits = model(prelabels=history_no_eos, labels=response)
        logits = adjust_lm_logits(logits, ds.vocab)
        loss = ce(logits.view(-1, logits.shape[-1]), response.view(-1))
        entropies.append(loss)

    entropy = torch.mean(torch.stack(entropies, dim=0))
    print('Cross entropy: %s' % entropy.item())
    perplexity = torch.exp(entropy)
    print('Validation perplexity: %s' % perplexity.item())




