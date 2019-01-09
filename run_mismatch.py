"""Train and evaluate model which classifies whether an output response is correct
for a given input message (they appear as a message-response pair in the dataset) or if
the response is mismatched (taken from a different message in the dataset). This classification
can be a measure of the appropriateness of the output response with respect to the input."""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from ubuntu import UbuntuCorpus
from models import MismatchClassifier
import numpy as np
from glove import GloveLoader
import os

from tg_alert import TelegramAlert
alert = TelegramAlert()

try:

    parser = argparse.ArgumentParser(description='Run seq2seq model on Opensubtitles conversations')
    parser.add_argument('--source', default=None, help='Directory to look for data files')
    parser.add_argument('--model_path', default=None, help='File path where model is saved')
    parser.add_argument('--temp', help='Where to save generated vocab file')
    parser.add_argument('--regen', default=False, action='store_true', help='Renerate vocabulary')
    parser.add_argument('--device', default='cuda:0', help='Cuda device (or cpu) for tensor operations')
    parser.add_argument('--epochs', default=1, action='store', type=int, help='Number of epochs to run model for')
    parser.add_argument('--restore', default=False, action='store_true', help='Set to restore model from save')
    parser.add_argument('--glove', default=None, help='Path to glove file')
    parser.add_argument('--val', default=None, help='Validation set used to evaluate model accuracy')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    max_history = 10
    max_len = 20
    max_examples = None
    max_vocab_examples = None
    max_vocab=50000
    num_epochs = args.epochs
    d_emb = 200
    d_enc = 200
    lr = .001

    ds = UbuntuCorpus(args.source, args.temp, max_vocab, max_len, max_history, max_examples=max_examples,
                      regen=args.regen, mismatch=True)

    print('Num examples: %s' % len(ds))
    print('Vocab length: %s' % len(ds.vocab))

    model = MismatchClassifier(d_emb, d_enc, len(ds.vocab))

    if args.glove is not None:
        # initialize input embeddings from glove embeddings
        loader = GloveLoader(args.glove)
        np_embs = loader.build_embeddings(ds.vocab)
        embs = nn.Parameter(torch.from_numpy(np_embs).float())
        model.m_enc.embs.weight = model.r_enc.embs.weight = embs

    if args.restore and args.model_path is not None:
        print('Restoring model from save')
        model.load_state_dict(torch.load(args.model_path))

    ######### TRAINING #####################################################################################################

    def calc_accuracy(logits, label):
        preds = (logits >= 0).long()
        accuracy = (preds == label).float().mean()
        return accuracy

    model.to(device)
    model.train()

    print('Training')

    # train model architecture
    bce = nn.BCEWithLogitsLoss()
    optim = optim.Adam(model.parameters(), lr=lr)
    losses = []
    accuracies = []

    for epoch_idx in range(num_epochs):
        dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=1)
        bar = tqdm(dl)  # visualize progress bar
        for i, data in enumerate(bar):
            data = [d.to(device) for d in data]
            history, response, label = data

            logits = model(history, response)
            loss = bce(logits, label.float())
            optim.zero_grad()
            loss.backward()
            losses.append(loss.item())
            accuracies.append(calc_accuracy(logits, label).item())
            optim.step()

            if i % 100 == 99:
                bar.set_description('Loss: %s, accuracy: %s' % (np.mean(losses), np.mean(accuracies)))
                losses = []
                accuracies = []
            if (i % 1000 == 999 or i == len(dl) - 1) and args.model_path is not None:
                torch.save(model.state_dict(), args.model_path)

    if args.epochs > 0: alert.write('run_mismatch: training finished')

    ####### EVALUATION #####################################################################################################
    print('Evaluating')
    eval_batches = 10000 // 32
    model.eval()

    if args.val is not None:
        print('Evaluating perplexity')
        val_temp_dir = os.path.join(args.temp, 'validation')
        ds = UbuntuCorpus(args.val, val_temp_dir, max_vocab, max_len, max_history,
                          regen=True, mismatch=True)

    labels = []

    with torch.no_grad():
        dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=1)
        bar = tqdm(dl, total=eval_batches)
        accuracies = []
        for i, data in enumerate(bar):

            if i >= eval_batches:
                bar.close()
                break

            data = [d.to(device) for d in data]
            history, response, label = data

            labels.append(label.float().mean().item())

            logits = model(history, response)
            accuracies.append(calc_accuracy(logits, label).item())

        print('Accuracy: %s' % np.mean(accuracies))
        print('Average label: %s' % np.mean(labels))

except Exception:
    alert.write('run_mismatch: exception!')
    raise