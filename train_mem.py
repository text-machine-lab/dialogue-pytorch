"""Train a sequence-to-sequence model on the Reddit dataset."""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from os_ds import OpenSubtitlesDataset
from ubuntu import UbuntuCorpus
from models import random_sample
from src.model import Seq2SeqMem, Seq2SeqAttn
from nlputils import convert_npy_to_str
from tqdm import tqdm

SAMPLE_PATH = '/data2/ymeng/ubuntu_csvfiles/sample.csv'

def main():
    parser = argparse.ArgumentParser(description='Run seq2seq model on Opensubtitles conversations')
    parser.add_argument('--source', default=SAMPLE_PATH, help='Directory to look for data files')
    parser.add_argument('--batch_size', default=32, help='batch_size for training')
    parser.add_argument('--max_examples', default=None, type=int, help='max num of example to use for training')
    parser.add_argument('--model_path', default=None, help='File path where model is saved')
    parser.add_argument('--model_type', default=None, help='model name')
    parser.add_argument('--vocab', default='ubuntu_vocab.pkl', help='Where to save generated vocab file')
    parser.add_argument('--regen', default=False, action='store_true', help='Renerate vocabulary')
    # parser.add_argument('--device', default='cuda:2', help='Cuda device (or cpu) for tensor operations')
    parser.add_argument('--epochs', default=1, action='store', type=int,
                        help='Number of epochs to run model for. 0 for evaluation only')
    parser.add_argument('--restore', default=False, action='store_true', help='Set to restore model from save')
    # parser.add_argument('--dataset', default='ubuntu', help='Choose either opensubtitles or ubuntu dataset to train')
    parser.add_argument('--val', default=None, help='Validation set to use for model evaluation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    max_history = 10
    max_len = 25
    max_examples = args.max_examples
    max_vocab_examples = None
    max_vocab = 50000
    num_epochs = args.epochs
    d_emb = 200
    # d_enc = 300
    d_sent_enc = 512
    d_turn_enc = 1024
    d_dec = 400
    lr = .001

    ########### DATASET CREATION ###########################################################################################

    print('Using Ubuntu dialogue corpus')

    split_history = False if args.model_type == 'seq2seqattn' else True
    ds = UbuntuCorpus(args.source, args.vocab, max_vocab, max_len, max_history, max_examples=max_examples,
                      max_examples_for_vocab=max_vocab_examples, regen=args.regen,
                      split_history=split_history)

    print('Num examples: %s' % len(ds))
    print('Vocab length: %s' % len(ds.vocab))

    if args.model_type == 'seq2seqattn':
        model = Seq2SeqAttn(d_sent_enc, len(ds.vocab), max_len, ds.vocab[ds.bos], ds.vocab[ds.eos])
    elif args.model_type == 'seq2seqmem':
        model = Seq2SeqMem(d_emb, d_sent_enc, len(ds.vocab), max_len, bos_idx=ds.vocab[ds.bos])
    else:
        model = Seq2SeqMem(d_emb, d_sent_enc, len(ds.vocab), d_turn_enc, d_dec, max_len, bos_idx=ds.vocab[ds.bos])

    if args.restore and args.model_path is not None:
        print('Restoring model from save')
        model.load_state_dict(torch.load(args.model_path))

    ######### TRAINING #####################################################################################################

    model.to(device)
    model.train()

    print('Training')

    if args.model_type == 'seq2seqattn':
        model.set_optimizer(lr)
        for epoch_idx in range(num_epochs):
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=1)
            bar = tqdm(dl)
            for i, data in enumerate(bar):
                data = [d.type(torch.LongTensor).to(device) for d in data]
                history, response = data
                loss = model.fit(history, response)

                if i % 10 == 9:
                    bar.set_description('Loss: {}'.format(loss.item()))
                if (i % 1000 == 999 or i == len(dl) - 1) and args.model_path is not None:
                    torch.save(model.state_dict(), args.model_path)

    else:
        # train model architecture
        ce = nn.CrossEntropyLoss(ignore_index=0)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []
        for epoch_idx in range(num_epochs):
            dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=1)
            bar = tqdm(dl)  # visualize progress bar
            for i, data in enumerate(bar):
                data = [d.type(torch.LongTensor).to(device) for d in data]
                history, response = data

                logits = model(history, labels=response)
                loss = ce(logits.view(-1, logits.shape[-1]), response.view(-1))
                optim.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optim.step()

                if i % 10 == 9:
                    bar.set_description('Loss: %s' % np.mean(losses))
                if (i % 1000 == 999 or i == len(dl) - 1) and args.model_path is not None:
                    torch.save(model.state_dict(), args.model_path)

    print('Printing examples')
    model.eval()
    with torch.no_grad():
        # print examples
        dl = DataLoader(ds, batch_size=5, num_workers=1)
        for i, data in enumerate(dl):
            data = [d.type(torch.LongTensor).to(device) for d in data]
            history, response = data
            if i > 0:
                break

            # if args.model_type == 'seq2seqattn':
            #     logits, preds = model(history, sample_func=random_sample)
            # else:
            logits, preds = model(history, sample_func=random_sample)
            # print(logits.size(), preds.size())
            for j in range(preds.shape[0]):
                np_pred = preds[j].cpu().numpy()
                # print(np_pred.shape)
                np_context = history[j].flatten().cpu().numpy()
                np_context = np_context[np.nonzero(np_context)]
                # print(np_context)
                if args.model_type in ('seq2seqattn', 'seq2seqmem'):
                    context = convert_npy_to_str(np_context, ds.vocab, None)
                else:
                    context = convert_npy_to_str(np_context, ds.vocab, ds.eos)
                pred = convert_npy_to_str(np_pred, ds.vocab, ds.eos)
                print('History: %s' % context)
                print('Prediction: %s\n' % pred)


    if args.val is not None:
        ds = UbuntuCorpus(args.val, args.vocab, max_vocab, max_len, max_history, max_examples=None,
                          max_examples_for_vocab=max_vocab_examples, regen=False, split_history=split_history)
        eval(model, ds, device)


######### EVALUATION ###################################################################################################
def eval(model, dataset, device):

    print('Evaluating perplexity')

    with torch.no_grad():
        dl = DataLoader(dataset, batch_size=128, num_workers=1, drop_last=False)
        print('test data length', len(dl))
        ce = nn.CrossEntropyLoss(ignore_index=0)

        entropies = []
        for i, data in enumerate(tqdm(dl)):
            # print(i, '---')
            data = [d.type(torch.LongTensor).to(device) for d in data]
            history, response = data
            # print(response.size())

            if hasattr(model, 'model_type') and model.model_type == 'seq2seqattn':
                loss, logits, pred = model(history, response, sample_func=random_sample)
            else:
                logits = model(history, response, sample_func=random_sample)
                loss = ce(logits.view(-1, logits.shape[-1]), response.view(-1))
            entropies.append(loss)

        print(len(entropies))
        entropy = torch.mean(torch.stack(entropies, dim=0))
        print('Cross entropy: %s' % entropy.item())
        perplexity = torch.exp(entropy)
        print('Validation perplexity: %s' % perplexity.item())


if __name__ == '__main__':
    main()



