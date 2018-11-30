"""Train a sequence-to-sequence model on the Reddit dataset."""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from os_ds import OpenSubtitlesDataset
from ubuntu import UbuntuCorpus
from models import Seq2Seq, random_sample
from nlputils import convert_npy_to_str
from tqdm import tqdm

SAMPLE_PATH = '/data2/ymeng/opensubtitles_sample/'

parser = argparse.ArgumentParser(description='Run seq2seq model on Opensubtitles conversations')
parser.add_argument('--source', default=SAMPLE_PATH, help='Directory to look for data files')
parser.add_argument('--model_path', default=None, help='File path where model is saved')
parser.add_argument('--vocab', help='Where to save generated vocab file')
parser.add_argument('--regen', default=False, action='store_true', help='Renerate vocabulary')
parser.add_argument('--device', default='cuda:1', help='Cuda device (or cpu) for tensor operations')
parser.add_argument('--epochs', default=1, action='store', type=int, help='Number of epochs to run model for')
parser.add_argument('--restore', default=False, action='store_true', help='Set to restore model from save')
parser.add_argument('--dataset', default='ubuntu', help='Choose either opensubtitles or ubuntu dataset to train')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

max_history = 10
max_len = 20
max_examples = None
max_vocab_examples = None
max_vocab = 50000
num_epochs = args.epochs
d_emb = 200
d_enc = 300
d_dec = 400
lr = .0001

ds = None
if args.dataset == 'opensubtitles':
    print('Using OpenSubtitles dataset')
    ds = OpenSubtitlesDataset(args.source, max_len, max_history, max_vocab, args.vocab, max_examples=max_examples,
                              regen=args.regen)
elif args.dataset == 'ubuntu':
    print('Using Ubuntu dialogue corpus')
    ds = UbuntuCorpus(args.source, args.vocab, max_vocab, max_len, max_history, max_examples=max_examples,
                      max_examples_for_vocab=max_vocab_examples, regen=args.regen)
else:
    print('Must specify either ubuntu or opensubtitles dataset.')
    exit()

print('Num lines: %s' % ds.num_lines)

print('Num examples: %s' % len(ds))
print('Num sources: %s' % len(ds.sources))

model = Seq2Seq(d_emb, d_enc, len(ds.vocab), d_dec, max_len, bos_idx=ds.vocab[ds.bos])

if args.restore and args.model_path is not None:
    print('Restoring model from save')
    model.load_state_dict(torch.load(args.model_path))

model.to(device)
model.train()

print('Training')

# train model architecture
ce = nn.CrossEntropyLoss(ignore_index=0)
optim = optim.Adam(model.parameters(), lr=lr)
losses = []

for epoch_idx in range(num_epochs):
    #ds.load_sources(args.source)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=1)
    bar = tqdm(dl)  # visualize progress bar
    for i, data in enumerate(bar):
        data = [d.to(device) for d in data]
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

print('Evaluating')
model.eval()

# print examples
#ds.load_sources(args.source)
dl = DataLoader(ds, batch_size=5, shuffle=True, num_workers=0)
for i, data in enumerate(dl):
    data = [d.to(device) for d in data]
    history, response = data
    if i > 0:
        break

    logits, preds = model(history, sample_func=random_sample)
    for j in range(preds.shape[0]):
        np_pred = preds[j].cpu().numpy()
        pred = convert_npy_to_str(np_pred, ds.vocab, ds.eos)
        print(pred)



