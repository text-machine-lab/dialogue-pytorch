"""Train a sequence-to-sequence model on the Reddit dataset."""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from os_ds import OpenSubtitlesDataset
from models import Seq2Seq, random_sample
from nlputils import convert_npy_to_str
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Run seq2seq model on Opensubtitles conversations')
parser.add_argument('--source', default=None, help='Directory to look for data files')
parser.add_argument('--model_path', default=None, help='File path where model is saved')
parser.add_argument('--vocab', help='Where to save generated vocab file')
parser.add_argument('--regen', default=False, action='store_true', help='Renerate vocabulary')
parser.add_argument('--device', default='cuda:0', help='Cuda device (or cpu) for tensor operations')
parser.add_argument('--epochs', default=1, action='store', type=int, help='Number of epochs to run model for')
parser.add_argument('--restore', default=False, action='store_true', help='Set to restore model from save')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

max_history = 10
max_len = 20
max_examples = 100000
max_vocab=10000
num_epochs = args.epochs
d_emb = 200
d_enc = 300
d_dec = 300
lr = .001

# here we first test the reddit dataset object
ds = OpenSubtitlesDataset(args.source, max_len, max_history, max_vocab, args.vocab, max_examples=max_examples,
                          regen=args.regen)

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

print('Evaluation')
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



