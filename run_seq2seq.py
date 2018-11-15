"""Train a sequence-to-sequence model on the Reddit dataset."""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from reddit import RedditDataset
from models import Seq2Seq
from nlputils import convert_npy_to_str
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Extract Reddit conversations')
parser.add_argument('--source', default=None, help='Reddit file to extract from')
parser.add_argument('--vocab', help='Where to save generated vocab file')
parser.add_argument('--regen', default=False, action='store_true', help='Renerate vocabulary for Reddit dataset')
parser.add_argument('--device', default='cuda:0', help='Cuda device (or cpu) for tensor operations')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
max_doc=300
max_title = 10
max_history = 50
max_response = 30
max_vocab=10000
d_emb = 200
d_enc = 300
d_dec = 300
lr = .001

# here we first test the reddit dataset object
ds = RedditDataset(args.source, args.vocab, max_doc=max_doc, max_title=max_title, max_history=max_history,
                   max_response=max_response, regen=args.regen, max_vocab=max_vocab)

model = Seq2Seq(d_emb, d_enc, len(ds.vocab), d_dec, max_response, bos_idx=ds.vocab[ds.bos])

model.to(device)

# train model architecture
ce = nn.CrossEntropyLoss(ignore_index=0)
optim = optim.Adam(model.parameters(), lr=lr)
dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)
bar = tqdm(dl)  # visualize progress bar
losses = []
for i, data in enumerate(bar):
    data = [d.to(device) for d in data]
    history, response, conv_title, doc_title, doc = data

    logits = model(history, labels=response)
    loss = ce(logits.view(-1, logits.shape[-1]), response.view(-1))
    optim.zero_grad()
    loss.backward()
    losses.append(loss.item())
    optim.step()

    if i % 100 == 99:
        bar.set_description('Loss: %s' % np.mean(losses))

# print examples
dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)
bar = tqdm(dl)  # visualize progress bar
for i, data in enumerate(bar):
    data = [d.to(device) for d in data]
    history, response, conv_title, doc_title, doc = data
    if i > 2:
        bar.close()
        break

    logits, preds = model(history)
    for j in range(preds.shape[0]):
        np_pred = preds[j].cpu().numpy()
        pred = convert_npy_to_str(np_pred, ds.vocab, ds.eos)
        print(pred)


#bar = tqdm(loader)

# for i in range(100):
#     np_history, np_response, np_conv_title, np_doc_title, np_doc = ds[i]
#

    # history = convert_npy_to_str(np_history, ds.vocab, ds.eos)
    # response = convert_npy_to_str(np_response, ds.vocab, ds.eos)
    # conv_title = convert_npy_to_str(np_conv_title, ds.vocab, ds.eos)
    # doc_title = convert_npy_to_str(np_doc_title, ds.vocab, ds.eos)
    # doc = convert_npy_to_str(np_doc, ds.vocab, ds.eos)
    # print('Title: %s' % conv_title)
    # print('Input: %s' % history)
    # print('Output: %s' % response)
    # print('Doc Title: %s' % doc_title)
    # print('Doc: %s' % doc)
    # print()



