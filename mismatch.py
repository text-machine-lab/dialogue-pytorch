"""Train and evaluate model which classifies whether an output response is correct
for a given input message (they appear as a message-response pair in the dataset) or if
the response is mismatched (taken from a different message in the dataset). This classification
can be a measure of the appropriateness of the output response with respect to the input."""
import argparse
import torch
from os_ds import OpenSubtitlesDataset
from models import MismatchClassifier

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
lr = .001

ds = OpenSubtitlesDataset(args.source, max_len, max_history, max_vocab, args.vocab, max_examples=max_examples,
                          regen=args.regen)

model = MismatchClassifier(d_emb, d_enc, len(ds.vocab))