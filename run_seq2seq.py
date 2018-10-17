"""Train a sequence-to-sequence model on the Reddit dataset."""
import argparse
from torch.utils.data import DataLoader
from reddit import RedditDataset
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Extract Reddit conversations')
parser.add_argument('--source', default=None, help='Reddit file to extract from')
parser.add_argument('--vocab', help='Where to save generated vocab file')
parser.add_argument('--regen', default=False, action='store_true', help='Renerate vocabulary for Reddit dataset')
args = parser.parse_args()

max_doc=100
max_title = 10
max_history = 50
max_response = 20
max_vocab=100

# here we first test the reddit dataset object
ds = RedditDataset(args.source, args.vocab, max_doc=max_doc, max_title=max_title, max_history=max_history,
                   max_response=max_response, regen=args.regen, max_vocab=max_vocab)

loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

for data in tqdm(loader):
    pass

print(ds.vocab.tk_to_idx)
