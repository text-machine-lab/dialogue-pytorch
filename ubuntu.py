"""Class which loads and preprocesses the Ubuntu dialogue corpus."""
from torch.utils.data import Dataset
import os
import pickle
import csv
import nltk
import argparse
import re
from tqdm import tqdm
from nlputils import Vocab, convert_str_to_npy, convert_npy_to_str, raw_count

class UbuntuCorpus(Dataset):
    def __init__(self, source_dir, tmp_dir, vocab_len, max_len, history_len,
                 max_examples=None, max_examples_for_vocab=None, regen=False):
        super().__init__()

        # set no upper bound on examples used in case of None
        if max_examples is None:
            max_examples = float('inf')
        if max_examples_for_vocab is None:
            max_examples_for_vocab = float('inf')

        self.max_len = max_len
        self.history_len = history_len
        self.max_examples = max_examples
        self.max_examples_for_vocab = max_examples_for_vocab
        self.source_dir = source_dir
        self.vocab_len = vocab_len
        self.eos = '<eos>'
        self.unk = '<unk>'
        self.bos = '<bos>'
        self.pad = '<pad>'
        self.url = '<url>'

        if tmp_dir is None or not os.path.exists(tmp_dir) or regen:

            # determine line count
            print('Counting lines in dataset')
            self.num_examples = raw_count(self.source_dir)

            print('Generating vocabulary')
            with open(source_dir, 'r') as f:
                csv_f = csv.reader(f)
                # regenerate vocabulary
                self.vocab = Vocab()
                bar = tqdm(csv_f, total=min(self.num_examples, max_examples_for_vocab))
                for idx, entry in enumerate(bar):
                    history, response, label = entry
                    # format line
                    if label == '1':
                        # we only use responses which are correct (label 1)
                        line = format_line(history + ' ' + response)
                        self.vocab.add_doc(line)
                        if idx + 1 >= self.max_examples_for_vocab:
                            # if there is a maximum number of examples, we only build vocab from those
                            bar.close()
                            break

                # prune vocab to set size and add special tokens
                self.vocab.prune(self.vocab_len - 4)
                self.vocab.insert_token(self.pad, 0)
                self.vocab.add_doc(self.unk)
                self.vocab.add_doc(self.eos)
                self.vocab.add_doc(self.bos)
                # save vocabulary and number of lines
                if tmp_dir is not None:
                    print('Saving vocabulary/dataset size')
                    pickle.dump([self.num_examples, self.vocab], open(tmp_dir, 'wb'))
        else:
            # load vocabulary and number of lines
            self.num_examples, self.vocab = pickle.load(open(tmp_dir, 'rb'))

        # open up csv reader to begin reading in examples
        self.source = csv.reader(open(source_dir, 'r'))

    def __len__(self):
        return min(self.num_examples, self.max_examples)

    def __getitem__(self, item):

        label = '0'
        history = response = None
        while label == '0':
            # find the next valid response
            history, response, label = next(self.source)

        history = format_line(history)
        response = format_line(response)

        np_history = convert_str_to_npy(history, self.vocab, self.max_len * self.history_len, eos=self.eos, pad=0, unk=self.unk)
        np_response = convert_str_to_npy(response, self.vocab, self.max_len, eos=self.eos, pad=0, unk=self.unk)

        return np_history, np_response

def format_line(line):
    line = re.sub(r"http\S+", '<url>', line)
    line = ' '.join(nltk.word_tokenize(line.strip().lower()))
    return line.replace('< /s >', '</s>').replace('< url >', '<url>')


########### TESTING ####################################################################################################


if __name__ == '__main__':
    # here we create a small ubuntu dataset and test it
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=None)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--regenerate', default=False, action='store_true')
    args = parser.parse_args()

    vocab_len = 10000
    max_len = 20
    history_len = 5
    max_examples = None
    max_vocab_examples = 10000

    # problem: history cuts off front end, not tale end

    ds = UbuntuCorpus(args.source, args.save_path, vocab_len, max_len, history_len, max_examples=max_examples,
                      regen=args.regenerate, max_examples_for_vocab=max_vocab_examples)

    print('Dataset length: %s' % len(ds))
    print('Vocab length: %s' % len(ds.vocab))

    # now we print actual examples from the dataset
    for i in range(10):
        np_history, np_response = ds[0]  # the index isn't used
        history = convert_npy_to_str(np_history, ds.vocab, eos=ds.eos)
        response = convert_npy_to_str(np_response, ds.vocab, eos=ds.eos)

        print('History: %s' % history)
        print('Response: %s' % response)
        print()