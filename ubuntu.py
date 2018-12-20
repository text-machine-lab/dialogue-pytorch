"""Class which loads and preprocesses the Ubuntu dialogue corpus."""
from torch.utils.data import Dataset
import os
import pickle
import csv
import nltk
import random
import numpy as np
import argparse
import re
from tqdm import tqdm
from nlputils import Vocab, convert_str_to_npy, convert_npy_to_str, raw_count

# lock to make csv accesses safe
import threading
lock = threading.Lock()


class UbuntuCorpus(Dataset):
    def __init__(self, source_dir, tmp_file, vocab_len, max_len, history_len,
                 max_examples=None, max_examples_for_vocab=None, regen=False,
                 mismatch=False, split_history=False):
        """
        This class loads the Ubuntu dialogue corpus .csv file and builds a vocabulary from it.
        This class is also able to be used by a dataloader to dispense individual (history, response)
        pairs from the dataset. In addition, a mismatch option can be set allowing the model to dispense
        (history, response, match) pairs, where match indicates if the response matches the history or if
        it was taken from a different example in the dataset (mismatch). Match and mismatch each have
        a 50% probability. The dataset is loaded line by line without shuffling, such that __getitem__
        does not rely on index.

        :param source_dir: .csv file to find ubuntu dialogues
        :param tmp_file: where to save vocabulary and line count results for dataset for faster loading
        :param vocab_len: size of vocabulary after pruning
        :param max_len: responses longer than this many tokens are clipped
        :param history_len: number of messages in history
        :param max_examples: only dispense this many examples
        :param max_examples_for_vocab: only use this many examples to build vocabulary
        :param regen: if true, regenerate vocab and line count no matter what
        :param mismatch: if true, dispense (history, response, match) tuples
        :param split_history: if true, history matrix is shape (history_len, max_len) where each utterance is separated
        """
        super().__init__()

        # set no upper bound on examples used in case of None
        if max_examples is None:
            max_examples = float('inf')
        if max_examples_for_vocab is None:
            max_examples_for_vocab = float('inf')

        self.max_len = max_len
        self.history_len = history_len
        self.split_history = split_history
        self.max_examples = max_examples
        self.max_examples_for_vocab = max_examples_for_vocab
        self.source_dir = source_dir
        self.vocab_len = vocab_len
        self.eos = '<eos>'
        self.unk = '<unk>'
        self.bos = '<bos>'
        self.pad = '<pad>'
        self.url = '<url>'
        self.prev_response = None
        self.mismatch = False

        if tmp_file is None or not os.path.exists(tmp_file) or regen:

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
                if tmp_file is not None:
                    print('Saving vocabulary/dataset size')
                    pickle.dump([self.num_examples, self.vocab], open(tmp_file, 'wb'))
        else:
            # load vocabulary and number of lines
            self.num_examples, self.vocab = pickle.load(open(tmp_file, 'rb'))

        self.reset()

        # dispense one example for testing
        history, response = self[0]
        assert isinstance(history, np.ndarray)
        assert isinstance(response, np.ndarray)
        if split_history:
            assert history.shape == (self.history_len, self.max_len)
        else:
            assert history.shape == (self.history_len * self.max_len,)
        assert response.shape == (self.max_len,)

        # if mismatch is true, we set it now after loading the first response
        self.mismatch = mismatch

    def reset(self):
        # open up csv reader to begin reading in examples
        self.source = csv.reader(open(self.source_dir, 'r'))

    def __len__(self):
        # only half of the examples are real (1), the rest are fake (0)
        return min(self.num_examples // 2, self.max_examples)

    def __getitem__(self, item):
        label = '0'
        history = response = None

        # we thread-lock the csv writer
        with lock:
            while label == '0':
                # find the next valid response
                line = next(self.source)
                if len(line) == 3:
                    history, response, label = line
                else:
                    print('End of file. Reloading file')
                    self.reset()

        history = format_line(history)
        response = format_line(response)

        # if mismatch option not set, feature is disabled and this always evaluates to true, normal behavior
        # otherwise, randomly chooses to use current response or previous response with is_match flag set true/false
        is_match = not self.mismatch or bool(random.getrandbits(1))

        if is_match:
            np_response = convert_str_to_npy(response, self.vocab, self.max_len, eos=self.eos, pad=0, unk=self.unk)
        else:
            np_response = convert_str_to_npy(self.prev_response, self.vocab, self.max_len, eos=self.eos, pad=0, unk=self.unk)


        if not self.split_history:
            np_history = convert_str_to_npy(history, self.vocab, self.max_len * self.history_len, eos=self.eos, pad=0,
                                            unk=self.unk)
        else:
            # we want each utterance to be on a separate line
            hist_utters = history.split('</s>')
            hist_utters = hist_utters[-self.history_len:]
            hist_utter_npys = []
            for utterance in hist_utters:
                np_utter = convert_str_to_npy(utterance, self.vocab, self.max_len, eos=self.eos, pad=0, unk=self.unk)
                hist_utter_npys.append(np_utter)
            np_history = np.stack(hist_utter_npys, axis=0)
            # we need add pad utterances to get array up to final shape (history_len, max_len)
            np_pad = np.zeros([self.history_len - np_history.shape[0], self.max_len])
            np_history = np.concatenate([np_pad, np_history], axis=0)

        self.prev_response = response

        np_match = np.array(is_match, dtype=int)

        if not self.mismatch:
            return np_history, np_response
        else:
            return np_history, np_response, np_match

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
    history_len = 7
    max_examples = None
    max_vocab_examples = 10000
    mismatch = False
    split_history = True

    # problem: history cuts off front end, not tale end

    ds = UbuntuCorpus(args.source, args.save_path, vocab_len, max_len, history_len, max_examples=max_examples,
                      regen=args.regenerate, max_examples_for_vocab=max_vocab_examples, mismatch=mismatch,
                      split_history=split_history)

    print('Dataset length: %s' % len(ds))
    print('Vocab length: %s' % len(ds.vocab))

    # now we print actual examples from the dataset
    for i in range(10):
        result = ds[0]  # the index isn't used
        np_history = result[0]
        np_response = result[1]

        if not split_history:
            history = convert_npy_to_str(np_history, ds.vocab, eos=ds.eos)
            print('History: %s' % history)
        else:
            for i in range(np_history.shape[0]):
                utterance = convert_npy_to_str(np_history[i], ds.vocab, eos=ds.eos)
                print('History #%d: %s' % (i, utterance))

        response = convert_npy_to_str(np_response, ds.vocab, eos=ds.eos)

        print('Response: %s' % response)

        if mismatch:
            print('Numpy match: %s' % bool(result[2]))
        print()