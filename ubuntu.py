"""Class which loads and preprocesses the Ubuntu dialogue corpus."""
from torch.utils.data import Dataset
import os
import pickle
import csv
import nltk
import h5py
import random
import numpy as np
import argparse
import re
from tqdm import tqdm
from nlputils import Vocab, convert_str_to_npy, convert_npy_to_str, raw_count


class UbuntuCorpus(Dataset):
    def __init__(self, source_dir, tmp_dir, vocab_len, max_len, history_len,
                 max_examples=None, regen=False,
                 mismatch=False, split_history=False, vocab=None):
        """
        This class loads the Ubuntu dialogue corpus .csv file and builds a vocabulary from it.
        This class is also able to be used by a dataloader to dispense individual (history, response)
        pairs from the dataset. In addition, a mismatch option can be set allowing the model to dispense
        (history, response, match) pairs, where match indicates if the response matches the history or if
        it was taken from a different example in the dataset (mismatch). Match and mismatch each have
        a 50% probability. The dataset is loaded line by line without shuffling, such that __getitem__
        does not rely on index.

        :param source_dir: .csv file to find ubuntu dialogues
        :param tmp_dir: where to save vocabulary and hdf5 dataset file
        :param vocab_len: size of vocabulary after pruning
        :param max_len: responses longer than this many tokens are clipped
        :param history_len: number of messages in history
        :param max_examples: only dispense this many examples
        :param max_examples_for_vocab: only use this many examples to build vocabulary
        :param regen: if true, regenerate vocab and line count no matter what
        :param mismatch: if true, dispense (history, response, match) tuples
        :param split_history: if true, history matrix is shape (history_len, max_len) where each utterance is separated
        :param vocab: if provided, use external vocabulary instead of generating one (for validation sets, etc)
        """
        super().__init__()

        # set no upper bound on examples used in case of None
        if max_examples is None:
            max_examples = float('inf')

        self.max_len = max_len
        self.history_len = history_len
        self.split_history = split_history
        self.max_examples = max_examples
        self.source_dir = source_dir
        self.vocab_len = vocab_len
        self.eos = '<eos>'
        self.unk = '<unk>'
        self.bos = '<bos>'
        self.pad = '<pad>'
        self.url = '<url>'
        self.vocab = vocab
        self.mismatch = False

        if not os.path.exists(tmp_dir) or regen:

            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)

            # determine line count
            print('Counting lines in dataset')
            # num_examples = min(raw_count(self.source_dir), max_examples)
            with open(source_dir, 'r') as f:
                num_examples = 0
                for line in f:
                    if line[-2] == '1':
                        num_examples += 1

            print('Line count: %s' % num_examples)
            num_examples = min(num_examples, max_examples)

            if self.vocab is None:
                print('Generating vocabulary')
                with open(source_dir, 'r') as f:
                    csv_f = csv.reader(f)
                    # regenerate vocabulary
                    self.vocab = Vocab()
                    bar = tqdm(csv_f, total=num_examples)
                    for idx, entry in enumerate(bar):
                        history, response, label = entry

                        # format line
                        if label == '1':
                            # we only use responses which are correct (label 1)
                            line = format_line(history + ' ' + response)
                            self.vocab.add_doc(line)
                            if idx + 1 >= num_examples:
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
            print('Saving vocabulary/dataset size')
            pickle.dump(self.vocab, open(os.path.join(tmp_dir, 'vocab.pkl'), 'wb'))

            print('Building dataset')
            ds_file = h5py.File(os.path.join(tmp_dir, 'ds.hdf5'), 'w')
            self.responses = ds_file.create_dataset('responses', [num_examples, self.max_len], int)
            # choose to split history by utterance
            if self.split_history:
                self.histories = ds_file.create_dataset('histories', [num_examples, history_len, max_len], int)
            else:
                self.histories = ds_file.create_dataset('histories', [num_examples, history_len*max_len], int)

            idx = 0  # index in dataset
            with open(source_dir, 'r') as f:
                csv_f = csv.reader(f)
                bar = tqdm(total=num_examples)
                while True:
                    if idx == num_examples:
                        bar.close()
                        break
                    history, response, label = next(csv_f)
                    if label == '1':
                        np_history, np_response = self.format_line_into_npy(history, response)
                        self.responses[idx] = np_response
                        self.histories[idx] = np_history
                        idx += 1
                        bar.update(1)
        else:
            # load vocabulary and number of lines
            if self.vocab is None:
                self.vocab = pickle.load(open(os.path.join(tmp_dir, 'vocab.pkl'), 'rb'))
            ds_file = h5py.File(os.path.join(tmp_dir, 'ds.hdf5'), 'r')
            self.responses = ds_file['responses']
            self.histories = ds_file['histories']

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

    def print_statistics(self):
        """
        Collect statistics on the first 10000 examples and print them.
        :return:
        """
        with open(self.source_dir, 'r') as f:
            csv_f = csv.reader(f)
            hist_lens = []
            resp_lens = []
            labels = []
            count = 0
            try:
                while True:
                    if count == 10000:
                        break
                    history, response, label = next(csv_f)
                    labels.append(float(label))
                    if label == '1':
                        hist_lens.append(len(history.split()))
                        resp_lens.append(len(response.split()))
                        count += 1
            except:
                # if iterator ends early
                pass


        print('History median length: %s' % np.median(hist_lens))
        print('History std length: %s' % np.std(hist_lens))
        print('Response median length: %s' % np.median(resp_lens))
        print('Response std length: %s' % np.std(resp_lens))
        print('Fraction real responses: %s' % np.mean(labels))


    def format_line_into_npy(self, history, response):
        """
        Perform preprocessing of history and response and convert them into numpy arrays.
        :param history: string containing message history
        :param response: string containing response to message history
        :return: np_history size (history_len*max_len,) or (history_len, max_len) and np_response size
        (max_len,)
        """
        history = format_line(history)
        response = format_line(response)

        np_response = convert_str_to_npy(response, self.vocab, self.max_len, eos=self.eos, pad=0,
                                         unk=self.unk)

        if not self.split_history:
            np_history = convert_str_to_npy(history, self.vocab, self.max_len * self.history_len,
                                            eos=self.eos, pad=0,
                                            unk=self.unk)
        else:
            # we want each utterance to be on a separate line
            hist_utters = history.split('</s>')
            hist_utters = hist_utters[-self.history_len:]
            hist_utter_npys = []
            for utterance in hist_utters:
                np_utter = convert_str_to_npy(utterance, self.vocab, self.max_len, eos=self.eos, pad=0,
                                              unk=self.unk)
                hist_utter_npys.append(np_utter)
            np_history = np.stack(hist_utter_npys, axis=0)
            # we need add pad utterances to get array up to final shape (history_len, max_len)
            np_pad = np.zeros([self.history_len - np_history.shape[0], self.max_len])
            np_history = np.concatenate([np_pad, np_history], axis=0)

        return np_history, np_response

    def __len__(self):
        # only half of the examples are real (1), the rest are fake (0)
        return min(self.responses.shape[0], self.max_examples)

    def __getitem__(self, index):
        """
        :param index: Index of example in dataset
        :return: history, response numpy arrays where response is of shape (max_len,) and history depends on split_history
        """
        np_history = self.histories[index]
        if not self.mismatch:
            np_response = self.responses[index]
            return np_history, np_response
        else:
            match = random.getrandbits(1)
            np_response = self.responses[index - 1 + match]  # grab current response or last response is mismatch
            return np_history, np_response, np.array(match)

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

    vocab_len = 50000
    max_len = 20
    history_len = 10
    max_examples = None
    mismatch = False
    split_history = False

    # problem: history cuts off front end, not tale end

    ds = UbuntuCorpus(args.source, args.save_path, vocab_len, max_len, history_len, max_examples=max_examples,
                      regen=args.regenerate, mismatch=mismatch,
                      split_history=split_history)

    print('Dataset length: %s' % len(ds))
    print('Vocab length: %s' % len(ds.vocab))

    # now we print actual examples from the dataset
    for i in range(10):
        result = ds[i]  # the index isn't used
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