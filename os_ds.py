"""OpenSubtitles dataset preprocessed using the torch.utils.data.Dataset class."""
from torch.utils.data import Dataset
from nlputils import Vocab, convert_str_to_npy, convert_npy_to_str
import argparse
from tqdm import tqdm
import numpy as np
import os
import random
import traceback
import pickle

class OpenSubtitlesDataset(Dataset):
    def __init__(self, source_dir, max_len, history_len, vocab_len, save_path, max_examples=None, regen=False):
        """
        Performs preprocessing on the open subtitles dataset to produce a vocabulary and prepares
        the number of lines in the input data files. Sets up the dataset for producing individual examples.
        Each example is of the form (history, response).

        :param source_dir: directory containing all input source files. each file contains one utterance per line
        :param max_len: maximum length of utterance to store in numpy embedding. extra tokens are pruned
        :param history_len: number of previous messages to use as history for a given response
        :param vocab_len: prune vocabulary to this size for dealing with unknown tokens
        :param save_path: place to save vocabulary/number of lines for faster loading
        :param max_examples: prune number of examples to this number for smaller testing
        :param regen: if true, force regeneration of vocabulary and calculated number of lines
        """
        super().__init__()
        self.max_len = max_len
        self.history_len = history_len
        self.max_examples = max_examples
        self.source_dir = source_dir
        self.vocab_len = vocab_len
        self.sources = None
        self.eos = '<eos>'
        self.unk = '<unk>'
        self.bos = '<bos>'

        # we open a bunch of data files, and read randomly from each file
        self.sources = [open(os.path.join(source_dir, f), 'r', encoding='utf-8') for f in os.listdir(source_dir)
                        if os.path.isfile(os.path.join(source_dir, f))]

        if save_path is None or not os.path.exists(save_path) or regen:
            # regenerate vocabulary
            self.vocab = Vocab()
            self.num_lines = 0
            for f in self.sources:
                for line in tqdm(f):
                    self.num_lines += 1
                    self.vocab.add_doc(line.strip().lower())
            self.vocab.prune(self.vocab_len - 4)
            self.vocab.insert_token('<pad>', 0)
            self.vocab.add_doc(self.unk)
            self.vocab.add_doc(self.eos)
            self.vocab.add_doc(self.bos)
            # save vocabulary and number of lines
            if save_path is not None:
                pickle.dump([self.num_lines, self.vocab], open(save_path, 'wb'))
        else:
            # load vocabulary and number of lines
            self.num_lines, self.vocab = pickle.load(open(save_path, 'rb'))

        self.load_sources(source_dir)

    def load_sources(self, source_dir):
        """
        Load data files containing dialogue utterances, one utterance per line.

        :param source_dir: Directory containing source files
        """
        if self.sources is not None:
            for source in self.sources:
                source.close()

        self.sources = [open(os.path.join(source_dir, f), 'r', encoding='utf-8') for f in os.listdir(source_dir)
                        if os.path.isfile(os.path.join(source_dir, f))]

        self.num_examples = self.num_lines // (self.history_len + 1) - 2 * len(self.sources)

        if self.max_examples is not None:
            self.num_examples = min(self.max_examples, self.num_examples)

    def __len__(self):
        """
        :return: The number of (history, response) examples in the dataset.
        """
        # we could lose an example at the end of each file, so we remove that many examples
        return self.num_examples

    def __getitem__(self, item):
        """
        Overrides Dataset method to allow dispensing of individual data examples. Ignores item
        index and instead reads from multiple data files simultaneously. Example: instead of randomly
        shuffling a large dataset, split into 10 pieces and grab new examples randomly from one of the
        10 pieces.

        :param item: not used
        :return: a (history, response) example where history is a numpy array of size (max_len * history_len,)
        and response is a numpy array of size (max_len,).
        """

        # here we randomly pick a file
        # if the file is empty, we discard it
        if len(self.sources) == 0:
            self.load_sources(self.source_dir)
        history = []
        index = random.randrange(len(self.sources))
        source = self.sources[index]

        # grab len_history messages
        for i in range(self.history_len + 1):
            message = source.readline()

            if message == "":
                self.sources.remove(source)
                #print('Removed source: %s' % source.name)
                # try getting an example from a different source
                return self.__getitem__(item)
            else:
                history.append(message.strip().lower())

        # grab response as last message
        response = history[-1]
        history = ' '.join(history[:-1])

        # build arrays

        np_history = convert_str_to_npy(history, self.vocab, self.max_len * self.history_len, eos=self.eos, pad=0, unk=self.unk)
        np_response = convert_str_to_npy(response, self.vocab, self.max_len, eos=self.eos, pad=0, unk=self.unk)

        return np_history, np_response


############ TESTING ###################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=None)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--regenerate', default=False, action='store_true')
    args = parser.parse_args()

    max_len = 20
    history_len = 10
    vocab_len = 10000
    max_examples = None

    ds = OpenSubtitlesDataset(args.source, max_len, history_len, vocab_len, args.save_path, regen=args.regenerate,
                              max_examples=max_examples)

    print('Vocab size: %s' % len(ds.vocab))
    print('Dataset size: %s' % len(ds))
    print('Num sources: %s' % len(ds.sources))

    result = []
    for i in tqdm(range(len(ds))):
        np_history, np_response = ds[i]

        if i < 10:
            history = convert_npy_to_str(np_history, ds.vocab, ds.eos)
            response = convert_npy_to_str(np_response, ds.vocab, ds.eos)
            result.append((history, response))

    for (history, response) in result:
        print(history)
        print(response)
        print()








