"""Contains the reddit dataset."""
from torch.utils.data import Dataset
from nlputils import convert_str_to_npy, Vocab
from nltk import word_tokenize
from tqdm import tqdm
import pickle
import os

class RedditDataset(Dataset):
    def __init__(self, path, vocab_path, max_doc, max_title, max_history, max_response, regen=False, max_vocab=None,
                 max_lines_vocab_gen=1000):
        """Create a Dataset object capable of loading the Reddit dataset.

        path - location of Reddit .tsv file containing (comment_title, num_comments, **comments, doc_title, doc) lines
        vocab_path - location to dump generated vocabulary file for faster loading
        max_doc - max length of document in tokens
        max_title - max length of conversation and document titles in tokens
        max_history - max length of appended previous comments before current response comment, in tokens
        max_response - max length of comment selected to be response to all previous comments, in tokens
        regen - if True, regenerate vocabulary and save
        max_vocab - maximum number of tokens in vocabulary
        max_line_vocab_gen - prune each line to this size for faster vocab creation"""
        # count number of examples
        self.num_examples = sum([int(line.split('\t')[1]) for line in open(path, 'r')])
        self.max_doc = max_doc  # max size of document in tokens (with stop)
        self.max_history = max_history
        self.max_response = max_response
        self.max_title = max_title
        self.max_lines_vocab_gen = max_lines_vocab_gen
        self.eos = '<eos>'
        self.unk = '<unk>'
        self.eoc = '<eoc>'  # end of comment
        if not os.path.exists(vocab_path) or regen:
            self.vocab = self.tokenize_and_build_vocab(path, max_vocab=max_vocab)
            pickle.dump(self.vocab, open(vocab_path, 'wb'))
        else:
            self.vocab = pickle.load(open(vocab_path, 'rb'))
        # open file for reading
        self.file = open(path, 'r')
        # we plan to load each thread, and produce an example from each comment
        # we store details of the thread here
        self.utter_index = 0  # current utterance to produce example from
        self.conv_title = None
        self.comments = None
        self.doc_title = None
        self.doc = None

    def tokenize_and_build_vocab(self, path, max_vocab=None):
        """Read through all examples, tokenize them and compile vocabulary of top-frequency tokens.

        Returns: Vocab object built from tokens in path file. """
        vocab = Vocab()
        bar = tqdm(open(path, 'r'))
        for i, line in enumerate(bar):
            if i > self.max_lines_vocab_gen:
                bar.close()
                break
            line = line.lower()
            tokens = word_tokenize(line[:self.max_doc])
            vocab.add_doc(tokens)
        if max_vocab is not None:
            vocab.prune(max_vocab)
        vocab.add_doc(self.eos)
        vocab.add_doc(self.unk)
        vocab.add_doc(self.eoc)
        vocab.insert_token('', 0)  # get that padding token in there
        return vocab

    def load_example(self):
        """Load next example from input file."""
        line = next(self.file).lower()
        #import pdb; pdb.set_trace()
        entries = line.split('\t')
        conv_title = word_tokenize(entries[0])
        num_comments = int(entries[1])
        comments = [' '.join(word_tokenize(comment)) for comment in entries[2:2+num_comments]]
        doc_title = word_tokenize(entries[2+num_comments])[:self.max_title]
        doc = word_tokenize(' '.join(entries[3+num_comments].split()[:self.max_doc]))
        return conv_title, comments, doc_title, doc

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        if self.comments is None or self.utter_index >= len(self.comments):
            # if we have created examples from all comments, load the next thread
            self.utter_index = 0
            self.conv_title, self.comments, self.doc_title, self.doc = self.load_example()

        # format example
        np_conv_title = convert_str_to_npy(self.conv_title, self.vocab, self.max_title, eos=self.eos, unk=self.unk)
        separator = ' ' + self.eoc + ' '
        history = separator.join(self.comments[:self.utter_index])
        np_history = convert_str_to_npy(history, self.vocab, self.max_history, eos=self.eos, unk=self.unk)
        np_comment = convert_str_to_npy(self.comments[self.utter_index], self.vocab, self.max_response, eos=self.eos, unk=self.unk)
        self.utter_index += 1
        np_doc_title = convert_str_to_npy(self.doc_title, self.vocab, self.max_title, eos=self.eos, unk=self.unk)
        np_doc = convert_str_to_npy(self.doc, self.vocab, self.max_doc, eos=self.eos, unk=self.unk)

        return np_history, np_comment, np_conv_title, np_doc_title, np_doc



