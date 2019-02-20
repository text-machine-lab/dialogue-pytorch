"""Functions and helper objects for processing text"""
import numpy as np

class Vocab:
    def __init__(self):
        """Object which allows a vocabulary of tokens to be produced from pythonic strings or lists of tokens.
        Creates a O(1) mapping from tokens to consecutive indices, and from indices to tokens. Keeps frequency counts
        of each token for vocab pruning. Vocab contains '' token as index zero by default. It will not be removed
        with pruning."""
        self.tk_to_idx = {}
        self.idx_to_tk = {}
        self.tk_to_count = {}  # will never be removed

    def add_docs(self, docs):
        """Add collection of documents/strings/lists to vocabulary."""
        for doc in docs:
            self.add_doc(doc)

    def add_doc(self, doc):
        """Adds tokens from a document/string/list to vocabulary.

        doc - string containing tokens
        """
        if isinstance(doc, str):
            doc = doc.split()
        for token in doc:
            if token not in self.tk_to_idx:
                self.tk_to_idx[token] = len(self.tk_to_idx)
                self.idx_to_tk[len(self.idx_to_tk)] = token
                self.tk_to_count[token] = 1
            else:
                self.tk_to_count[token] += 1

    def insert_token(self, token, index):
        """Inserts token at specific position in vocabulary. This can be
        used to indicate a padding token as the zeroth index. WARNING: pruning
        destroys the order of indices. Does not check if token already exists."""
        old = self.idx_to_tk[index]
        self.idx_to_tk[len(self)] = old
        self.idx_to_tk[index] = token
        self.tk_to_idx[old] = len(self)
        self.tk_to_idx[token] = index
        self.tk_to_count[token] = 1

    def prune(self, max_len, keep=None):
        """Prune vocabulary, removing lowest frequency tokens
        first. Will not have same indices per token as before!

        max_len - maximum number of tokens to keep
        keep - list of tokens to keep regardless of max_len or frequency
        """
        #import pdb; pdb.set_trace()
        # sort tokens and counts, grab top max_len
        tokens = list(self.tk_to_count.keys())
        counts = [self.tk_to_count[token] for token in tokens]
        max_indices = np.argsort(counts)[-max_len:].tolist()
        max_counts = [counts[idx] for idx in max_indices]
        max_tokens = [tokens[idx] for idx in max_indices]

        # keep particular tokens
        if keep is not None:
            keep_tokens = keep
            keep_counts = [self.tk_to_count[token] for token in keep_tokens]
            max_counts += keep_counts
            max_tokens += keep_tokens

        # rebuild vocabulary from tokens and counts
        self.tk_to_count = {}
        self.tk_to_idx = {}
        self.idx_to_tk = {}
        for token, count in zip(max_tokens, max_counts):
            self.tk_to_count[token] = count
            self.tk_to_idx[token] = len(self.tk_to_idx)
            self.idx_to_tk[len(self.idx_to_tk)] = token

    def __getitem__(self, item):
        """This is the benefit of the vocabulary. Can index into vocabulary
        with a token, and get an index. In the reverse direction, can index into vocabulary
        with an index and get a token."""
        if isinstance(item, str):
            return self.tk_to_idx[item]
        else:
            return self.idx_to_tk[item]

    def __contains__(self, item):
        """Check if vocabulary contains token or index."""
        return item in self.tk_to_idx or item in self.idx_to_tk

    def __len__(self):
        return len(self.tk_to_idx)

    def __str__(self):
        return str(self.tk_to_idx)


def raw_count(filename):
    """Extremely fast line count."""
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines


def convert_str_to_npy(string, vocab, max_len, eos=None, pad=0, unk=None, left_pad=False):
    """Convert a string into a numpy vector containing
    indices which represent each token, as indicated by a vocabulary. If
    unk token is not provided, throws error if token does not appear in vocab.

    string - str object (preferably tokenized)
    vocab - mapping from tokens to indices
    max_len - length of array
    pad - index used to pad extra tokens
    unk - token to use if token does not appear in vocab
    """
    if isinstance(string, str):
        # could be input list of tokens
        string = string.split()
    if eos is not None:
        string.append(eos)
    shift = max(max_len - len(string), 0)
    output = np.full(max_len, pad, np.int)
    for i, token in enumerate(string):
        if i < max_len:
            if left_pad:
                ptr = shift + i  # if left pad, shift all indices over to right side
            else:
                ptr = i
            if token in vocab:
                output[ptr] = vocab[token]
            elif unk is not None:
                output[ptr] = vocab[unk]
            else:
                raise ValueError('Unknown token not specified. Tokens must be in vocabulary.')
        else:
            break
    return output


def convert_npy_to_str(vector, vocab, eos=None):
    """Convert a numpy vector into a string, by mapping
    indices in the array to tokens using a vocabulary object.

    vector - numpy array of shape (max_len,) filled with token indices
    vocab - mapping from indices to tokens
    stop - token to indicate stop of string creation (ignore later indices)
    """
    output = []
    for i in range(vector.size):
        token = vocab[vector[i]]
        output.append(token)
        if eos is not None and token == eos:
            break
    return ' '.join(output)








