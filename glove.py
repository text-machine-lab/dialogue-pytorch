"""Loader for Glove embeddings, builds glove matrix from vocab."""
import numpy as np
from nlputils import raw_count
from tqdm import tqdm

class GloveLoader:
    def __init__(self, glove_file):
        self.glove_file = glove_file

    def build_embeddings(self, vocab, factor=0.1):
        """
        Builds an embedding matrix for all tokens in the vocabulary. Tokens for
        which glove embeddings exist will be represented as glove embeddings. Missing tokens are
        randomly initialized.

        :param vocab: Vocab object from nlputils.py mapping words-->indices and visa versa.
        :return: Numpy matrix (vocab_len, emb_size)
        """
        embs = None

        line_count = raw_count(self.glove_file)
        num_embs_found = 0

        with open(self.glove_file, 'r') as f:
            for line in tqdm(f, total=line_count):
                entries = line.strip().split()
                word = entries[0]
                embedding = entries[1:]

                # we wait for first line to calculate embedding size, then build array
                if embs is None:
                    embs = np.random.rand(len(vocab), len(embedding)) * factor

                if word in vocab:
                    embedding = np.array(embedding)
                    embs[vocab[word]] = embedding
                    num_embs_found += 1

        print('Fraction embeddings found: %s' % (num_embs_found / line_count))

        return embs


