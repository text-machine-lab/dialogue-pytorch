"""Deep learning models coded in Pytorch."""
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


class MismatchSeq2Seq(nn.Module):
    def __init__(self, d_emb, d_enc, d_vocab, d_dec, max_len, bos_idx, context_size=0, dropout=0.1):
        super().__init__()
        self.mismatch = MismatchClassifier(d_emb, d_enc, d_vocab, dropout=dropout)
        self.decoder = Decoder(d_vocab, d_emb, d_dec, max_len, d_enc + context_size, bos_idx)

    def forward(self, x, labels=None, context=None, sample_func=None):

        logit = None
        if labels is None:
            m_vector = self.mismatch.encode_message(x)
        else:
            logit, m_vector, r_vector = self.mismatch(x, labels, return_vectors=True)

        # allow user to condition the decoder on external context
        if context is not None:
            m_vector = torch.cat([m_vector, context], dim=-1)

        decoder_result = self.decoder(m_vector, labels=labels, sample_func=sample_func)

        # if labels are available, compute output of mismatch classifier
        if labels is not None:
            return decoder_result, logit
        else:
            return decoder_result



class Seq2Seq(nn.Module):
    def __init__(self, d_emb, d_enc, d_vocab, d_dec, max_len, bos_idx, context_size=0):
        super().__init__()
        self.encoder = Encoder(d_emb, d_enc, d_vocab)
        self.decoder = Decoder(d_vocab, d_emb, d_dec, max_len, d_enc + context_size, bos_idx)

    def forward(self, x, labels=None, context=None, sample_func=None):
        """
        Run a sequence-to-sequence model on input sequence. Model learns
        embeddings for vocabulary internally.

        :param x: (batch_size, num_steps) containing indices per token in each sequence
        :param labels: (batch_size, max_len) containing label indices used for teacher forcing
        :param context: (batch_size, context_size) containing extra information for decoder (optional)
        :param sample_func: function mapping logit tensor (batch_size, d_vocab) --> output tensor (batch_size,)
        :return: logits tensor (batch_size, max_len, d_vocab), and predictions tensor (batch_size, max_len) if
        labels are not provided (inference)
        """

        e_states, e_final = self.encoder(x)

        # allow user to condition the decoder on external context
        if context is not None:
            e_final = torch.cat([e_final, context], dim=-1)

        return self.decoder(e_final, labels=labels, sample_func=sample_func)


class Encoder(nn.Module):
    def __init__(self, d_emb, d_enc, d_vocab, dropout=0.0):
        super().__init__()
        self.embs = nn.Embedding(d_vocab, d_emb)
        self.rnn = nn.GRU(d_emb, d_enc, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Runs GRU encoder over input tokens.
        :param x: indices tensor (batch_size, max_len) or embeddings (batch_size, max_len, d_emb)
        :return: encoder states (batch_size, max_len, d_enc) and final state (batch_size, d_enc)
        """
        if len(x.shape) == 2:
            x = self.embs(x)
        x = self.dropout(x)
        e_states, e_final = self.rnn(x)
        e_final = e_final.squeeze(0)
        return e_states, e_final


def random_sample(x):
    """
    Sample indices from a matrix of logits based on the distribution they define.
    :param x: Tensor shape (batch_size, num_logits), operation is per example in batch.
    :return: Tensor shape (batch_size,)
    """
    result = torch.multinomial(F.softmax(x, -1), 1).squeeze(-1)
    return result


class Decoder(nn.Module):
    def __init__(self, d_vocab, d_emb, d_dec, max_len, d_context, bos_idx):
        super().__init__()
        self.embs = nn.Embedding(d_vocab, d_emb)
        self.rnn = nn.GRUCell(d_emb + d_context, d_dec)
        self.init = nn.Parameter(torch.zeros(1, d_dec), requires_grad=True)
        self.bos_idx = nn.Parameter(torch.tensor([bos_idx]), requires_grad=False)
        self.linear = nn.Linear(d_dec, d_vocab)
        self.max_len = max_len

    def forward(self, context, labels=None, sample_func=None):
        if isinstance(context, int):
            # allow decoder to function as NLM
            b = context
            context = None
        else:
            # number of contexts denotes batch size
            b = context.shape[0]

        if sample_func is None:
            sample_func = partial(torch.argmax, dim=-1)

        t = self.max_len
        state = self.init.expand(b, -1)  # repeat across batch dimension
        word = self.embs(self.bos_idx.expand(b))
        all_logits = []
        all_preds = []
        for step in range(t):
            if context is not None:
                word = torch.cat([word, context], dim=-1)
            state = self.rnn(word, state)
            logits = self.linear(state)
            all_logits.append(logits)
            if labels is not None:
                word = self.embs(labels[:, step])
            else:
                pred = sample_func(logits)
                word = self.embs(pred)
                all_preds.append(pred)
        logits = torch.stack(all_logits, dim=1)
        if labels is None:
            return logits, torch.stack(all_preds, dim=1)
        else:
            return logits



class MismatchClassifier(nn.Module):
    def __init__(self, d_emb, d_enc, d_vocab, dropout=0.1):
        super().__init__()
        self.m_enc = Encoder(d_emb, d_enc, d_vocab, dropout=dropout)
        self.r_enc = Encoder(d_emb, d_enc, d_vocab, dropout=dropout)
        self.m_linear = nn.Linear(d_enc, d_enc)
        self.r_linear = nn.Linear(d_enc, d_enc)
        self.out_linear = nn.Linear(d_enc * 4, 1)

    def encode_message(self, x):
        """
        Encode x using encoder, unless x is greater than two-dimensional, then assume already encoded.
        :param x:
        :return:
        """
        _, m_enc_out = self.m_enc(x)

        return self.m_linear(m_enc_out)

    def encode_response(self, y):
        _, r_enc_out = self.r_enc(y)

        return self.r_linear(r_enc_out)

    def forward(self, x, y, return_vectors=False):
        # read in message and produce a vector
        m_vector = self.encode_message(x)
        r_vector = self.encode_response(y)

        diff = (m_vector - r_vector).abs()
        mul = m_vector * r_vector

        features = torch.cat([m_vector, r_vector, diff, mul], dim=-1)

        comparison = self.out_linear(features).squeeze(-1)

        if return_vectors:
            return comparison, m_vector, r_vector
        else:
            return comparison
















