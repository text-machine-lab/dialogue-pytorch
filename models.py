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

        # do not send gradients decoder
        with torch.no_grad():
            m_vector = self.mismatch.encode_message(x)

        # allow user to condition the decoder on external context
        if context is not None:
            m_vector = torch.cat([m_vector, context], dim=-1)

        decoder_result = self.decoder(m_vector, labels=labels, sample_func=sample_func)

        # if labels are available, compute output of mismatch classifier
        return decoder_result



class Seq2Seq(nn.Module):
    def __init__(self, d_emb, d_enc, d_vocab, d_dec, max_len, bos_idx):
        super().__init__()
        self.encoder = Encoder(d_emb, d_enc, d_vocab)
        self.decoder = Decoder(d_vocab, d_emb, d_dec, max_len, d_enc, bos_idx)

    def forward(self, x, labels=None, sample_func=None):
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

        return self.decoder(labels=labels, sample_func=sample_func, state=e_final)


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
        self.d_vocab = d_vocab
        self.embs = nn.Embedding(d_vocab, d_emb)
        self.rnn = nn.GRU(d_emb + d_context, d_dec, batch_first=True)
        self.init = nn.Parameter(torch.zeros(1, d_dec), requires_grad=True)
        self.bos_idx = nn.Parameter(torch.tensor([bos_idx]), requires_grad=False)
        self.linear = nn.Linear(d_dec, d_vocab)
        self.max_len = max_len

    def forward(self, prelabels=None, labels=None, state=None, sample_func=None, batch_size=None, num_steps=None):
        """
        Run decoder on input context with optional conditioning on labels and prelabels
        :param context: None if no context, otherwise concatenated as input to decoder, or integer b for unconditional generation
        :param prelabels: run decoder on these labels before running on labels (complete the sequence)
        :param labels: labels conditioned on at each timestep in next-prediction task (used during training)
        :param sample_func: specify a function logits (batch_size, vocab_size)--> predictions (batch_size,)
        for model sampling. Default: argmax
        :return: If labels are provided, returns logits tensor (batch_size, num_steps, vocab_size). If labels are not provided,
        returns predictions tensor (batch_size, num_steps) using provided sampling function.
        """
        if batch_size is None:
            if labels is not None:
                batch_size = labels.shape[0]
            else:
                batch_size = prelabels.shape[0]
        if num_steps is None:
            num_steps = self.max_len
        if state is None:
            state = self.init.expand(batch_size, -1).contiguous()  # bxh

        if sample_func is None:
            sample_func = partial(torch.argmax, dim=-1)

        init = self.embs(self.bos_idx.expand(batch_size).unsqueeze(1))  # bx1xh

        # give ability to run model on prelabels before actual labels
        # we run on the prelabels, then extract the final states after reading all
        # non-padding words
        if prelabels is not None:
            state = state.unsqueeze(0)  # 1 x b x h
            in_embs = self.embs(prelabels)  # b x t x w
            in_embs = torch.cat([init, in_embs], dim=1)  # b x (t+1) x w
            outputs, _ = self.rnn(in_embs, state)  # b x (t+1) x h
            prelabel_lens = (prelabels != 0).sum(dim=1)  # b
            final_states = batch_index3d(outputs.contiguous(), prelabel_lens)  # b x h
            state = final_states  # b x h

        labels_no_last = labels[:, :-1] # b x (t-1) # we don't take last word as input
        in_embs = self.embs(labels_no_last)  # b x (t-1) x w
        # don't append bos if we have already read it with prelabels
        if prelabels is None:
            in_embs = torch.cat([init, in_embs], dim=1)  # b x t x w
        outputs, _ = self.rnn(in_embs, state.unsqueeze(0))  # b x (t-1) x h OR b x t x h
        if prelabels is not None:
            outputs = torch.cat([state.unsqueeze(1), outputs], dim=1)  # b x t x h
        logits = self.linear(outputs)  # b x t x v
        return logits


    def complete(self, x, state=None, sample_func=None):
        """
        Given tensor x containing token indices, fill in all padding token (zero) elements
        with predictions from the decoder.
        :param x: (batch_size, num_steps) tensor containing token indices
        :return: tensor same shape as x, where zeros have been filled with decoder predictions
        """
        batch_size = x.shape[0]
        num_steps = x.shape[1]
        # create a initial state
        if state is None:
            state = self.init.expand(batch_size, -1).contiguous()  # bxh
        # default to argmax sampling function if none is specified
        if sample_func is None:
            sample_func = partial(torch.argmax, dim=-1)
        # initialize bos embedding to append to beginning of sequence
        init = self.embs(self.bos_idx.expand(batch_size).unsqueeze(1))  # bx1xh

        all_logits = []
        all_preds = []
        word = init.squeeze(1)  # b x w
        state = state.unsqueeze(0) # b x 1 x h

        for step in range(num_steps):
            word = word.unsqueeze(1)  # b x 1 x w
            _, state = self.rnn(word, state)  # 1 x b x h
            logits = self.linear(state.squeeze(0))  # b x v
            all_logits.append(logits)
            pred = sample_func(logits)  # b

            # here, we grab word from x if it exists, otherwise use prediction
            mask = (x[:, step] != 0).long()  # b
            word_index = x[:, step] * mask + pred * (1 - mask)
            word = self.embs(word_index)  # b x w

            all_preds.append(word_index)
        logits = torch.stack(all_logits, dim=1)
        predictions = torch.stack(all_preds, dim=1)
        return logits, predictions


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


def batch_index3d(x, i):
    b = x.shape[0]
    t = x.shape[1]
    e = x.shape[2]
    x_flat = x.view(b * t, e)
    r = torch.arange(b).to(x.device) * t  # index adjustments
    i_adj = i + r
    return x_flat[i_adj].view(b, e)


def batch_index(x, i):
    b = x.shape[0]
    t = x.shape[1]
    x_flat = x.view(b * t)
    r = torch.arange(b) * t  # index adjustments
    i_adj = i + r
    return x_flat[i_adj].view(b)


if __name__ == '__main__':

    # test
    x = torch.rand(32, 20, 200)
    i = (torch.rand(32) * 20).long()

    result = batch_index(x, i)

    assert result.shape == (x.shape[0], x.shape[2])
    for j in range(x.shape[0]):
        assert torch.equal(x[j, i[j]], result[j])












