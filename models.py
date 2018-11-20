"""Deep learning models coded in Pytorch."""
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    def __init__(self, d_emb, d_enc, d_vocab, d_dec, max_len, bos_idx):
        super().__init__()
        self.encoder = Encoder(d_emb, d_enc, d_vocab)
        self.decoder = Decoder(d_vocab, d_emb, d_dec, max_len, d_enc, bos_idx)

    def forward(self, x, labels=None, sample_func=None):
        e_states, e_final = self.encoder(x)
        return self.decoder(e_final, labels=labels, sample_func=sample_func)


class Encoder(nn.Module):
    def __init__(self, d_emb, d_enc, d_vocab):
        super().__init__()
        self.embs = nn.Embedding(d_vocab, d_emb)
        self.rnn = nn.GRU(d_emb, d_enc, batch_first=True)

    def forward(self, x):
        x_embs = self.embs(x)
        e_states, e_final = self.rnn(x_embs)
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


