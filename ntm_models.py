"""Contains model code which uses the EncapsulatedNTM module from the pytorch-ntm Github repo. Clone the repo,
and add the ntm module to your working , or add to your Python path!"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from ntm.aio import EncapsulatedNTM  # found in the pytorch-ntm Github repo


class Decoder(nn.Module):
    def __init__(self, d_vocab, d_emb, d_dec, max_len, d_context, bos_idx, num_inputs, num_outputs,
                 controller_size, controller_layers, num_heads, N, M, seg_size=10):
        super().__init__()
        self.d_vocab = d_vocab
        self.seg_size = seg_size
        self.embs = nn.Embedding(d_vocab, d_emb)
        self.rnn = nn.GRU(d_emb + d_context, d_dec, batch_first=True)
        self.ntm = EncapsulatedNTM(num_inputs, num_outputs, controller_size, controller_layers, num_heads, N, M)
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
            #final_states = batch_index3d(outputs.contiguous(), prelabel_lens)  # b x h
            final_states = outputs[:, -1, :]
            state = final_states  # b x h

        if labels is not None:
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
        else:
            all_logits = []
            all_preds = []
            if prelabels is None:
                word = init.squeeze(1)  # b x w
            else:
                logits = self.linear(state)  # b x v
                all_logits.append(logits)
                pred = sample_func(logits)  # b
                all_preds.append(pred)
                word = self.embs(pred)  # b x w
            state = state.unsqueeze(0)

            for step in range(num_steps):
                word = word.unsqueeze(1)  # 1 x b x w
                _, state = self.rnn(word, state)  # 1 x b x h
                logits = self.linear(state.squeeze(0))  # b x v
                all_logits.append(logits)
                pred = sample_func(logits)  # b
                word = self.embs(pred)  # b x w
                all_preds.append(pred)
            logits = torch.stack(all_logits, dim=1)
            return logits, torch.stack(all_preds, dim=1)
