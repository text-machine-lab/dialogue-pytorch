"""Contains model code which uses the EncapsulatedNTM module from the pytorch-ntm Github repo. Clone the repo,
and add the ntm module to your working , or add to your Python path!"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from ntm import NTM


class NTMAugmentedDecoder(nn.Module):
    def __init__(self, d_vocab, d_emb, d_dec, max_len, bos_idx, num_heads=8, N=64, M=32, seg_size=20):

        super().__init__()
        self.d_vocab = d_vocab
        self.seg_size = seg_size
        self.embs = nn.Embedding(d_vocab, d_emb)
        self.rnn = nn.GRU(d_emb, d_dec, batch_first=True)
        self.ntm_scale = nn.Parameter(torch.zeros([1, d_dec]), requires_grad=True)
        self.ntm = NTM('mem-aug', embedding_size=d_dec, hidden_size=d_dec, memory_size=M, head_num=num_heads,
                       memory_feature_size=N, output_size=d_dec)
        self.init = nn.Parameter(torch.zeros(1, d_dec), requires_grad=True)
        self.bos_idx = nn.Parameter(torch.tensor([bos_idx]), requires_grad=False)
        self.out_layer = nn.Linear(d_dec, d_vocab)
        self.max_len = max_len

    def forward(self, labels, state=None):
        """
        Run decoder on input context with optional conditioning on labels and prelabels
        :param labels: labels conditioned on at each timestep in next-prediction task (used during training)
        :param state: initial state to begin decoding with. Could be output of encoder.
        :return: If labels are provided, returns logits tensor (batch_size, num_steps, vocab_size). If labels are not provided,
        returns predictions tensor (batch_size, num_steps) using provided sampling function.
        """
        batch_size = labels.shape[0]
        self.ntm.reset(batch_size, device=labels.device)
        if state is None:
            state = self.init.expand(batch_size, -1).contiguous()  # bxh

        init = self.embs(self.bos_idx.expand(batch_size).unsqueeze(1))  # bx1xh

        # initialize ntm state, which keeps track of reads and writes
        ntm_state = torch.zeros_like(state).to(state.device)

        labels_no_last = labels[:, :-1]  # b x (t-1) # we don't take last word as input
        # break input into slices, read and write between slices
        num_slices = labels.shape[1] // self.seg_size
        all_logit_slices = []

        for slice in range(num_slices):
            # grab slice of input
            labels_slice = labels_no_last[:, slice*self.seg_size:slice*self.seg_size+self.seg_size]
            in_embs = self.embs(labels_slice)  # b x (t-1) x w

            if slice == 0:  # add bos index on first iteration
                in_embs = torch.cat([init, in_embs], dim=1)  # b x t x w

            # give ntm state as input to all time steps of next slice
            # multiple ntm state by scalars before giving to RNN, so it is not used in the beginning of training
            #scaled_ntm_state = ntm_state * self.ntm_scale
            #exp_ntm_state = scaled_ntm_state.unsqueeze(1).expand([-1, in_embs.shape[1], -1])  # b x t x h
            rnn_input = in_embs  # torch.cat([in_embs, exp_ntm_state], dim=-1)
            # read slice of conversation history, with access to ntm state
            outputs, _ = self.rnn(rnn_input, state.unsqueeze(0))  # b x (t-1) x h OR b x t x h
            # grab last state and use it to read and write from ntm
            state = outputs[:, -1, :]
            #ntm_state = self.ntm(state)
            # predict outputs for this slice
            logits = self.out_layer(outputs)  # b x t x v
            all_logit_slices.append(logits)
        # append predictions for all slices together
        logits = torch.cat(all_logit_slices, dim=1)

        return logits


    def complete(self, x, state=None, sample_func=None):
        """
        Given tensor x containing token indices, fill in all padding token (zero) elements
        with predictions from the NTM decoder.
        :param x: (batch_size, num_steps) tensor containing token indices
        :return: tensor same shape as x, where zeros have been filled with decoder predictions
        """
        batch_size, num_steps = x.shape
        self.ntm.reset(batch_size, device=x.device)
        if state is None:
            state = self.init.expand(batch_size, -1).contiguous()  # bxh
        if sample_func is None:
            sample_func = partial(torch.argmax, dim=-1)

        ntm_state = torch.zeros_like(state).to(state.device)

        all_logits = []
        all_preds = []
        init = self.embs(self.bos_idx.expand(batch_size).unsqueeze(1))  # bx1xh
        word = init.squeeze(1)  # b x w

        for step in range(num_steps):
            # run RNN over input words
            rnn_input = word.unsqueeze(1)  # torch.cat([word, ntm_state], dim=-1).unsqueeze(1)
            _, state = self.rnn(rnn_input, state.unsqueeze(0))  # 1 x b x h
            state = state.squeeze(0)

            # produce prediction at each time step
            logits = self.out_layer(state)  # b x v
            all_logits.append(logits)
            pred = sample_func(logits)  # b

            # # at the end of each segment, read and write from NTM
            # if step % self.seg_size == (self.seg_size - 1):
            #     # end of each segment, read and write from NTM
            #     ntm_state = self.ntm(state)

            # here, we grab word from x if it exists, otherwise use prediction
            mask = (x[:, step] != 0).long()  # b
            word_index = x[:, step] * mask + pred * (1 - mask)  # use label or prediction, whichever is available
            word = self.embs(word_index)  # b x w

            all_preds.append(word_index)
        logits = torch.stack(all_logits, dim=1)
        return logits, torch.stack(all_preds, dim=1)