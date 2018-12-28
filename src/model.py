"""Deep learning models coded in Pytorch."""
import src.config
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import random
import numpy as np

from ntm.aio import EncapsulatedNTM


# class Seq2SeqMem(nn.Module):
#     """memory enhanced model"""
#     def __init__(self, d_emb, d_sent_enc, d_vocab, d_turn_enc, d_dec, max_len, bos_idx):
#         super().__init__()
#         self.sentence_encoder = Encoder(d_emb, d_sent_enc, d_vocab)
#         self.rnn_turn = nn.GRU(d_sent_enc, d_turn_enc, batch_first=True)
#         self.decoder = Decoder(d_vocab, d_emb, d_dec, max_len, d_turn_enc, bos_idx)
#
#     def forward(self, x, labels=None, sample_func=None):
#         _, sent_finals = self.sentence_encoder(x) # sent_finals: # [batch, turns, d_sent_enc]
#         _, turn_finals = self.rnn_turn(sent_finals)
#         turn_finals = turn_finals.squeeze(0)  # [batch, d_turn_enc]
#
#         return self.decoder(turn_finals, labels=labels, sample_func=sample_func)

############  memory model #####################
class Seq2SeqMem(nn.Module):
    """memory enhanced model"""
    def __init__(self, d_emb, d_sent_enc, d_vocab, max_len, bos_idx):
        super().__init__()
        self.sentence_encoder = Encoder(d_emb, d_sent_enc, d_vocab)
        # self.rnn_turn = nn.GRU(d_sent_enc, d_turn_enc, batch_first=True)
        self.decoder = NTMDecoder(d_vocab, d_emb, d_sent_enc, max_len, d_sent_enc, bos_idx)

        self.ntm_a = EncapsulatedNTM(d_sent_enc, d_sent_enc, d_sent_enc, 1, 1, 20, d_sent_enc)
        self.ntm_b = EncapsulatedNTM(d_sent_enc, d_sent_enc, d_sent_enc, 1, 1, 20, d_sent_enc)
        self.model_type = 'seq2seqmem'

    def forward(self, x, labels=None, sample_func=None):
        batch_size, n_turns, input_length = x.size()
        sent_states, sent_finals = self.sentence_encoder(x) # states [batch, turns, sent_len, d_sent_enc]
        # print(sent_states.size())
        end_points = np.array([1, 2, 3, 4]) * input_length // 4

        self.ntm_a.init_sequence(batch_size)
        self.ntm_b.init_sequence(batch_size)
        for t in range(n_turns):
            state = sent_states[:, t, :, :]
            # print(state.size())
            for i in range(4):
                end = int(end_points[i])
                current_state = state[:, end-1, :]  # [batch, d_sent_enc]
                # print("current state", current_state.size())
                if t % 2 == 0:
                    self.ntm_a(current_state)
                else:
                    self.ntm_b(current_state)

        return self.decoder(self.ntm_a, self.ntm_b, sent_finals, labels=labels, sample_func=sample_func)

class Encoder(nn.Module):
    def __init__(self, d_emb, d_sent_enc, d_vocab):
        super().__init__()
        self.embs = nn.Embedding(d_vocab, d_emb)
        self.rnn = nn.GRU(d_emb, d_sent_enc, batch_first=True)

    def forward(self, x):
        x_embs = self.embs(x)
        batch_size, n_turns, sent_len, d_emb = x_embs.size()

        x_embs = x_embs.view(batch_size, -1, d_emb)  # [batch, turns*sent_len, d_emb]
        e_states, e_final = self.rnn(x_embs)
        # print(e_states.size(), e_final.size())
        e_states = e_states.view(batch_size, n_turns, sent_len, -1)  # [batch, turns, sent_len, d_sent_enc]
        e_final = e_final.view(batch_size, -1)  # [batch, d_sent_enc]
        return e_states, e_final


class NTMDecoder(nn.Module):
    def __init__(self, d_vocab, d_emb, d_dec, max_len, d_context, bos_idx):
        super().__init__()
        self.embs = nn.Embedding(d_vocab, d_emb)
        self.rnn = nn.GRUCell(d_emb + d_context, d_dec)
        self.init = nn.Parameter(torch.zeros(1, d_dec), requires_grad=True)
        self.bos_idx = nn.Parameter(torch.tensor([bos_idx]), requires_grad=False)
        self.linear = nn.Linear(2 * d_dec, d_vocab)
        self.max_len = max_len

    def forward(self, ntm_a, ntm_b, context, labels=None, sample_func=None):
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
            state = self.rnn(word, state)  # [batch, d_sen_enc]
            # print("grucell state", state.size())
            # logits = self.linear(state)
            ntm_a_out, _ = ntm_a(state)  # [batch, d_sent_enc]
            ntm_b_out, _ = ntm_b(state)
            logits = self.linear(torch.cat([ntm_a_out, ntm_b_out], -1))

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


############## Attention model ###################

class Seq2SeqAttn(nn.Module):
    """seq2seq model with attention
       Adapted from pytorch tutorial. may not be very efficient
    """
    def __init__(self, d_enc, d_vocab, max_len, bos_idx, eos_idx):
        super().__init__()
        self.encoder = AttnEncoder(d_vocab, d_enc)
        self.decoder = AttnDecoder(d_enc, d_vocab, max_len)
        self.device = torch.cuda.current_device()
        self.max_len = max_len
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.criterion = None
        self.optim = None
        self.model_type = 'seq2seqattn'

    def set_optimizer(self, lr=0.001):
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def fit(self, x, y, teacher_forcing_ratio=1.0):
        x, y = x.transpose(0, 1), y.transpose(0,1)  # [seq, batch, dim]

        input_length, batch_size = x.size()
        # print(input_length)
        encoder_hidden = self.encoder.initHidden(batch_size)
        encoder_outputs = torch.zeros(input_length, batch_size, self.encoder.hidden_size, device=self.device)

        # loss = 0
        self.optimizer.zero_grad()

        # collect input word embeddings
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[ei], encoder_hidden)
            encoder_outputs[ei,:,:] += encoder_output[0]

        decoder_input = torch.tensor([[self.bos_idx]], device=self.device).expand(1, batch_size)  # beginning

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        all_logits = []
        for di in range(self.max_len):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # loss += self.criterion(decoder_output, y[di])
            if use_teacher_forcing:
                decoder_input = y[di,:]  # Teacher forcing
                # print("y[di,:] size", y[di,:].size())
            else:
                predv, pred = decoder_output.topk(1)
                decoder_input = pred.squeeze().detach()  # detach from history as input
                # print("decoder_input size", decoder_input.size())
                # if decoder_input.item() == self.eos_idx:
                #     break
            all_logits.append(decoder_output)

        all_logits = torch.stack(all_logits, dim=0)
        # print(all_logits.size(), y.size())
        loss = self.criterion(all_logits.contiguous().view(self.max_len * batch_size, -1), y.contiguous().view(-1))

        loss.backward()
        self.optimizer.step()

        return loss

    def forward(self, x, y=None, sample_func=None):
        with torch.no_grad():
            x = x.transpose(0, 1)  # [seq, batch, dim]
            # print(x.size())
            if y is not None:
                y = y.transpose(0, 1)
            input_length, batch_size = x.size()
            encoder_hidden = self.encoder.initHidden(batch_size)
            encoder_outputs = torch.zeros(input_length, batch_size, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(x[ei], encoder_hidden)
                encoder_outputs[ei, :, :] += encoder_output[0]

                decoder_input = torch.tensor([[self.bos_idx]], device=self.device).expand(1, batch_size)  # beginning

            decoder_hidden = encoder_hidden

            all_logits = []
            all_preds = []
            # loss = 0
            # decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(self.max_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)  # decoder_output [batch, d_vocab]

                if sample_func is None:
                    predv, pred = decoder_output.topk(1)
                    # print(pred.size(), decoder_output.size())
                else:
                    pred = sample_func(decoder_output)
                    # print(pred.size(), decoder_output.size())

                if y is not None:
                    # loss += self.criterion(decoder_output, y[di])
                    decoder_input = y[di, :]  # use true label input for evaluation
                    # print("y[di,:] size", y[di,:].size())
                else:
                    decoder_input = pred.squeeze().detach()

                all_logits.append(decoder_output)
                all_preds.append(pred)

            all_logits = torch.stack(all_logits, dim=0)
            all_preds = torch.stack(all_preds, dim=0)
            if y is not None:
                loss = self.criterion(all_logits.contiguous().view(self.max_len * batch_size, -1), y.contiguous().view(-1))
                return loss, all_logits.transpose(0, 1), all_preds.transpose(0, 1)

            return all_logits.transpose(0, 1), all_preds.transpose(0, 1)  # batch first

class AttnEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttnEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        embedded = self.embedding(input).view(1, batch_size, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        # print("output size", output.size())
        return output, hidden  # output size [seq_len, batch, num_directions * hidden_size]

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=torch.cuda.current_device())


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size  # hidden size is the output size of encoder
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length * 10)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # print(input.size(), hidden.size())
        hidden_size = hidden.size(-1)
        batch_size = input.size(-1)  # input_len should be 1, for 1 step
        embedded = self.embedding(input).view(1, batch_size, -1)
        embedded = self.dropout(embedded)
        # print(embedded.size(), hidden.size())

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), -1)), dim=-1)
        # print(attn_weights.size(), encoder_outputs.size())
        attn_applied = torch.bmm(attn_weights.view(batch_size, 1, -1),
                                 encoder_outputs.view(batch_size, -1, hidden_size))

        # print(embedded.size(), attn_applied.size())
        output = torch.cat((embedded.view(batch_size, -1), attn_applied.view(batch_size, -1)), -1)  # [batch, 2*hidden_size]
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=torch.cuda.current_device())
