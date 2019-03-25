"""Train a language model augmented with neural turing machine, on the Ubuntu dataset."""

import argparse
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ubuntu import UbuntuCorpus
from models import random_sample
from ntm_models import NTMAugmentedDecoder
from tensorboard_logger import configure, log_value
from tqdm import tqdm
from tgalert import TelegramAlert
from utils import load_train_args, print_numpy_examples, gather_logits, move_prob_from_s_to_eos, replace_eos_slashs, \
    gather_response
from run_lm import eval_perplexity, print_examples, train

alert = TelegramAlert()

def setup():
    parser = argparse.ArgumentParser(description='Run ntm model on Opensubtitles conversations')
    load_train_args(parser)
    args = parser.parse_args()

    if args.tgdisable:
        alert.disable = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.run is not None and args.epochs > 0:
        configure(os.path.join(args.run, str(datetime.now())), flush_secs=5)

    history_len = 170
    max_len = 30
    max_examples = None if args.max_examples is None else int(args.max_examples)
    max_vocab_examples = None
    max_vocab = 50000
    num_epochs = args.epochs
    num_print = int(args.num_print)
    d_emb = 200
    d_dec = 400
    lr = .0001

    print('Using Ubuntu dialogue corpus')
    ds = UbuntuCorpus(args.source, args.temp, max_vocab, max_len, history_len,
                      concat_feature=True, max_examples=max_examples, regen=args.regen)

    print('Printing statistics for training set')
    ds.print_statistics()

    # use validation set if provided
    valds = None
    if args.val is not None:
        print('Using validation set for evaluation')
        if args.tempval is None:
            args.tempval = os.path.join(args.temp, 'validation')
        valds = UbuntuCorpus(args.val, args.tempval, max_vocab, max_len, history_len, concat_feature=True, regen=args.regenval,
                          vocab=ds.vocab, max_examples=max_examples)
        print('Printing statistics for validation set')
        valds.print_statistics()
    else:
        print('Using training set for evaluation')
        valds = ds

    if args.regen: alert.write('run_lm: Building datasets complete')

    print('Num examples: %s' % len(ds))
    print('Vocab length: %s' % len(ds.vocab))

    model = NTMAugmentedDecoder(len(ds.vocab), d_emb, d_dec, history_len + max_len, bos_idx=ds.vocab[ds.bos])
    #model = Decoder(len(ds.vocab), d_emb, d_dec, history_len + max_len, d_context=0, bos_idx=ds.vocab[ds.bos])

    if args.restore and args.model_path is not None:
        print('Restoring model from save')
        model.load_state_dict(torch.load(args.model_path))

    hyperparams = [history_len, max_len, max_examples, max_vocab_examples, max_vocab, num_epochs, num_print, d_emb, d_dec, lr]
    return args, ds, valds, model, device, hyperparams

class NTMPerplexityCallback:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    def __call__(self, history_no_eos, response, convo):
        """Allow NTM-based language model to predict logits for convo history."""
        history_lens = (history_no_eos != 0).long().sum(dim=1)
        convo_logits = self.model(labels=convo)
        logits = gather_logits(convo_logits, history_lens, max_len)
        logits = move_prob_from_s_to_eos(logits, self.vocab)
        return logits


if __name__ == '__main__':
    args, ds, valds, model, device, hyperparams = setup()
    history_len, max_len, max_examples, max_vocab_examples, max_vocab, num_epochs, num_print, d_emb, d_dec, lr = hyperparams
    if args.samples_file is not None:
        args.samples_file = open(str(args.samples_file), 'w')

    callback = lambda convo: model(labels=convo)
    train(model, callback, ds, args.epochs, lr, model_path=args.model_path,
          valds=valds, device=device)

    print_examples(valds, model, device, max_len, samples_file=args.samples_file)

    callback = NTMPerplexityCallback(model, valds.vocab)
    entropy, perplexity = eval_perplexity(callback, valds, device)
    print('Cross entropy: %s' % entropy.item())
    print('Perplexity: %s' % perplexity.item())


