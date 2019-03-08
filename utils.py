"""Code taken and modified from Github repository https://github.com/kuc2477/pytorch-ntm by Ha Junsoo under MIT license."""
from functools import reduce
import operator
import os
import os.path
import shutil
import torch
from torch.autograd import Variable
from torch.nn import init

from nlputils import convert_npy_to_str


def load_train_args(parser):
    parser.add_argument('--source', default=None, help='Directory to look for data files')
    parser.add_argument('--model_path', default=None, help='File path where model checkpoint is saved')
    parser.add_argument('--temp', help='Where to save generated dataset and vocab files')
    parser.add_argument('--tempval', default=None, help='Where to save generated validation dataset and vocab files')
    parser.add_argument('--regen', default=False, action='store_true', help='Renerate vocabulary')
    parser.add_argument('--regenval', default=False, action='store_true', help='Regenerate validation dataset')
    parser.add_argument('--device', default='cuda:0', help='Cuda device (or cpu) for tensor operations')
    parser.add_argument('--epochs', default=1, action='store', type=int, help='Number of epochs to run model for')
    parser.add_argument('--restore', default=False, action='store_true', help='Set to restore model from save')
    parser.add_argument('--num_print', default='1', help='Number of batches to print')
    parser.add_argument('--run', default=None, help='Path to save run values for viewing with Tensorboard')
    parser.add_argument('--tgdisable', default=False, action='store_true', help='If true, suppress Telegram alerts')
    parser.add_argument('--max_examples', default=None, help='Number of examples to use when training on dataset')
    # parser.add_argument('--dataset', default='ubuntu', help='Choose either opensubtitles or ubuntu dataset to train')
    parser.add_argument('--val', default=None, help='Validation set to use for model evaluation')
    parser.add_argument('--samples_file', default=None, help='File to write sampled outputs')


def save_checkpoint(model, model_dir, iteration, precision, best=True):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({
        'state': model.state_dict(),
        'iteration': iteration,
        'precision': precision,
    }, path)

    # override the best model if it's the best.
    if best:
        shutil.copy(path, path_best)
        print('=> updated the best model of {name} at {path}'.format(
            name=model.name, path=path_best
        ))

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))


def load_checkpoint(model, model_dir, best=True):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))

    # load the checkpoint.
    checkpoint = torch.load(path_best if best else path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path_best if best else path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    iteration = checkpoint['iteration']
    precision = checkpoint['precision']
    return iteration, precision


def validate(model, task,
             test_size=256, batch_size=32,
             cuda=False, verbose=True):
    data_loader = task.data_loader(batch_size)
    total_tested = 0
    total_bits = 0
    wrong_bits = 0

    for x, y in data_loader:
        # break on test size.
        if total_tested >= test_size:
            break

        # prepare the batch and reset the model's state variables.
        model.reset(x.size(0), cuda=cuda)
        x = Variable(x).cuda() if cuda else Variable(x)
        y = Variable(y).cuda() if cuda else Variable(y)

        # run the model through the sequences.
        for index in range(x.size(1)):
            model(x[:, index, :])

        # run the model to output sequences.
        predictions = []
        for index in range(y.size(1)):
            activation = task.model_output_activation(model())
            predictions.append(activation.round())
        predictions = torch.stack(predictions, 1).long()

        # calculate the wrong bits per sequence.
        total_tested += x.size(0)
        total_bits += reduce(operator.mul, y.size())
        wrong_bits += torch.abs(predictions-y.long()).sum().data[0]

    precision = 1 - wrong_bits/total_bits
    if verbose:
        print('=> precision: {prec:.4}'.format(prec=precision))
    return precision


def xavier_initialize(model, uniform=False):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'linear' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters() if
        p.dim() >= 2
    ]

    for p in parameters:
        init.xavier_normal(p) if uniform else init.xavier_normal(p)


def gaussian_intiailize(model, std=.1):
    for p in model.parameters():
        init.normal(p, std=std)


def write_example(file, history, label, prediction):
    """
    Write a single conversation to a file.
    :param file: file to write to, in write mode
    :param history: string, conversation history leading up to response
    :param prediction: string, response prediction of model
    :param label: string, true response from dataset
    :return: None
    """
    file.write('Input sentence: %s\n' % history)
    file.write('Ground truth: %s\n' % label)
    file.write('Generated response: %s\n' % prediction)
    file.write('\n')


def print_numpy_examples(vocab, eos, history, response, response_preds, convo_preds=None, samples_file=None):
    for j in range(history.shape[0]):
        np_pred = response_preds[j].cpu().numpy()
        np_context = history[j].cpu().numpy()
        np_target = response[j].cpu().numpy()
        context = convert_npy_to_str(np_context, vocab, eos)

        pred = convert_npy_to_str(np_pred, vocab, eos)
        target = convert_npy_to_str(np_target, vocab, eos)

        if convo_preds is not None:
            # optionally print out the whole conversation if it is available
            np_complete_convo = convo_preds[j].cpu().numpy()
            complete_convo = convert_npy_to_str(np_complete_convo, vocab, eos)
            print('Completed convo: %s' % complete_convo)

        print('History: %s' % context)
        print('Prediction: %s' % pred)
        print('Target: %s' % target)
        print()
        if samples_file is not None:
            write_example(samples_file, context, target, pred)