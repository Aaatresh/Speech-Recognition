## Importing necessary libraries

import argparse
import logging
import os
import re

import torch
from torch.utils.data import DataLoader

from deepspeech.data.datasets import LibriSpeech
from deepspeech.data.loader import collate_input_sequences
from deepspeech.decoder import BeamCTCDecoder
from deepspeech.decoder import GreedyCTCDecoder
from deepspeech.global_state import GlobalState
from deepspeech.logging import LogLevelAction
from deepspeech.models import DeepSpeech
from deepspeech.models import DeepSpeech2
from deepspeech.models import Model

#__________________________________________________________________________________________________________________

# MODEL_CHOICES = ["ds1", "ds2"]

def main(args=None):
    """Train and evaluate a DeepSpeech network.

    Args:
        args: List of arguments to use. This is optional.
    """
    args = get_parser().parse_args(args)

    global_state = GlobalState(exp_dir=args.exp_dir,
                               log_frequency=args.slow_log_freq)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    decoder_cls, decoder_kwargs = get_decoder(args)

    model = get_model(args, decoder_cls, decoder_kwargs, global_state.exp_dir)

    train_loader = get_train_loader(args, model)

    val_loader = get_val_loader(args, model)

    if train_loader is not None:
        for epoch in range(model.completed_epochs, args.n_epochs):
            maybe_eval(model, val_loader, args.dev_log)
            model.train(train_loader)
            _save_model(args.model, model, args.exp_dir)

    maybe_eval(model, val_loader, args.dev_log)


def maybe_eval(model, val_loader, dev_log):
    """Method to evaluate the model on val_loader for each statistic in the dev_log.

    Arguments:
        model: A deepspeech.models.Model.
        val_loader: A data loader object of torch for the validation set. This is optional.
        dev_log: a string list of statistics used to evaluate the model. 
    """
    if val_loader is not None:
        for stat in set(dev_log):
            if stat == "loss":
                model.eval_loss(val_loader)
            elif stat == "wer":
                model.eval_wer(val_loader)
            else:
                raise ValueError("unknown evaluation statistic: %r" % stat)


def get_parser():
    """
        Returns argument parser data.
    """
    parser = argparse.ArgumentParser(
        description="train and evaluate a DeepSpeech model for comparison with MFCC and DTW, and HMM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    
    ## arguments for model parameters
    parser.add_argument("model",
                        default = "deepspeech")

    parser.add_argument("--state_dict_path",
                        default=None)

#     parser.add_argument("--no_resume_from_exp_dir",
#                         action="store_true",
#                         default=False,
#                         help="do not load the last state_dict in exp_dir")

    ## arguments for decoder parameters
    parser.add_argument("--decoder",
                        default="greedy")

    parser.add_argument("--lm_path",
                        default=None,
                        help="path to language model - if None, no lm is used")

    parser.add_argument("--lm_weight",
                        default=None,
                        type=float,
                        help="language model weight in loss (i.e. alpha)")

    parser.add_argument("--word_weight",
                        default=None,
                        type=float,
                        help="word bonus weight in loss (i.e. beta)")

    ## arguments for optimizer
    parser.add_argument("--learning_rate",
                        default=0.0003,
                        type=float,
                        help="learning rate of Adam optimizer")

    ## arguments for cache directory
    parser.add_argument("--cachedir",
                        default="/tmp/data/cache/",
                        help="location to download dataset(s)")

    ## training process arguments
    TRAIN_SUBSETS = ["train-clean-100",
                     "train-clean-360",
                     "train-other-500"]
                     
    parser.add_argument("--train_subsets",
                        default=TRAIN_SUBSETS,
                        choices=TRAIN_SUBSETS,
                        help="LibriSpeech subsets to train on",
                        nargs="*")

    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="number of samples in a training batch")

    parser.add_argument("--train_num_workers",
                        default=4,
                        type=int,
                        help="number of subprocesses for train DataLoader")

    # validation process arguments
    parser.add_argument("--dev_log",
                        default=["loss", "wer"],
                        choices=["loss", "wer"],
                        nargs="*",
                        help="validation statistics to log")

    parser.add_argument("--dev_subsets",
                        default=["dev-clean", "dev-other"],
                        choices=["dev-clean", "dev-other",
                                 "test-clean", "test-other"],
                        help="LibriSpeech subsets to evaluate loss and WER on",
                        nargs="*")

    parser.add_argument("--dev_batch_size",
                        default=16,
                        type=int,
                        help="number of samples in a validation batch")

    parser.add_argument("--dev_num_workers",
                        default=4,
                        type=int,
                        help="number of subprocesses for dev DataLoader")


    parser.add_argument("--n_epochs",
                        default=15,
                        type=int,
                        help="number of epochs for training")


    return parser



def get_model(args, decoder_cls, decoder_kwargs, exp_dir):
    """Returns a deepspeech.models.Model.

    Args:
        args: An argparse.Namespace for the argparse.ArgumentParser
            returned by get_parser.
        decoder_cls: See deepspeech.models.Model.
        decoder_kwargs: See deepspeech.models.Model.
        exp_dir: path to directory where all experimental data will be stored.
    """
    
    model_cls = {"ds": DeepSpeech}[args.model]

    model = model_cls(optimiser_cls=torch.optim.Adam,
                      optimiser_kwargs={"lr": args.learning_rate},
                      decoder_cls=decoder_cls,
                      decoder_kwargs=decoder_kwargs)

    state_dict_path = args.state_dict_path
    if state_dict_path is None and not args.no_resume_from_exp_dir:
    
        # Restore from last saved  state_dict  in  exp_dir .
        state_dict_path = _get_last_state_dict_path(args.model, exp_dir)

    if state_dict_path is not None:
        # Restore from user-specified  state_dict .
        logging.debug("restoring state_dict at %s" % state_dict_path)
        map_location = "cpu" if not torch.cuda.is_available() else None
        model.load_state_dict(torch.load(state_dict_path, map_location))
    else:
        logging.debug("using randomly initialised model")
        _save_model(args.model, model, exp_dir)

    return model


def get_decoder(args):
    """Returns a deepspeech.decoder.Decoder.

    Args:
        args: An argparse.Namespace for the argparse.ArgumentParser
            returned by get_parser.
    """
    decoder_kwargs = {"alphabet": Model.ALPHABET,
                      "blank_symbol": Model.BLANK_SYMBOL}

    
    decoder_cls = GreedyCTCDecoder

    beam_args = ["lm_weight", "word_weight", "beam_width", "lm_path"]
    for arg in beam_args:
        if getattr(args, arg) is not None:
            raise ValueError("greedy decoder used but %r is not "
                             "None" % arg)

    return decoder_cls, decoder_kwargs


def all_state_dicts(model_str, exp_dir):
    """Returns a dict of (epoch, filename) for all state_dicts in  exp_dir .

    Args:
        model_str: Model whose state_dicts to consider.
        exp_dir: path to directory where all experimental data will be stored.
    """
    state_dicts = {}

    for f in os.listdir(exp_dir):
    
        match = re.match("(%s-([0-9]+).pt)" % model_str, f)
        if not match:
            continue

        groups = match.groups()
        name = groups[0]
        epoch = groups[1]

        state_dicts[epoch] = name

    return state_dicts


def _get_last_state_dict_path(model_str, exp_dir):
    """Returns the absolute path of the last state_dict in  exp_dir  or  None .

    Args:
        model_str: Model whose state_dicts to consider.
        exp_dir: path to directory where all experimental data will be stored.
    """
    state_dicts = all_state_dicts(model_str, exp_dir)

    if len(state_dicts) == 0:
        return None

    last_epoch = sorted(state_dicts.keys())[-1]

    return os.path.join(exp_dir, state_dicts[last_epoch])


def _save_model(model_str, model, exp_dir):
    """Saves the model"s  state_dict  in  exp_dir .

    Args:
        model_str: Argument name of  model .
        model: A  deepspeech.models.Model .
        exp_dir: path to directory where the  model "s  state_dict  will be
            stored.
    """
    save_name = "%s-%d.pt" % (model_str, model.completed_epochs)
    save_path = os.path.join(exp_dir, save_name)
    torch.save(model.state_dict(), save_path)


def get_train_loader(args, model):
    """Returns a  torch.nn.DataLoader over the training data.

    Args:
        args: An  argparse.Namespace  for the  argparse.ArgumentParser 
            returned by  get_parser .
        model: A  deepspeech.models.Model .
    """
    if len(args.train_subsets) == 0:
        logging.debug("no  train_subsets  specified")
        return

    todo_epochs = args.n_epochs - model.completed_epochs
    if todo_epochs <= 0:
        logging.debug(" n_epochs  <=  model.completed_epochs ")
        return

    train_cache = os.path.join(args.cachedir, "train")
    train_dataset = LibriSpeech(root=train_cache,
                                subsets=args.train_subsets,
                                transform=model.transform,
                                target_transform=model.target_transform,
                                download=True)

    return DataLoader(train_dataset,
                      collate_fn=collate_input_sequences,
                      pin_memory=torch.cuda.is_available(),
                      num_workers=args.train_num_workers,
                      batch_size=args.train_batch_size,
                      shuffle=True)


def get_val_loader(args, model):
    """Returns a  torch.nn.DataLoader over the validation data.

    Args:
        args: An  argparse.Namespace  for the  argparse.ArgumentParser 
            returned by  get_parser .
        model: A  deepspeech.models.Model .
    """
    if len(args.dev_subsets) == 0:
        logging.debug("no  dev_subsets  specified")
        return

    if len(args.dev_log) == 0:
        logging.debug("no  dev_log  statistics specified")
        return

    dev_cache = os.path.join(args.cachedir, "dev")
    dev_dataset = LibriSpeech(root=dev_cache,
                              subsets=args.dev_subsets,
                              transform=model.transform,
                              target_transform=model.target_transform,
                              download=True)

    return DataLoader(dev_dataset,
                      collate_fn=collate_input_sequences,
                      pin_memory=torch.cuda.is_available(),
                      num_workers=args.dev_num_workers,
                      batch_size=args.dev_batch_size,
                      shuffle=False)
