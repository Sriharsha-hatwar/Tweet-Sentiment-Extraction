# Write the validation functions over here.. error functions.
import os
import numpy as np
import urllib
import tokenizers
import torch.nn as nn
from transformers import AutoTokenizer

def download_vocab_files_for_tokenizer(tokenizer, model_type, output_path):
    '''
    This is used to download some of the vocab files and merges file for tokenizers.
    '''
    vocab_files_map = tokenizer.pretrained_vocab_files_map
    vocab_files = {}
    for resource in vocab_files_map.keys():
        download_location = vocab_files_map[resource][model_type]
        f_path = os.path.join(output_path, os.path.basename(download_location))
        urllib.request.urlretrieve(download_location, f_path)
        vocab_files[resource] = f_path
    return vocab_files

def download_files():
    # Change the name and path.
    model_type = 'roberta-base'
    output_path = "..\input\/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    vocab_files = download_vocab_files_for_tokenizer(tokenizer, model_type, output_path)
    fast_tokenizer = tokenizers.ByteLevelBPETokenizer(vocab_files.get('vocab_file'), vocab_files.get('merges_file'))

def loss_fn(start_logits, end_logits, start_position, ending_position):
    cross_entropy_function = nn.CrossEntropyLoss()
    start_loss = cross_entropy_function(start_logits, start_position)
    end_loss = cross_entropy_function(end_logits, end_loss)
    return start_loss + end_loss

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

def calculate_running_jaccard(tweet_txt, original_selected_text, sentiment, offset, orig_tweet_start, orig_tweet_end):
    # So the main idea of this function is to return the jaccard index by using the above function : jaccard(str1, str2):

    if orig_tweet_end < orig_tweet_start : 
        orig_tweet_end = orig_tweet_start
    
    model_selected_text = tweet_txt[ offset[orig_tweet_start][0] : offset[orig_tweet_end][1]]

    # Here for some postprocessing.

    return jaccard(model_selected_text, original_selected_text)


def get_weight_decay_parameters(param_optimizer, no_weight_decay):
    non_weight_decay_params = []
    weight_decay_params = []
    for name, parameters in param_optimizer:
        found_entity = False
        for string in no_weight_decay:
            if string in name and not found_entity:
                non_weight_decay_params.append(parameters)
                found_entity = True
        if not found_entity:
            weight_decay_params.append(parameters)

    return [{'params' : weight_decay_params, 'weight_decay' : 0.001 } , {'params' : non_weight_decay_params, 'weight_decay' : 0 }]    
