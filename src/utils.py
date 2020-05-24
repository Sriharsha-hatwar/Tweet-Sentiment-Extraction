# Write the validation functions over here.. error functions.
import os
import numpy as np
import urllib
import math
import torch
import tokenizers
import torch.nn as nn
from transformers import AutoTokenizer


class Conv1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.cut_last_element = (kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1)
        self.padding = math.ceil((1 - stride + dilation * (kernel_size-1))/2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, stride=stride, dilation=dilation)

    def forward(self, x):
        if self.cut_last_element:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)
            3

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
    end_loss = cross_entropy_function(end_logits, ending_position)
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

def preprocess_bert(tweet_main_text, selected_text, sentiment, tokenizer, max_len):
    # Finding the starting point and ending point of the selected_text
    # index = tweet_text.find(selected_text)
    #print(index)
    # Please test this function with some inputs.
    starting_index = -1
    ending_index = -1
    same_first_character = [ind for ind, char in enumerate(tweet_main_text) if char == selected_text[0]]
    for index in same_first_character:
        if(tweet_main_text[index : index + len(selected_text)] == selected_text):
            starting_index = index
            ending_index = index + len(selected_text)
    char_target = [0] * len(tweet_main_text)
    for i in range(starting_index, ending_index):
        char_target[i] = 1

    tokenized_tweet = tokenizer.encode(tweet_main_text)
    token_ids = tokenized_tweet.ids[1:-1] # The reason we have to do 1:-1 is by default they would have added.
    offsets = tokenized_tweet.offsets[1:-1]
    
    target_token = []

    for index, (start, end) in enumerate(offsets):
        if sum(char_target[start : end]) > 0:
            target_token.append(index)

    target_token_start = target_token[0]
    target_token_end = target_token[-1]

    sentiment_ids = {
        'positive' : 3893,
        'negative' : 4997,
        'neutral' : 8699,
    }

    input_ids = [101] + [sentiment_ids[sentiment]] + [102] + token_ids + [102]
    token_type_ids = [0, 0, 0] + [1] * (len(token_ids) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 3 + offsets + [(0, 0)]
    target_token_start =+ 3
    target_token_end += 3

    # Add pad if necessary
    remaining_len = max_len - len(input_ids)
        
    if (remaining_len > 0):
        input_ids = input_ids + [0] * remaining_len # Pad with zeros
        token_type_ids = token_type_ids + [0] * remaining_len
        mask = mask + [0] * remaining_len
        tweet_offsets = tweet_offsets + [(0,0) * remaining_len]
        
    return {
        'ids' : input_ids,
        'mask' : mask,
        'token_type_ids' : token_type_ids,
        'tweet_offsets' : tweet_offsets,
        'sentiment' : sentiment,
        'orig_tweet' : tweet_main_text,
        'orig_selected' : selected_text,
        'target_start' : target_token_start,
        'target_end' : target_token_end    
    }

def preprocess_roberta(tweet_main_text, selected_text, sentiment, tokenizer, max_len):
    
    # Please test this function with some inputs.

    tweet_main_text = " " + " ".join(str(tweet_main_text).split())
    selected_text = " "+ " ".join(str(selected_text).split())

    starting_index = -1
    ending_index = -1
    starting_indices = [ind for ind, char in enumerate(tweet_main_text) if char == selected_text[1]]
    for index in starting_indices:
        if (" "+tweet_main_text[index : index + len(selected_text)-1] == selected_text):
            starting_index = index
            ending_index = index + len(selected_text) - 1
            break
    
    char_target = [0] * len(tweet_main_text)
    for i in range(starting_index, ending_index):
        char_target[i] = 1
    
    # Get the tokens out of roberta.
    tokenized_tweet = tokenizer.encode(tweet_main_text)
    token_ids = tokenized_tweet.ids
    token_offsets = tokenized_tweet.offsets

    target_token = []

    for index, (start, end) in enumerate(token_offsets):
        if sum(char_target[start : end]) > 0:
            target_token.append(index)

    target_token_start = target_token[0]
    target_token_end = target_token[-1]

    sentiment_ids = {
        'positive' : 1313,
        'negative' : 2430,
        'neutral' : 7974
    }

    input_ids = [0] + [sentiment_ids[sentiment]] + [2] + [2] + token_ids + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(token_ids) + 1)
    mask = [1] * (len(token_type_ids))
    tweet_offsets = [(0, 0)] * 4 + token_offsets + [(0, 0)]
    target_token_start += 4
    target_token_end += 4


    padding_len = max_len - (len(input_ids))

    if padding_len > 0:
        input_ids = input_ids + padding_len * [1]
        token_type_ids = token_type_ids + padding_len * [0]
        mask = mask + padding_len * [0]
        tweet_offsets =  tweet_offsets + [(0, 0)] * padding_len
    
    return {
        'ids' : input_ids,
        'mask' : mask,
        'token_type_ids' : token_type_ids,
        'tweet_offsets' : tweet_offsets,
        'sentiment' : sentiment,
        'orig_tweet' : tweet_main_text,
        'orig_selected' : selected_text,
        'target_start' : target_token_start,
        'target_end' : target_token_end    
    }
    
if __name__ == "__main__":
    input_a = torch.randn(16, 192, 768)
    input_a_t = input_a.transpose(1,2)
    m = Conv1dSame(768, 128, 2)
    out = m(input_a_t)
    print(out.shape)
