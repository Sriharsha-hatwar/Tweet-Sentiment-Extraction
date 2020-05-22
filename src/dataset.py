import os 
import utils
import torch
import logging
from config import BERTConfig, RoBERTaConfig
from torch.utils import data


class TweetDataSetForBert(data.Dataset):
    def __init__(self, tweet_texts, selected_texts, sentiments, preprocess_texts):
        self.tweet_texts = tweet_texts
        self.selected_texts = selected_texts
        self.sentiments = sentiments
        self.tokenizer = BERTConfig.TOKENIZER
        self.max_len = BERTConfig.MAX_LEN
        self.preprocess_texts = preprocess_texts

    def __len__(self):
        return len(self.tweet_texts)

    def __getitem__(self, item):
        
        data = utils.preprocess_bert(
                self.tweet_texts[item],
                self.selected_texts[item],
                self.sentiments[item],
                self.tokenizer,
                self.max_len
            )
        # Write the preprocessing step here.
        return {
            'ids' : torch.tensor(data['ids'], dtype=torch.long),
            'mask' : torch.tensor(data['mask'], dtype=torch.long),
            'token_type_ids' : torch.tensor(data['token_type_ids'], dtype=torch.long),
            'tweet_offsets' : torch.tensor(data['tweet_offsets'], dtype=torch.long),
            'target_start' : torch.tensor(data['target_start'], dtype=torch.long),
            'target_end' : torch.tensor(data['target_end'], dtype=torch.long),
            'sentiment' : data['sentiment'],
            'orig_tweet' : data['orig_tweet'],
            'orig_selected' : data['orig_selected']
        }

    
class TweetDataSetForRoBERTa(data.Dataset):
    def __init__(self, tweet_texts, selected_texts, sentiments, preprocess_texts):
        self.tweet_texts = tweet_texts
        self.selected_texts = selected_texts
        self.sentiments = sentiments
        self.tokenizer = RoBERTaConfig.TOKENIZER
        self.max_len = RoBERTaConfig.MAX_LEN
        self.preprocess_texts = preprocess_texts
    
    def __len__(self):
        return len(self.tweet_texts)

    def __getitem__(self, item):
        
        data = utils.preprocess_roberta(
                self.tweet_texts[item],
                self.selected_texts[item],
                self.sentiments[item],
                self.tokenizer,
                self.max_len
            )
        return {
            'ids' : torch.tensor(data['ids'], dtype=torch.long),
            'mask' : torch.tensor(data['mask'], dtype=torch.long),
            'token_type_ids' : torch.tensor(data['token_type_ids'], dtype=torch.long),
            'tweet_offsets' : torch.tensor(data['tweet_offsets'], dtype=torch.long),
            'target_start' : torch.tensor(data['target_start'], dtype=torch.long),
            'target_end' : torch.tensor(data['target_end'], dtype=torch.long),
            'sentiment' : data['sentiment'],
            'orig_tweet' : data['orig_tweet'],
            'orig_selected' : data['orig_selected']
        }

