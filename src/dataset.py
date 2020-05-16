import os 
import torch
import logging
from config import BERTConfig, RoBERTaConfig
from torch.utils import data


class TweetDataSetForBert:
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
        
        data = self.preprocess_bert(
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
            'offsets' : torch.tensor(data['offsets'], dtype=torch.long),
            'target_start' : torch.tensor(data['target_start'], dtype=torch.long),
            'target_end' : torch.tensor(data['target_end'], dtype=torch.long),
            'sentiment' : data['sentiment'],
            'orig_tweet' : data['orig_tweet'],
            'orig_selected' : data['orig_selected']
        }

    def preprocess_bert(self, tweet_main_text, selected_text, sentiment, tokenizer, max_len):
        # Finding the starting point and ending point of the selected_text
        # index = tweet_text.find(selected_text)
        #print(index)
        # They are not using the sentiment here.. why?
        starting_index = None
        ending_index = None
        same_first_character = [ind for ind, char in enumerate(tweet_main_text) if char == selected_text[1]]
        for index in same_first_character:
            if(tweet_main_text[index : index + len(selected_text)] == selected_text):
                starting_index = index
                ending_index = index + len(selected_text) - 1
        char_target = [0] * len(tweet_main_text)
        char_target[starting_index : ending_index] = 1

        tokenized_tweet = tokenizer.encode(tweet_main_text)
        token_ids = tokenized_tweet.ids[1:-1] # The reason we have to do 1:-1 is by default they would have added.
        offsets = tokenized_tweet.offsets[1:-1]

        target_token = []

        for i in offsets:
            if sum(char_target[i[0] : i[1]]) > 0:
                target_token.append(i)

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


#if __name__ == '__main__':
