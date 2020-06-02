import torch
import utils
import transformers
import numpy as np
import pandas as pd
from torch.utils import data
from config import RoBERTaConfig
from models import TweetRoBERTaModel
from tqdm.autonotebook import tqdm

class TweetDataSetForRoBERTa(data.Dataset):
    def __init__(self, tweet_texts, sentiments, preprocess_texts, textID):
        self.tweet_texts = tweet_texts
        self.sentiments = sentiments
        self.textID = textID
        self.tokenizer = RoBERTaConfig.TOKENIZER
        self.max_len = RoBERTaConfig.MAX_LEN
        self.preprocess_texts = preprocess_texts
    
    def __len__(self):
        return len(self.tweet_texts)

    def __getitem__(self, item):
        
        data = preprocess_roberta(
                self.tweet_texts[item],
                self.sentiments[item],
                self.tokenizer,
                self.max_len,
                self.textID
            )
        return {
            'ids' : torch.tensor(data['ids'], dtype=torch.long),
            'mask' : torch.tensor(data['mask'], dtype=torch.long),
            'token_type_ids' : torch.tensor(data['token_type_ids'], dtype=torch.long),
            'tweet_offsets' : torch.tensor(data['tweet_offsets'], dtype=torch.long),
            'sentiment' : data['sentiment'],
            'orig_tweet' : data['orig_tweet'],
            'textID' : data['textID']
        }

def preprocess_roberta(tweet_main_text, sentiment, tokenizer, max_len, textID):
    
    # Please test this function with some inputs.

    tweet_main_text = " " + " ".join(str(tweet_main_text).split())

    # Get the tokens out of roberta.
    tokenized_tweet = tokenizer.encode(tweet_main_text)
    token_ids = tokenized_tweet.ids
    token_offsets = tokenized_tweet.offsets

    sentiment_ids = {
        'positive' : 1313,
        'negative' : 2430,
        'neutral' : 7974
    }

    input_ids = [0] + [sentiment_ids[sentiment]] + [2] + [2] + token_ids + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(token_ids) + 1)
    mask = [1] * (len(token_type_ids))
    tweet_offsets = [(0, 0)] * 4 + token_offsets + [(0, 0)]


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
        'textID' : list(textID)
    }

def get_span(start_offset, end_offset, sentiment, tweet, offset):
    selected_text = ''
    if end_offset < start_offset:
        end_offset = start_offset
    if sentiment == 'neutral' or len(tweet) <= 2:
        selected_text = tweet
    else :
        selected_text = tweet[offset[start_offset][0] :  offset[end_offset][1]]
    return selected_text


def main():
    test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

    test_tweet_dataset = TweetDataSetForRoBERTa(
        test_df.text.values,
        test_df.sentiment.values,
        RoBERTaConfig.PREPROCESS_TEXT,
        test_df.textID.values
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_tweet_dataset,
        batch_size=RoBERTaConfig.VALID_BATCH_SIZE,
        num_workers=0,
        pin_memory=True
    )

    device = torch.device("cuda")
    model_config = transformers.RobertaConfig.from_pretrained(RoBERTaConfig.ROBERTA_PATH)
    model_config.output_hidden_states = True
    
    model_first = TweetRoBERTaModel(config=model_config)
    model_first.to(device)
    model_first.load_state_dict(torch.load(f'{RoBERTaConfig.ROBERTA_PATH}model_roberta_cnn_two_hidden_0.bin'))
    model_first.eval()

    model_second = TweetRoBERTaModel(config=model_config)
    model_second.to(device)
    model_second.load_state_dict(torch.load(f'{RoBERTaConfig.ROBERTA_PATH}model_roberta_cnn_two_hidden_1.bin'))
    model_second.eval()

    model_third = TweetRoBERTaModel(config=model_config)
    model_third.to(device)
    model_third.load_state_dict(torch.load(f'{RoBERTaConfig.ROBERTA_PATH}model_roberta_cnn_two_hidden_2.bin'))
    model_third.eval()

    model_fourth = TweetRoBERTaModel(config=model_config)
    model_fourth.to(device)
    model_fourth.load_state_dict(torch.load(f'{RoBERTaConfig.ROBERTA_PATH}model_roberta_cnn_two_hidden_3.bin'))
    model_fourth.eval()

    model_fifth = TweetRoBERTaModel(config=model_config)
    model_fifth.to(device)
    model_fifth.load_state_dict(torch.load(f'{RoBERTaConfig.ROBERTA_PATH}model_roberta_cnn_two_hidden_4.bin'))
    model_fifth.eval()

    

    tk0 = tqdm(test_data_loader, total=len(test_data_loader))
    output_id = []
    output_sentence = []
    with torch.no_grad():
        for batch_index, data in enumerate(tk0):
            input_ids = data["ids"]
            token_type_ids = data['token_type_ids']
            attention_mask = data['mask']
            tweet_offsets = data['tweet_offsets']
            sentiments = data['sentiment']
            orig_tweet = data['orig_tweet']
            textIDs = data['textID']
            #print(textIDs)

            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device , dtype=torch.long)

            start_logits_first, end_logits_first = model_first(input_ids, token_type_ids, attention_mask)
            start_logits_second, end_logits_second = model_second(input_ids, token_type_ids, attention_mask)
            start_logits_third, end_logits_third = model_third(input_ids, token_type_ids, attention_mask)
            start_logits_fourth, end_logits_fourth = model_fourth(input_ids, token_type_ids, attention_mask)
            start_logits_fifth, end_logits_fifth = model_fifth(input_ids, token_type_ids, attention_mask)

            start_logits = (start_logits_first + start_logits_second + start_logits_third + start_logits_fourth + start_logits_fifth) / 5.0
            end_logits = (end_logits_first + end_logits_second + end_logits_third + end_logits_fourth + end_logits_fifth) / 5.0

            start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            for index, tweet in enumerate(orig_tweet):
                sentiment = sentiments[index]
                offset = tweet_offsets[index]
                textID = textIDs[index]
                selected_tweet_start = np.argmax(start_logits[index, :])
                selected_tweet_end = np.argmax(end_logits[index, : ])
                model_selected_text_span = get_span(selected_tweet_start, selected_tweet_end, sentiment, tweet, offset)
                output_sentence.append(model_selected_text_span)

    sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
    sample.loc[:, 'selected_text'] = output_sentence
    sample.to_csv("../output/submission_10_roberta_cnn_2hidden.csv", index=False)

if __name__ == '__main__':
    main()
