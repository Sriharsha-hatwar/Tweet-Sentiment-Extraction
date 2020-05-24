import sys
import utils
import torch
import numpy as np
import transformers
import pandas as pd
import torch.nn as nn
from config import RoBERTaConfig
from tqdm.autonotebook import tqdm
from models import TweetRoBERTaModel
from dataset import TweetDataSetForRoBERTa

def train(data_loader, model, optimizer, device, scheduler):
    # so first of all....
    model.train() # To make sure that the gradients are being stored.
    # please put that average meter thing here for jaccard and loss?
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for batch_index, data in enumerate(tk0):
        input_ids = data["ids"]
        token_type_ids = data['token_type_ids']
        attention_mask = data['mask']
        tweet_offsets = data['tweet_offsets']
        sentiments = data['sentiment']
        target_start = data['target_start']
        target_end = data['target_end']
        orig_selected = data['orig_selected']
        orig_tweet = data['orig_tweet']

        # Push the relevant details to the device.
        #print("The device is ",device)
        input_ids = input_ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device , dtype=torch.long)
        target_start = target_start.to(device, dtype=torch.long)
        target_end = target_end.to(device, dtype=torch.long)
        model.zero_grad()
        start_logits, end_logits = model(input_ids, token_type_ids, attention_mask)
        loss = utils.loss_fn(start_logits, end_logits, target_start, target_end)
        

        loss.backward()
        optimizer.step()
        scheduler.step()

        # We can do one more thing, we can get the val score as we go on training right?


        start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
        end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
        # Before this we need to put the start and end logits into a softmax function.
        # Calculating the jaccard index fo this step.
        #print("The shape of start_logits and end_logits is :",start_logits.shape, end_logits.shape)
        #print("The shape of target start and target end", target_start.shape, target_end.shape)
        jaccard_indices = []
        #print("The sentiment is ",sentiment)
        for index, tweet in enumerate(orig_tweet):
            original_selected_text = orig_selected[index]
            #print("The index is : ",index)
            sentiment = sentiments[index]
            offset = tweet_offsets[index]
            selected_tweet_start = np.argmax(start_logits[index, :])
            selected_tweet_end = np.argmax(end_logits[index, : ])

            jaccard_index = utils.calculate_running_jaccard(
                tweet,
                original_selected_text,
                sentiment,
                offset,
                selected_tweet_start,
                selected_tweet_end
            )
            #print("The tweet main text : ",tweet)
            #print("The selected text is : ", original_selected_text)
            if selected_tweet_end < selected_tweet_start:
                selected_tweet_end = selected_tweet_start
            #print("The selected token start : ",selected_tweet_start)
            #print("The selected token end : ", selected_tweet_end)

            #print("The model selected text is ",tweet[offset[selected_tweet_start][0]:offset[selected_tweet_end][1]])
            #print("The jaccard score is ", jaccard_index)
            #sys.exit()

            jaccard_indices.append(jaccard_index)

        jaccards.update(np.mean(jaccard_indices), input_ids.size(0))
        losses.update(loss.item(), input_ids.size(0))

        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

def validation(data_loader, model, device):
    model.eval()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))
    with torch.no_grad():
        for batch_index, data in enumerate(tk0):
            input_ids = data["ids"]
            token_type_ids = data['token_type_ids']
            attention_mask = data['mask']
            tweet_offsets = data['tweet_offsets']
            sentiments = data['sentiment']
            target_start = data['target_start']
            target_end = data['target_end']
            orig_selected = data['orig_selected']
            orig_tweet = data['orig_tweet']

            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device , dtype=torch.long)
            target_start = target_start.to(device, dtype=torch.long)
            target_end = target_end.to(device, dtype=torch.long)

            start_logits, end_logits = model(input_ids, token_type_ids, attention_mask)
            loss = utils.loss_fn(start_logits, end_logits, target_start, target_end)

            start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            jaccard_indices = []
            
            for index, tweet in enumerate(orig_tweet):
                original_selected_text = orig_selected[index]
                sentiment = sentiments[index]
                offset = tweet_offsets[index]
                selected_tweet_start = np.argmax(start_logits[index, :])
                selected_tweet_end = np.argmax(end_logits[index, : ])

                jaccard_index = utils.calculate_running_jaccard(
                    tweet,
                    original_selected_text,
                    sentiment,
                    offset,
                    selected_tweet_start,
                    selected_tweet_end
                )
                jaccard_indices.append(jaccard_index)

            jaccards.update(np.mean(jaccard_indices), input_ids.size(0))
            losses.update(loss.item(), input_ids.size(0))

            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
    print("The jaccard score after the epoch is ", jaccards.avg)
    return jaccards.avg


def main(fold):
    # Get the training data, create a dataset and dataloader
    # Get the config and add the condition of getting all the outputs of the embedding.
    # get the parameters that needs to have no weight decay.
    # Intialize the early stopping criterion.
    # 
    # start with the epoch.. 

    main_df = pd.read_csv(RoBERTaConfig.TRAINING_FILE)
    train_df = main_df[main_df['kfold'] != fold].reset_index(drop=True)
    valid_df = main_df[main_df['kfold'] == fold].reset_index(drop=True)
    preprocess_texts = RoBERTaConfig.PREPROCESS_TEXT
    
    train_tweet_dataset = TweetDataSetForRoBERTa(
        tweet_texts = train_df.text.values,
        selected_texts = train_df.selected_text.values,
        sentiments = train_df.sentiment.values,
        preprocess_texts = preprocess_texts
    )

    validation_tweet_dataset = TweetDataSetForRoBERTa(
        tweet_texts = valid_df.text.values,
        selected_texts = valid_df.selected_text.values,
        sentiments = valid_df.sentiment.values,
        preprocess_texts = preprocess_texts
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_tweet_dataset,
        batch_size=RoBERTaConfig.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    valid_dataloader = torch.utils.data.DataLoader(
        validation_tweet_dataset,
        batch_size=RoBERTaConfig.VALID_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    device = torch.device("cuda")
    model_config = transformers.RobertaConfig.from_pretrained(RoBERTaConfig.ROBERTA_PATH)
    model_config.output_hidden_states = True
    model = TweetRoBERTaModel(config=model_config)
    model.to(device)

    num_training_steps = (len(train_df) / RoBERTaConfig.TRAIN_BATCH_SIZE) * RoBERTaConfig.EPOCHS

    no_weight_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    param_optimizer = model.named_parameters()
    optimizer_parameters = utils.get_weight_decay_parameters(param_optimizer, no_weight_decay)

    optimizer = transformers.AdamW(optimizer_parameters, lr = 3e-5)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0,
        num_training_steps=num_training_steps
    )
    early_stopping = utils.EarlyStopping(patience = 2, mode = 'max')

    print("Starting training..")

    for epoch in range(RoBERTaConfig.EPOCHS):
        train(train_dataloader, model, optimizer, device, scheduler)
        jaccard = validation(valid_dataloader, model, device)
        early_stopping(jaccard, model, f'{RoBERTaConfig.ROBERTA_PATH}model_conv_head_with_leaky_relu_{fold}.bin')
        if early_stopping.early_stop:
            print("No improvement in the validation score, stopping training.")
    

if __name__ == "__main__":
    main(2)