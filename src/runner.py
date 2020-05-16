import utils
import torch
import transformers
import pandas as pd
import torch.nn as nn
from config import BERTConfig
from tqdm.autonotebook import tqdm
from models import TweetBERTModel
from dataset import TweetDataSetForBert


def train(data_loader, model, optimizer, device, scheduler):
    # so first of all....
    model.train() # To make sure that the gradients are being stored.
    # please put that average meter thing here for jaccard and loss?
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()

    for batch_index, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        print("..................",batch_index)
        print("..................",type(data))
        input_ids = data["ids"]
        token_type_ids = data['token_type_ids']
        attention_mask = data['mask']
        tweet_offsets = data['tweet_offsets']
        sentiment = data['sentiment']
        target_start = data['target_start']
        target_end = data['target_end']
        orig_selected = data['orig_selected']
        orig_tweet = data['orig_tweet']

        # Push the relevant details to the device.
        input_ids = input_ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device , dtype=torch.long)

        model.zero_grad()
        start_logits, end_logits = model(input_ids, token_type_ids, attention_mask)
        loss = utils.loss_fn(start_logits, end_logits, target_start, target_end)
        

        loss.backward()
        optimizer.step()
        scheduler.step()

        # We can do one more thing, we can get the val score as we go on training right?


        start_logits = nn.Softmax(start_logits, dim=1).cpu().detach().numpy()
        end_logits = nn.Softmax(end_logits, dim=1).cpu().detach().numpy()
        # Before this we need to put the start and end logits into a softmax function.
        # Calculating the jaccard index fo this step.
        jaccard_indices = []
        for index, tweet in enumerate(orig_tweet):
            original_selected_text = orig_selected[index]
            sentiment = sentiment[index]
            offset = tweet_offsets[index]
            orig_tweet_start = target_start[index]
            orig_tweet_end = targte_end[index]

            jaccard_index = utils.calculate_running_jaccard(
                tweet,
                original_selected_text,
                sentiment,
                offset,
                orig_tweet_start,
                orig_tweet_end
            )
            jaccard_indices.append(jaccard_index)

        jaccards.update(np.mean(jaccard_indices), input_ids.size(0))
        losses.update(loss.item(), input_ids.size(0))

        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)


def validation(data_loader, model, device):
    model.eval()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()

    with torch.no_grad():
        for batch_index, data in tqdm(enumerate(data_loader), length=len(data_loader)):
            input_ids = data["ids"]
            token_type_ids = data['token_type_ids']
            attention_mask = data['mask']
            tweet_offsets = data['tweet_offsets']
            sentiment = data['sentiment']
            target_start = data['target_start']
            target_end = data['target_end']
            orig_selected = data['orig_selected']
            orig_tweet = data['orig_tweet']

            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device , dtype=torch.long)

            start_logits, end_logits = model(input_ids, token_type_ids, attention_mask)
            loss = utils.loss_fn(start_logits, end_logits, target_start, target_end)

            start_logits = nn.Softmax(start_logits, dim=1).cpu().detach().numpy()
            end_logits = nn.Softmax(end_logits, dim=1).cpu().detach().numpy()

            jaccard_indices = []
            
            for index, tweet in enumerate(orig_tweet):
                original_selected_text = orig_selected[index]
                sentiment = sentiment[index]
                offset = tweet_offsets[index]
                orig_tweet_start = target_start[index]
                orig_tweet_end = targte_end[index]

                jaccard_index = utils.calculate_running_jaccard(
                    tweet,
                    original_selected_text,
                    sentiment,
                    offset,
                    orig_tweet_start,
                    orig_tweet_end
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

    main_df = pd.read_csv(BERTConfig.TRAINING_FILE)
    train_df = main_df[main_df['kfold'] != fold].reset_index(drop=True)
    valid_df = main_df[main_df['kfold'] == fold].reset_index(drop=True)
    preprocess_texts = BERTConfig.PREPROCESS_TEXT
    
    train_tweet_dataset = TweetDataSetForBert(
        tweet_texts = train_df.text.values,
        selected_texts = train_df.selected_text.values,
        sentiments = train_df.sentiment.values,
        preprocess_texts = preprocess_texts
    )

    validation_tweet_dataset = TweetDataSetForBert(
        tweet_texts = valid_df.text.values,
        selected_texts = valid_df.selected_text.values,
        sentiments = valid_df.sentiment.values,
        preprocess_texts = preprocess_texts
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_tweet_dataset,
        batch_size=BERTConfig.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    valid_dataloader = torch.utils.data.DataLoader(
        validation_tweet_dataset,
        batch_size=BERTConfig.VALID_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    device = torch.device("cuda")
    model_config = transformers.BertConfig.from_pretrained(BERTConfig.BERT_PATH)
    model_config.output_hidden_states = True
    model = TweetBERTModel(config=model_config)
    model.to(device)

    num_training_steps = (len(train_df) / BERTConfig.TRAIN_BATCH_SIZE) * BERTConfig.EPOCHS

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

    for epoch in range(BERTConfig.EPOCHS):
        train(train_dataloader, model, optimizer, device, scheduler)
        jaccard = validation(valid_dataloader, model, device)
        early_stopping(jaccard, model, '../models/model_{fold}.bin')
        if early_stopping.early_stop:
            print("No improvement in the validation score, stopping training.")
    

if __name__ == "__main__":
    print("Starting the MAHUT!")
    main(2)
    

    


