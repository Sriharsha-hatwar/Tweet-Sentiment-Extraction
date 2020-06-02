import pandas as pd
import logging 
from sklearn.model_selection import KFold, StratifiedKFold

def create_folds(n_folds = 5):
    train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
    X = train_df[['textID', 'text', 'sentiment']]
    y = train_df.selected_text
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    train_df.loc[ : ,  'kfold'] = -1
    kf = KFold(n_splits = n_folds)
    logging.info("Verfication : ",kf.get_n_splits(X))
    for fold, (train_indices, val_indices) in enumerate(kf.split(X)):
        logging.info("The train index : ",train_indices)
        logging.info("Valid index : ", val_indices)
        train_df.loc[val_indices, 'kfold'] = fold
    
    train_df.to_csv('../input/tweet-sentiment-extraction/train_folds.csv')

def create_sentiment_wise_folds(n_folds = 5):
    train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
    sentiment_df_container = {}
    sentiments = train_df.sentiment.value_counts()
    for sentiment in sentiments.keys():
        X = train_df[train_df['sentiment'] == sentiment].reset_index()
        X.loc[:, 'kfold'] = -1
        kf = KFold(n_splits = n_folds)
        for fold, (train_indices, val_indices) in enumerate(kf.split(X)):
            X.loc[val_indices, 'kfold'] = fold
        sentiment_df_container[sentiment] = X
    sentiment_list = list(sentiment_df_container.keys())
    final_df = pd.concat([sentiment_df_container[sentiment_list[0]], sentiment_df_container[sentiment_list[1]], 
                sentiment_df_container[sentiment_list[2]]], axis=0)
    final_df = final_df.sample(frac=1).reset_index(drop=True)
    final_df.to_csv('../input/tweet-sentiment-extraction/train_sentiment_folds.csv')

def stratified_wise_folds(n_folds = 5):
    train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
    train_df = train_df.dropna().reset_index(drop=True)
    train_df.loc[:, 'kfold'] = -1

    train_df = train_df.sample(frac=1).reset_index(drop=True)

    kf = StratifiedKFold(n_splits=n_folds)

    for fold, (train_indices, val_indices) in enumerate(kf.split(X=train_df, y=train_df.sentiment.values)):
        print("Len : train_indices , val_indices ", len(train_indices), len(val_indices))
        train_df.loc[val_indices, 'kfold'] = fold

    train_df.to_csv('../input/train_stratified.csv')


    
if __name__ == "__main__":
    #create_folds(5)
    #create_sentiment_wise_folds(5)
    stratified_wise_folds(5)