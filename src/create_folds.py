import pandas as pd
from sklearn.model_selection import KFold

def create_folds(n_folds = 5):
    train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
    X = train_df[['textID', 'text', 'sentiment']]
    y = train_df.selected_text
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    train_df.loc[ : ,  'kfold'] = -1
    kf = KFold(n_splits=n_folds)
    print("Verfication : ",kf.get_n_splits(X))
    for fold, (train_indices, val_indices) in enumerate(kf.split(X)):
        print("The train index : ",train_indices)
        print("Valid index : ", val_indices)
        train_df.loc[val_indices, 'kfold'] = fold
    
    train_df.to_csv('../input/tweet-sentiment-extraction/train_folds.csv')

if __name__ == "__main__":
    create_folds(5)