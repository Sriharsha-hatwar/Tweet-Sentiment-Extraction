import tokenizers
import urllib
import os
from transformers import AutoTokenizer


class BertConfig:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    EPOCHS = 10
    BERT_PATH='../input/bert-base-uncased/'
    MODEL_PATH='model.bin'
    TRAINING_FILE='../input/tweet-sentiment-extraction/train_folds.csv'
    TOKENIZER=tokenizers.BertWordPieceTokenizer(
        f"{BERT_PATH}/bert-base-uncased-vocab.txt",
        lowercase=True
    )

class RoBERTaConfig:
    MAX_LEN = 96
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    EPOCHS = 5
    ROBERTA_PATH="../input/roberta-base/"
    MODEL_PATH='model.bin'
    TRAINING_FILE='../input/tweet-sentiment-extraction/train_folds.csv'
    TOKENIZER=tokenizers.ByteLevelBPETokenizer(
        vocab_file=f"{ROBERTA_PATH}/roberta-base-vocab.json",
        merges_file=f"{ROBERTA_PATH}/roberta-base-merges.txt",
        lowercase=True,
        add_prefix_space=True
    )
