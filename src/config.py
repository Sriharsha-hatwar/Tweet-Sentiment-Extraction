import os
import urllib
import tokenizers
from transformers import AutoTokenizer


class BERTConfig:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    EPOCHS = 5
    BERT_TYPE = 'bert-base-uncased'
    BERT_PATH='../input/bert-base-uncased/'
    MODEL_PATH='model.bin'
    TRAINING_FILE='../input/tweet-sentiment-extraction/train_folds.csv'
    PREPROCESS_TEXT = False
    TOKENIZER=tokenizers.BertWordPieceTokenizer(
        f"../input/bert-base-uncased/vocab.txt",
        lowercase=True
    )

class RoBERTaConfig:
    MAX_LEN = 96
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    EPOCHS = 7
    ROBERTA_PATH="../input/roberta-base/"
    MODEL_PATH='model.bin'
    TRAINING_FILE='../input/tweet-sentiment-extraction/train_stratified.csv'
    PREPROCESS_TEXT = False
    WARMUP_RATIO = 0.3
    TOKENIZER=tokenizers.ByteLevelBPETokenizer(
        vocab_file=f"{ROBERTA_PATH}/roberta-base-vocab.json",
        merges_file=f"{ROBERTA_PATH}/roberta-base-merges.txt",
        lowercase=True,
        add_prefix_space=True
    )
