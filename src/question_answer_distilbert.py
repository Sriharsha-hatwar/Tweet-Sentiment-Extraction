'''
Creating the data in this format : 

train_data = [
    {
        'context': "This tweet sentiment extraction challenge is great",
        'qas': [
            {
                'id': "00001",
                'question': "positive",
                'answers': [
                    {
                        'text': "is great",
                        'answer_start': 43
                    }
                ]
            }
        ]
    }
    ]

Refer Docs : https://github.com/ThilinaRajapakse/simpletransformers#data-format
'''

import os
import json
import logging
import numpy as np
import pandas as pd
from simpletransformers.question_answering import QuestionAnsweringModel


BEST_MODEL_DIR = '../trained-models/simple-transformers'

def remove_nan_rows(df):
    '''
    Should be moved to utils.py
    ''' 
    pass



def return_answer_text(complete_text, selected_text):
    index = complete_text.find(selected_text)
    answer_dict = {'text' : selected_text , 'answer_start' : index}
    return answer_dict

def generate_training_data_in_json(train_np_arr):
    logging.info("Generating the json dump for training.")
    output = []
    for each_row in train_np_arr:
        ctxt = each_row[1]
        row_id = each_row[0]
        question = each_row[3]
        selected_answer = each_row[2]
        answer = [return_answer_text(ctxt, selected_answer)]
        question_and_answer_list = []
        question_and_answer = {'id' : row_id, 'is_impossible' : False, 'question' : question , 'answers' : answer}
        question_and_answer_list.append(question_and_answer)
        output.append({'context' : ctxt, 'qas' : question_and_answer_list})

    with open('../data/train.json', 'w') as output_file:
        json.dump(output, output_file)

def generate_testing_data_in_json(test_np_arr):
    logging.info("Logging.")
    logging.info("Generating the json dump for training.")
    output = []
    for each_row in test_np_arr:
        ctxt = each_row[1]
        row_id = each_row[0]
        question = each_row[2]
        answer = [{'text' : '' , 'answer_start' : -1}]
        question_and_answer = [{'id' : row_id, 'is_impossible' : False, 'question' : question , 'answers' : answer}]
        output.append({'context' : ctxt, 'qas' : question_and_answer})

    with open('../data/test.json', 'w') as output_file:
        json.dump(output, output_file)

def train():
    model = QuestionAnsweringModel('distilbert', 
                               'distilbert-base-uncased-distilled-squad',
                               args={'reprocess_input_data': True,
                                     'overwrite_output_dir': True,
                                     'learning_rate': 5e-5,
                                     'num_train_epochs': 3,
                                     'max_seq_length': 192,
                                     'doc_stride': 64,
                                     'fp16': False,
                                     'best_model_dir': os.path.join(BEST_MODEL_DIR, 'distilbert-base-uncased-distilled-squad')
                                    },
                              use_cuda=True)
    model.train_model('../data/train.json')

def test():
    logging.info("Testing")


def main(): 
    train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
    test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
    sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)


    train_array = np.array(train_df)
    test_array = np.array(test_df)


    #generate_training_data_in_json(train_array)
    #generate_testing_data_in_json(test_array)
    train()   

if __name__ == "__main__":
    main()
        





