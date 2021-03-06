# Create different models here.
import sys
import utils
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as functional
from config import BERTConfig, RoBERTaConfig

class TweetBERTModel(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super(TweetBERTModel, self).__init__(config)
        self.bert = transformers.BertModel.from_pretrained(BERTConfig.BERT_PATH, config=config)
        self.dropout = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU()
        self.conv1 = utils.Conv1dSame(768, 128, 2)
        self.conv2 = utils.Conv1dSame(128, 64, 2)
        self.l0 = nn.Linear(64, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, toke_type_ids):
        last_hidden_state, pooling_layer, all_hidden_states = self.bert(
            input_ids=ids,
            attention_mask=mask, # This is required to mask the padding tokens
            token_type_ids=token_type_ids # Ok so this contains the tokens of input sentence.
        )
        # Now concatinate the last two layers.
        # The dimension of the all_hidden_states -> (no_of_bert_layers + 1, batch_size, sequence_length , hidden_state_size)
        # so all_hidden_states[-1] and all_hidden_states[-2] has dimension =   (batch_size, sequence_length , hidden_state_size)
        all_embeddings = torch.cat(all_hidden_states[-1], all_hidden_states[-2], axis=-1)
        # This means that the dimensionality of hidden state is twice of len(hidden_state_size) as it contains
        out = self.dropout(all_embeddings)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

    
class TweetRoBERTaModel(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super(TweetRoBERTaModel, self).__init__(config) 
        self.roberta = transformers.RobertaModel.from_pretrained(RoBERTaConfig.ROBERTA_PATH, config=config)
        self.dropout = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU()
        self.conv1 = utils.Conv1dSame(768*2, 128, 2)
        self.conv2 = utils.Conv1dSame(128, 64, 2)
        self.l0 = nn.Linear(64, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, input_ids, token_type_ids, mask):
        last_hidden_state, pooling_layer, all_hidden_states = self.roberta(
            input_ids=input_ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        all_embeddings = torch.cat((all_hidden_states[-1], all_hidden_states[-2]), axis=-1)
        # Need to find out an effective strategy to combine two hidden states.
        #all_embeddings = all_hidden_states[-1]
        # So bascially the output will be ->  (Batch_size, sequence_length, 768) 
        #                                                     +
        #                                         (Batch_size, sequence_length, 768)
        #                                        (Batch_size, sequence_length, 768*2)
        output = self.dropout(all_embeddings)
        output = output.transpose(1,2)
        conv_one_out = self.leakyrelu(self.conv1(output))
        conv_two_out = self.conv2(conv_one_out)
        conv_two_out = conv_two_out.transpose(1,2)
        logits = self.l0(conv_two_out) # This results in (Batch_size, sequence_length, 2) # Logits
        #print("The size of logits : ",logits.shape)
        #sys.exit()

        start_logits, end_logits = logits.split(1, dim=-1) # So now the dim of each wil be (batch_size, sequence_lenght, 1)
        #print("The shape of start_logits : ",start_logits.shape)
        #print("The shape of end_logits : ",end_logits.shape)
        start_logits = start_logits.squeeze(dim=-1)
        end_logits = end_logits.squeeze(dim=-1)

        return start_logits, end_logits
        
        


        
        
        