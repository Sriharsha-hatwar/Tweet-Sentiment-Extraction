# Create different models here.
import transformers
import torch.nn as nn
import torch.nn.functional as functional
from config import BERTConfig, RoBERTaConfig

class TweetBERTModel(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super(TweetBERTModel, self).__init__(config)
        self.bert = transformers.BertModel.from_pretrained(BERTConfig.BERT_PATH, config=config)
        self.dropout = nn.Dropout(0.1)
        self.l0  = nn.Linear(768*2, 2)
    
    def forward(self, ids, mask, toke_type_ids):
        last_hidden_state, pooling_layer, all_hidden_states = self.bert(
            input_ids=ids,
            attention_mask=mask, # This is required to mask the padding tokens
            token_type_ids=token_type_ids # Ok so this contains the tokens of input sentence.
        )
        # Now concatinate the last two layers.
        # The dimension of the all_hidden_states -> (no_of_bert_layers + 1, batch_size, sequence_length , hidden_state_size)
        # so all_hidden_states[-1] and all_hidden_states[-2] has dimension =   (batch_size, sequence_length , hidden_state_size)
        embeddings = torch.cat(all_hidden_states[-1], all_hidden_states[-2], axis=-1)
        # This means that the dimensionality of hidden state is twice of len(hidden_state_size) as it contains
        out = self.dropout(embeddings)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


        
        
        