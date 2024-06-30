import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

class BertNER(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        self.args = args
        self.bert = AutoModel.from_pretrained(args.name_or_path, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(args.name_or_path)
        self.ner_layer =args.ner_layer

        self.ner_head = nn.Sequential(
            nn.Linear(in_features=self.bert.config.hidden_size, out_features=args.ner_hidden_size),
            nn.ReLU(), nn.Dropout(p=args.dropout_rate),
            nn.Linear(in_features=args.ner_hidden_size, out_features=args.num_ner_tags)
        )

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        if self.ner_layer == 'last':
            ner_layer_hidden_states = bert_outputs['last_hidden_state']
        else:
            ner_layer_hidden_states = bert_outputs['hidden_states'][self.args.ner_layer]
        ner_logits = self.ner_head(ner_layer_hidden_states)
        return ner_logits
    
    def get_ner_ids(self, sentences):
        '''
        Get ner label and word ids for each token in input sentences
        '''
        if not isinstance(sentences, list):
            sentences = [sentences]
        tokenized_sentences = self.tokenizer(sentences, return_tensors='pt', padding=True)
        word_ids = [tokenized_sentences.word_ids(batch_index=i) for i in range(len(sentences))]
        ner_ids = self.forward(
            input_ids=tokenized_sentences['input_ids'],
            attention_mask=tokenized_sentences['attention_mask']
        ).max(dim=2)[1]
        return ner_ids, word_ids