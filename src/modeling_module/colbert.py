import string

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import AutoModel

from modeling_module import *
from modeling_module.utils import *
from modeling_module.tokenizer import QueryTokenizer, DocTokenizer
from args import Args


class ColBERT(nn.Module, PyTorchModelHubMixin):

    def __init__(self, args: Args, config):
        super(ColBERT, self).__init__()

        self.args = args

        self.bert = AutoModel.from_pretrained(args.name_or_path, config=config)
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features=self.bert.config.hidden_size, out_features=args.embedding_hidden_size),
            nn.ReLU(), nn.Dropout(p=args.dropout_rate),
            nn.Linear(in_features=args.embedding_hidden_size, out_features=args.embedding_dim)
        )
        
        self.doc_tokenizer = DocTokenizer(self.args)
        self.query_tokenizer = QueryTokenizer(self.args)

        doc_mask_symbol_list = [self.doc_tokenizer.raw_tokenizer.pad_token_id]
        if args.mask_punctuation:
            doc_mask_symbol_list += self.doc_tokenizer.raw_tokenizer.convert_tokens_to_ids(list(string.punctuation))
        self.register_buffer('doc_mask_buffer', torch.tensor(doc_mask_symbol_list))

    def forward(self, Q, D):
        """
        Q: {query_input_ids, query_attention_mask} each with dim (B, L)
        D: {doc_input_ids, doc_attention_mask}
        """
        query_embedding, query_mask = self.query(**Q)
        doc_embedding, punctuation_mask = self.doc(**D)

        return self.score(
            query_embedding=query_embedding,
            query_mask=query_mask,
            doc_embedding=doc_embedding,
            doc_mask=punctuation_mask
        )

    def query(self, input_ids, attention_mask) -> tuple[torch.Tensor, torch.Tensor]:
        query_embedding = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        query_embedding = self.embedding_head(query_embedding)
        if self.args.query_pad_token is not None:
            query_embedding = F.normalize(query_embedding, p=2, dim=2)
            return query_embedding, None
        else:
            query_mask = attention_mask.unsqueeze(2)
            query_embedding = query_embedding * query_mask.float()
            query_embedding = F.normalize(query_embedding, p=2, dim=2)
            return query_embedding, query_mask.bool()
            
    def doc(self, input_ids, attention_mask, return_list=False):
        doc_embedding = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        doc_embedding = self.embedding_head(doc_embedding)
        # Mask for pad tokens and punctuation
        punctuation_mask = self.mask(input_ids, self.doc_mask_buffer).unsqueeze(2)
        doc_embedding = doc_embedding * punctuation_mask.float()
        doc_embedding = F.normalize(doc_embedding, p=2, dim=2)

        if not return_list:
            return doc_embedding, punctuation_mask.bool()
        else:
            doc_embedding = doc_embedding.cpu().to(dtype=torch.float16)
            punctuation_mask = punctuation_mask.cpu().bool().squeeze(-1)
            doc_embedding = [d[punctuation_mask[idx]] for idx, d in enumerate(doc_embedding)]
            return doc_embedding

    def mask(self, input_ids, mask_buffer):
        mask = ~(input_ids.unsqueeze(-1) == mask_buffer).any(dim=-1)
        return mask
    
    def score_pair(self, query_embedding, doc_embedding, doc_mask, query_mask=None) -> torch.Tensor:
        scores_no_masking = query_embedding @ doc_embedding.permute(0, 2, 1) # B * LQ * LD

        mask = doc_mask.permute(0, 2, 1).repeat([1, query_embedding.size(1), 1])
        masked_scores = scores_no_masking * mask.float()
        if query_mask is None:
            scores = masked_scores.max(dim=-1).values.mean(-1)   # B
        else:
            scores = masked_scores.max(-1).values.sum(-1) / query_mask.squeeze().sum(-1)
        return scores
    
    def score_cart(self, query_embedding, doc_embedding, doc_mask, query_mask=None):
        scores_no_masking = torch.einsum('qih,djh->qidj', query_embedding, doc_embedding) # Q * LQ * D * LD
        mask = doc_mask.squeeze().unsqueeze(0).unsqueeze(0).repeat([query_embedding.size(0), query_embedding.size(1), 1, 1])
        masked_scores = scores_no_masking * mask.float()
        if query_mask is None:
            scores = masked_scores.max(-1).values.mean(1)   # 
        else:
            scores = masked_scores.max(-1).values.sum(1) / query_mask.sum(1)
        return scores

    def evaluate_score(self, Q, D):
        query_embedding, query_mask = Q
        doc_embedding, punctuation_mask = D
        return self.score_cart(
            query_embedding=query_embedding,
            doc_embedding=doc_embedding,
            query_mask=query_mask,
            doc_mask=punctuation_mask
        )
    
    def encode_query(self, queries):
        '''
        Tokenize and embedd queries
        '''
        if not isinstance(queries, list):
            queries = [queries]
        tokenized_queries = self.query_tokenizer(
            queries,
            padding=True,
            return_tensors='pt'
        ).to(get_model_device(self))
        return self.query(tokenized_queries['input_ids'], tokenized_queries['attention_mask'])
    
    def encode_doc(self, docs):
        '''
        Tokenize and embedd docs
        '''
        if not isinstance(docs, list):
            docs = [docs]
        tokenized_docs = self.doc_tokenizer(
            docs,
            padding=True,
            add_special_tokens=self.args.add_special_tokens,
            truncation=True,
            return_tensors='pt'
        ).to(get_model_device(self))
        return self.doc(tokenized_docs['input_ids'], tokenized_docs['attention_mask'])

class ColBERTwKWMask(ColBERT):
    # Implement ColBERT model with only important token from passage are kept

    def __init__(self, args, config):
        super(ColBERTwKWMask, self).__init__(args=args, config=config)

        self.kw_detection = nn.Sequential(
            nn.Linear(in_features=self.bert.config.hidden_size, out_features=args.embedding_hidden_size),
            nn.ReLU(), nn.LayerNorm(normalized_shape=args.embedding_hidden_size), nn.Dropout(p=args.dropout_rate),
            nn.Linear(in_features=args.embedding_hidden_size, out_features=args.embedding_hidden_size, bias=False),
            nn.ReLU(), nn.Dropout(p=args.dropout_rate),
            nn.Linear(in_features=args.embedding_hidden_size, out_features=1, bias=False)
        )
        init.zeros_(self.kw_detection[-1].weight)

    def query(self, input_ids, attention_mask):
        query_embedding = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        query_embedding = self.embedding_head(query_embedding)
        if self.args.query_pad_token is not None:
            query_embedding = F.normalize(query_embedding, p=2, dim=2)
            return query_embedding, None
        else:
            query_mask = attention_mask.unsqueeze(2)
            query_embedding = query_embedding * query_mask.float()
            query_embedding = F.normalize(query_embedding, p=2, dim=2)
            return query_embedding, query_mask.bool()
        
    def doc(self, input_ids, attention_mask):
        full_hidden_states = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).hidden_states
        doc_hiddens = full_hidden_states[-1]
        doc_embedding = self.embedding_head(doc_hiddens)
        # Mask for pad tokens and punctuation
        punctuation_mask = self.mask(input_ids, self.doc_mask_buffer).unsqueeze(2)
        doc_embedding = doc_embedding * punctuation_mask.float()
        doc_embedding = F.normalize(doc_embedding, p=2, dim=2)
        # mask for keyword detection, this mask is soft
        if isinstance(self.args.weight_mask, int):
            weight_mask = torch.sigmoid(self.kw_detection(full_hidden_states[self.args.weight_mask]))
            # weight_mask = torch.arctan(self.kw_detection(full_hidden_states[self.args.weight_mask]))/torch.pi + 0.5
        weight_mask = weight_mask * punctuation_mask.float()
        return doc_embedding, punctuation_mask.bool(), weight_mask
    
    def score_pair(self, query_embedding, doc_embedding, doc_mask, doc_weight_mask, query_mask=None):
        weighted_doc_embedding = doc_embedding * doc_weight_mask    # B * LD * d
        return super(ColBERTwKWMask, self).score_pair(
            query_embedding=query_embedding,
            doc_embedding=weighted_doc_embedding,
            doc_mask=doc_mask,
            query_mask=query_mask
        )

    def score_cart(self, query_embedding, doc_embedding, doc_mask, doc_weight_mask, query_mask=None):
        weighted_doc_embedding = doc_embedding * doc_weight_mask # D * LD * h
        return super(ColBERTwKWMask, self).score_cart(
            query_embedding=query_embedding,
            doc_embedding=weighted_doc_embedding,
            doc_mask=doc_mask,
            query_mask=query_mask
        )
    
    def evaluate_score(self, Q, D, apply_weight=True):
        query_embedding, query_mask = Q
        doc_embedding, punctuation_mask, weight_mask = D
        # Only account for keywords
        filtered_weight_mask = weight_mask.detach().clone()
        filtered_weight_mask[filtered_weight_mask < self.args.kw_threshold] = 0.0
        if not apply_weight:
            filtered_weight_mask[filtered_weight_mask >= self.args.kw_threshold] = 1.0
        return self.score_cart(
            query_embedding=query_embedding,
            doc_embedding=doc_embedding,
            query_mask=query_mask,
            doc_mask=punctuation_mask,
            doc_weight_mask=filtered_weight_mask
        )
 
    def forward(self, Q, D):
        """
        Q: {query_input_ids, query_attention_mask} each with dim (B, L)
        D: {doc_input_ids, doc_attention_mask}
        """
        query_embedding, query_mask = self.query(**Q)
        doc_embedding, punctuation_mask, weight_mask = self.doc(**D)
        if self.training:
            return self.score_cart(
                query_embedding=query_embedding,
                query_mask=query_mask,
                doc_embedding=doc_embedding,
                doc_mask=punctuation_mask,
                doc_weight_mask=weight_mask
            )
        else:
            return self.evaluate_score(
                Q=(query_embedding, query_mask),
                D=(doc_embedding, punctuation_mask, weight_mask)
            )