import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

from args import Args
from modeling_module.utils import *

class CITADEL(nn.Module):
    def __init__(self, args: Args, config):
        super(CITADEL, self).__init__()

        self.args = args
        
        self.bert = AutoModelForMaskedLM.from_pretrained(args.name_or_path, config=config)

        self.tok_embedding_head = nn.Linear(in_features=self.bert.config.hidden_size, out_features=args.tok_embedding_dim)
        self.cls_embedding_head = nn.Linear(in_features=self.bert.config.hidden_size, out_features=args.cls_embedding_dim)

        nn.init.normal_(self.tok_embedding_head.weight, mean=0, std=0.02)
        nn.init.normal_(self.cls_embedding_head.weight, mean=0, std=0.02)

        self.tokenizer = AutoTokenizer.from_pretrained(args.name_or_path)

    def forward(self, tokens, topk=1):
        pass

    def score_pair(self, query_citadel_emb: dict, doc_citadel_emb: dict, ):
        assert query_citadel_emb['expert_repr'].size()[0] == doc_citadel_emb['expert_repr'].size()[0], 'Expect same number of query/ doc!'
        tok_scores = self.tok_score_pair(query_citadel_emb, doc_citadel_emb)

        if self.args.citadel_score_type == 'tok_only':
            scores = tok_scores
        elif self.args.citadel_score_type == 'full':
            assert 'cls_repr' in query_citadel_emb and 'cls_repr' in doc_citadel_emb, 'Cls emb missing!'
            cls_scores = (query_citadel_emb['cls_repr'] * doc_citadel_emb['cls_repr']).sum(-1) # B
            scores = tok_scores + cls_scores
        return scores
    
    def score_cart(self, query_citadel_emb: dict, doc_citadel_emb: dict):
        tok_scores = self.tok_score_cart(query_citadel_emb, doc_citadel_emb)    # Q * D

        if self.args.citadel_score_type == 'tok_only':
            scores = tok_scores
        elif self.args.citadel_score_type == 'full':
            assert 'cls_repr' in query_citadel_emb and 'cls_repr' in doc_citadel_emb, 'Cls emb missing!'
            cls_scores = query_citadel_emb['cls_repr'] @ doc_citadel_emb['cls_repr'].transpose(0, 1)    # Q * D
            scores = tok_scores + cls_scores
        return scores


    def tok_score_pair(self, query_citadel_emb: dict, doc_citadel_emb: dict):
        # citadel emb already masked
        query_expert_repr, query_expert_weights, query_expert_ids = query_citadel_emb['expert_repr'], query_citadel_emb['expert_weights'], query_citadel_emb['expert_ids']
        doc_expert_repr, doc_expert_weights, doc_expert_ids = doc_citadel_emb['expert_repr'], doc_citadel_emb['expert_weights'], doc_citadel_emb['expert_ids']

        scores_wo_weight_match = torch.einsum('ijd,ikd->ijk', query_expert_repr, doc_expert_repr)    # B * LQ * LD
        weights = query_expert_weights.unsqueeze(-1).unsqueeze(-1) * doc_expert_weights.unsqueeze(1).unsqueeze(1) # B * LQ * KQ * LD * KD  
        exact_match = query_expert_ids.unsqueeze(-1).unsqueeze(-1) == doc_expert_ids.unsqueeze(1).unsqueeze(1) # B * LQ * KQ * LD * KD

        scores = scores_wo_weight_match.unsqueeze(2).unsqueeze(-1) * weights * exact_match.float() # B * LQ * KQ * LD * KD  

        scores = scores.view(scores.shape[0], scores.shape[1] * scores.shape[2], scores.shape[3] * scores.shape[4])
        scores = scores.max(-1).values.sum(-1) # B
        return scores
    
    def tok_score_cart(self, query_citadel_emb: dict, doc_citadel_emb: dict):
        query_expert_repr, query_expert_weights, query_expert_ids = query_citadel_emb['expert_repr'], query_citadel_emb['expert_weights'], query_citadel_emb['expert_ids']
        doc_expert_repr, doc_expert_weights, doc_expert_ids = doc_citadel_emb['expert_repr'], doc_citadel_emb['expert_weights'], doc_citadel_emb['expert_ids']
    
        scores_wo_weight_match = torch.einsum('ikd,jqd->ikjq', query_expert_repr, doc_expert_repr)    # Q * LQ * D * LD
        weights = query_expert_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * doc_expert_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0) # Q * LQ * KQ * D * LD * KD
        exact_match = query_expert_ids.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) == doc_expert_ids.unsqueeze(0).unsqueeze(0).unsqueeze(0) # Q * LQ * KQ * D * LD * KD
        scores = scores_wo_weight_match.unsqueeze(2).unsqueeze(-1) * weights * exact_match.float() # Q * LQ * KQ * D * LD * KD

        scores = scores.view(scores.shape[0], scores.shape[1] * scores.shape[2], scores.shape[3], scores.shape[4] * scores.shape[5]) # Q * (LQ * KQ) * D * (LD * KD)
        scores = scores.max(-1).values.sum(1) # Q * D
        return scores

    def embed(self, input_ids, attention_mask, topk=1):
        result = {}
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        attention_mask = self.mask_sep(attention_mask)[:, 1:]
        hidden_states = output.hidden_states[-1][:, 1:, :]   # v
        logits = output.logits[:, 1:, :]    # B * (L-1) * |V|
        # result
        # router representation
        full_router_repr = torch.log(1 + torch.relu(logits)) * attention_mask.unsqueeze(-1)

        result['router_repr'] = torch.max(full_router_repr, dim=1).values
        # result['router_repr'] = F.normalize(torch.max(full_router_repr, dim=1).values, p=2, dim=1) # B * |V|, to train roter with constrastive loss Lr
        # routing, assign every token to top-k expert
        result['expert_weights'], result['expert_ids'] = torch.topk(full_router_repr, dim=2, k=topk) # B x T x E
        # expert representation
        result['expert_repr'] = self.tok_embedding_head(hidden_states) * attention_mask.unsqueeze(-1)
        # result['expert_repr'] = F.normalize(self.tok_embedding_head(hidden_states), p=2, dim=-1) * attention_mask.unsqueeze(-1)
        
        if self.training:
            result['router_mask'] = torch.zeros_like(full_router_repr)
            result['router_mask'].scatter_(dim=2, index=result['expert_ids'], src=(result['expert_weights'] > 0.).to(result['expert_weights'].dtype)) # B x T x |V|
            router_softmax_repr = torch.softmax(logits, dim=-1) * attention_mask.unsqueeze(-1) # B * (L-1) * |V|
            result["router_softmax_repr"] = router_softmax_repr.sum(1) # B * |V|

        if self.args.citadel_score_type == 'tok_only':
            return result
        elif self.args.citadel_score_type == 'full':
            result['cls_repr'] = self.cls_embedding_head(output.hidden_states[-1][:, 0, :])
            # result['cls_repr'] = F.normalize(self.cls_embedding_head(output.hidden_states[-1][:, 0, :]), p=2, dim=-1)
            return result
        
    def encode(self, text, topk):
        tokenized_text = self.tokenizer(text, truncation=True, max_length=self.args.model_max_length, padding=True, return_tensors='pt').to(get_model_device(self))
        citadel_embed = self.embed(input_ids=tokenized_text['input_ids'], attention_mask=tokenized_text['attention_mask'], topk=topk)
        return citadel_embed

    def encode_query(self, query):
        return self.encode(query, topk=1)
    
    def encode_doc(self, doc):
        return self.encode(doc, topk=5)
    
    def evaluate_score(self, Q, D):
        return self.score_cart(
            query_citadel_emb=Q,
            doc_citadel_emb=D
        )

    def mask_sep(self, attention_mask):
        if self.args.no_sep:
            sep_pos = attention_mask.sum(1).unsqueeze(1) - 1 # the sep token position
            _zeros = torch.zeros_like(sep_pos)
            attention_mask_wo_sep = torch.scatter(attention_mask, 1, sep_pos.long(), _zeros)
        return attention_mask_wo_sep