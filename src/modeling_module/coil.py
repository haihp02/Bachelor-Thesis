import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from modeling_module import *
from modeling_module.utils import *
from args import Args

class COIL(nn.Module, PyTorchModelHubMixin):
    def __init__(self, args: Args, config):
        super(COIL, self).__init__()

        self.args = args

        self.bert = AutoModel.from_pretrained(args.name_or_path, config=config)
        self.tok_embedding_head = nn.Sequential(
            nn.Linear(in_features=self.bert.config.hidden_size, out_features=args.tok_embedding_dim),
            nn.LayerNorm(normalized_shape=args.tok_embedding_dim)
        )
        self.cls_embedding_head = nn.Sequential(
            nn.Linear(in_features=self.bert.config.hidden_size, out_features=args.cls_embedding_dim),
            nn.LayerNorm(normalized_shape=args.cls_embedding_dim)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(args.name_or_path)

    def forward(self, Q, D):
        """
        Q: {query_input_ids, query_attention_mask} each with dim (B, L)
        D: {doc_input_ids, doc_attention_mask}
        """
        query_cls_emb, query_tok_embs = self.embed(**Q)
        doc_cls_emb, doc_tok_embs = self.embed(**D)

        # mask ingredients
        doc_input_ids: torch.Tensor = D['input_ids']
        query_input_ids: torch.Tensor = Q['input_ids']
        query_attention_mask: torch.Tensor = Q['attention_mask']
        self.mask_sep(query_attention_mask)

        if not self.training:
            pass
        else:
            pass

    def score_pair(self, query_tok_embs, doc_tok_embs, doc_input_ids, query_input_ids, query_attention_mask, query_cls_emb=None, doc_cls_emb=None):
        query_attention_mask = self.mask_sep(query_attention_mask)
        assert doc_input_ids.size(0) == query_input_ids.size(0), 'Expect same number of query/ doc!'
        tok_scores = self.tok_score_pair(query_tok_embs, doc_tok_embs, doc_input_ids, query_input_ids, query_attention_mask)    # B

        if self.args.coil_score_type == 'tok_only':
            scores = tok_scores
        elif self.args.coil_score_type == 'full':
            assert query_cls_emb is not None and doc_cls_emb is not None, 'Cls emb missing!'
            cls_scores = (query_cls_emb * doc_cls_emb).sum(-1)
            scores = tok_scores + cls_scores
        return scores
    
    def score_cart(self, query_tok_embs, doc_tok_embs, doc_input_ids, query_input_ids, query_attention_mask, query_cls_emb=None, doc_cls_emb=None):
        query_attention_mask = self.mask_sep(query_attention_mask)
        tok_scores = self.tok_score_cart(query_tok_embs, doc_tok_embs, doc_input_ids, query_input_ids, query_attention_mask)    # Q * D

        if self.args.coil_score_type == 'tok_only':
            scores = tok_scores
        elif self.args.coil_score_type == 'full':
            assert query_cls_emb is not None and doc_cls_emb is not None, 'Cls emb missing!'
            cls_scores = query_cls_emb @ doc_cls_emb.transpose(0, 1)    # Q * D
            scores = tok_scores + cls_scores
        return scores

    def tok_score_pair(self, query_tok_embs, doc_tok_embs, doc_input_ids, query_input_ids, query_attention_mask):
        exact_match = query_input_ids.unsqueeze(2) == doc_input_ids.unsqueeze(1)    # B * LQ * LD
        exact_match = exact_match.float()
        scores_no_masking = query_tok_embs @ doc_tok_embs.permute(0, 2, 1)  # B * LQ * LD
        tok_scores = (scores_no_masking * exact_match).max(dim=-1).values  # B * LQ
        # remove padding and cls token
        tok_scores = (tok_scores * query_attention_mask)[:, 1:].sum(-1)
        # normalized_tok_scores = tok_scores / (query_attention_mask.sum(-1) - 1)  # B
        return tok_scores
    
    def tok_score_cart(self, query_tok_embs, doc_tok_embs, doc_input_ids, query_input_ids, query_attention_mask):
        query_input_ids = query_input_ids.unsqueeze(2).unsqueeze(3) # Q * LQ * 1 * 1
        doc_input_ids = doc_input_ids.unsqueeze(0).unsqueeze(1) # 1 * 1 * D * LD
        exact_match = doc_input_ids == query_input_ids  # Q * LQ * D * LD
        exact_match = exact_match.float()
        scores_no_masking = torch.matmul(
            query_tok_embs.view(-1, self.args.tok_embedding_dim),   # (Q * LQ) * d
            doc_tok_embs.view(-1, self.args.tok_embedding_dim).transpose(0, 1)   # d * (D * LD)
        )
        scores_no_masking = scores_no_masking.view(*query_tok_embs.shape[:2], *doc_tok_embs.shape[:2])  # Q * LQ * D * LD
        scores = (scores_no_masking * exact_match).max(-1).values  # Q * LQ * D
        tok_scores = (scores * query_attention_mask.unsqueeze(-1))[:,1:,:].sum(-2) # Q * D
        # normalized_tok_scores = tok_scores / (query_attention_mask.sum(-1) - 1).unsqueeze(-1)
        return tok_scores

    def embed(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        cls_emb = self.cls_embedding_head(output[:, 0])
        cls_emb = F.normalize(cls_emb, p=2, dim=1)
        tok_embs = self.tok_embedding_head(output)
        tok_embs = F.normalize(tok_embs, p=2, dim=2)
        return cls_emb, tok_embs
    
    def encode(self, text):
        tokenized_text = self.tokenizer(text, truncation=True, max_length=self.args.model_max_length, padding=True, return_tensors='pt').to(get_model_device(self))
        cls_emb, tok_embs = self.embed(input_ids=tokenized_text['input_ids'], attention_mask=tokenized_text['attention_mask'])
        return {
            'input_ids': tokenized_text['input_ids'],
            'attention_mask': tokenized_text['attention_mask'],
            'cls_emb': cls_emb,
            'tok_embs': tok_embs
        }
    
    def encode_query(self, queries):
        if not isinstance(queries, list):
            queries = [queries]
        return self.encode(queries)

    def encode_doc(self, docs):
        if not isinstance(docs, list):
            docs = [docs]
        return self.encode(docs)
    
    def evaluate_score(self, Q, D):
        return self.score_cart(
            query_tok_embs=Q['tok_embs'],
            doc_tok_embs=D['tok_embs'],
            doc_input_ids=D['input_ids'],
            query_input_ids=Q['input_ids'],
            query_attention_mask=Q['attention_mask'],
            query_cls_emb=Q['cls_emb'],
            doc_cls_emb=D['cls_emb']
        )

    def mask_sep(self, attention_mask):
        if self.args.no_sep:
            sep_pos = attention_mask.sum(1).unsqueeze(1) - 1 # the sep token position
            _zeros = torch.zeros_like(sep_pos)
            attention_mask_wo_sep = torch.scatter(attention_mask, 1, sep_pos.long(), _zeros)
        return attention_mask_wo_sep
