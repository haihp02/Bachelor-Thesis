from typing import Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def bert_ner_cross_entroly_loss(logits, targets, ignore_index):
    # logits have shape [B, L, H]
    return F.cross_entropy(
        input=logits.permute(0, 2, 1),
        target=targets,
        ignore_index=ignore_index
    )

def pairwise_softmax_cross_entropy_loss(positive_score, negative_score, targets):
    # logits have shape [B, 2], the 2 dimensions are for positive pair scores and negative pair scores:
    # target = torch.zeros_like(logits, dtype=torch.long, device=logits.device)
    logits = torch.stack([positive_score, negative_score], dim=1)
    return F.cross_entropy(logits, targets)

def in_batch_negative_constrastive_loss(positive_score, negative_scores, targets):
    logits = torch.cat([positive_score.unsqueeze(1), negative_scores], dim=1)
    return F.cross_entropy(logits, targets)

def positive_pair_loss(positive_score, targets):
    return F.mse_loss(positive_score, targets)

def margin_loss(positive_score, negative_score, targets):
    return F.margin_ranking_loss(positive_score, negative_score, targets)

def online_contrastive_loss(scores, labels, margin=0.5):
    distances = 1 - scores
    negs = distances[labels == 0]
    poss = distances[labels == 1]

    # select hard sample
    negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
    positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

    positive_loss = positive_pairs.pow(2).sum()
    negative_loss = F.relu(margin - negative_pairs).pow(2).sum()
    loss = positive_loss + negative_loss
    return loss

def coil_online_constrastive_loss_dynamic(positive_scores: torch.Tensor, negative_scores_cart: torch.Tensor, margin=0.5):
    negs_upper_bound = positive_scores.unsqueeze(-1).detach().clone()   # make sure the bound are not part of gradient graph
    poss_lower_bound = negative_scores_cart.max(dim=1).values.detach().clone()  # if bound gradient are backwarded, this will collapse

    positive_loss = ((poss_lower_bound + margin - positive_scores)[positive_scores < poss_lower_bound]).pow(2).sum()
    negative_loss = ((negs_upper_bound - margin - negative_scores_cart)[negative_scores_cart > negs_upper_bound]).pow(2).sum()
    loss = positive_loss + negative_loss
    return loss

def coil_online_constrastive_loss(positive_scores: torch.Tensor, negative_scores_cart: torch.Tensor, query_attention_mask: torch.Tensor, pos_target=1, neg_target=0):
    positive_scores = positive_scores / query_attention_mask.sum(-1)
    negative_scores_cart = negative_scores_cart / query_attention_mask.sum(-1).unsqueeze(-1)

    negs_upper_bound = positive_scores.unsqueeze(-1)
    poss_lower_bound = negative_scores_cart.max(dim=1).values

    positive_loss = ((pos_target - positive_scores)[positive_scores < poss_lower_bound]).pow(2).sum()
    negative_loss = ((neg_target - negative_scores_cart)[negative_scores_cart > negs_upper_bound]).pow(2).sum()
    loss = positive_loss + negative_loss
    return loss

def query_normalized_constrastive_loss(positive_scores: torch.Tensor, negative_scores_cart: torch.Tensor, pos_target=1, neg_target=0):
    scores = torch.cat([positive_scores.unsqueeze(1), negative_scores_cart], dim=1)
    scores_norm = torch.norm(scores, p=2, dim=-1).detach()
    scores = scores / scores_norm.unsqueeze(-1)
    positive_scores, negative_scores_cart = scores[:,0], scores[:,1:]

    negs_upper_bound = positive_scores.unsqueeze(-1)
    poss_lower_bound = negative_scores_cart.max(dim=1).values

    positive_loss = ((pos_target - positive_scores)[positive_scores < poss_lower_bound]).pow(2).sum()
    negative_loss = ((neg_target - negative_scores_cart)[negative_scores_cart > negs_upper_bound]).pow(2).sum()
    loss = positive_loss + negative_loss
    return loss
        
def citadel_loss(
        positive_scores,
        negative_scores_cart,
        citadel_query_emb: Dict[str, torch.Tensor],
        citadel_doc_emb: Dict[str, torch.Tensor],
        citadel_neg_doc_emb: Dict[str, torch.Tensor],
        query_attention_mask,
        alpha,
        beta
    ):
    batch_size = positive_scores.size(0)

    def constrastive_loss():
        # scores = torch.cat([positive_scores.unsqueeze(1), negative_scores_cart], dim=1)
        # targets = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
        # return F.cross_entropy(scores, targets)
        # return online_cross_entrpy(scores, targets)
        return query_normalized_constrastive_loss(positive_scores=positive_scores, negative_scores_cart=negative_scores_cart)
    
    def router_constrastive_loss():
        query_router_repr = citadel_query_emb['router_repr']
        doc_router_repr = citadel_doc_emb['router_repr']
        neg_doc_router_repr = citadel_neg_doc_emb['router_repr']
        normalize_constant = query_router_repr.size(-1)
        router_repr_sim = (query_router_repr*doc_router_repr).sum(-1)   # B
        neg_router_repr_sim_cart = query_router_repr @ neg_doc_router_repr.transpose(0, 1) # B * B
        router_repr_sim = router_repr_sim / normalize_constant
        neg_router_repr_sim_cart = neg_router_repr_sim_cart / normalize_constant
        # repr_sim = torch.cat([router_repr_sim.unsqueeze(1), neg_router_repr_sim_cart], dim=1) / normalize_constant
        # targets = torch.zeros(repr_sim.size(0), device=repr_sim.device, dtype=torch.long)
        # return F.cross_entropy(repr_sim, targets)
        # return online_cross_entrpy(repr_sim, targets)
        return query_normalized_constrastive_loss(positive_scores=router_repr_sim, negative_scores_cart=neg_router_repr_sim_cart, neg_target=0)
                
    def regularization_loss():
        if beta > 0:
            return beta * (citadel_query_emb['expert_weights'].sum(-1).sum(-1)
                            + citadel_doc_emb['expert_weights'].sum(-1).sum(-1)
                            + citadel_neg_doc_emb['expert_weights'].sum(-1).sum(-1)).sum() / batch_size
        else:
            return 0

    def load_balancing_loss():
        if alpha > 0:
            query_load_balancing_loss = (citadel_query_emb['router_mask'].sum(0) * citadel_query_emb['router_softmax_repr'].sum(0)).sum()
            doc_load_balancing_loss = (citadel_doc_emb['router_mask'].sum(0) * citadel_doc_emb['router_softmax_repr'].sum(0)).sum()
            neg_doc_load_balancing_loss = (citadel_neg_doc_emb['router_mask'].sum(0) * citadel_neg_doc_emb['router_softmax_repr'].sum(0)).sum()
            return alpha * (query_load_balancing_loss + doc_load_balancing_loss + neg_doc_load_balancing_loss) / batch_size
        else:
            return 0

    return constrastive_loss(), router_constrastive_loss(), regularization_loss(), load_balancing_loss()


def mask_distill_loss(weight_mask: torch.Tensor, maxsim_indices: torch.Tensor):
    weight_mask = weight_mask.squeeze()
    target = torch.ones_like(weight_mask, dtype=torch.float32)  # B * LD
    target_mask = (1-target).long().scatter(dim=-1, index=maxsim_indices, value=1)
    # loss_wo_mask = target - weight_mask # B * Q * D
    # loss = (loss_wo_mask * target_mask).sum()
    loss_wo_mask_sqr = (target - weight_mask)**2
    loss = (loss_wo_mask_sqr * target_mask).sum().sqrt()
    return loss.clone()

def mask_distill_loss_with_ner_guide(weight_mask: torch.Tensor, maxsim_indices: torch.Tensor, ner_tags_ids: torch.Tensor):
    weight_mask = weight_mask.squeeze()
    target = torch.ones_like(weight_mask, dtype=torch.float32)  # B * LD
    target_mask = (1-target).long().scatter(dim=-1, index=maxsim_indices, value=1)
    target_mask[ner_tags_ids > 0] = 1
    # loss_wo_mask = target - weight_mask # B * Q * D
    # loss = (loss_wo_mask * target_mask).sum()
    loss_wo_mask_sqr = (target - weight_mask)**2
    loss = (loss_wo_mask_sqr * target_mask).sum().sqrt()
    return loss.clone()

def mask_distill_loss_with_pos_guide(weight_mask: torch.Tensor, maxsim_indices: torch.Tensor, pos_tags_ids: torch.Tensor):
    weight_mask = weight_mask.squeeze()
    target = torch.ones_like(weight_mask, dtype=torch.float32)  # B * LD
    target_mask = (1-target).long().scatter(dim=-1, index=maxsim_indices, value=1)
    target_mask[pos_tags_ids > 0] = 1
    # loss_wo_mask = target - weight_mask # B * Q * D
    # loss = (loss_wo_mask * target_mask).sum()
    loss_wo_mask_sqr = (target - weight_mask)**2
    loss = (loss_wo_mask_sqr * target_mask).sum().sqrt()
    return loss.clone()
