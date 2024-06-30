import math
from tqdm.auto import tqdm

from torch.nn.modules import Module
from accelerate.logging import get_logger

from training_and_evaluating_module.trainer import BertTrainer
from training_and_evaluating_module.evaluate import *
from data_module import DataModule
from data_module.ir_data_module import IRTripletDataModule
from modeling_module.colbert import ColBERT, ColBERTwKWMask
from modeling_module.coil import COIL
from modeling_module.citadel import CITADEL
from modeling_module.loss import *

logger = get_logger(__name__)

class ColBERTTripletTrainer(BertTrainer):
    def __init__(self, args):
        super(ColBERTTripletTrainer, self).__init__(args)
    
    def prepare(self, model: Module, data: DataModule):
        super().prepare(model, data)

    def fit(self, model: ColBERT, data: IRTripletDataModule):
        self.prepare(model=model, data=data)
        self.model: ColBERT

        first_epoch, progress_bar, global_step = self.setup()
        targets = torch.zeros(self.args.batch_size, dtype=torch.long, device=self.accelerator.device)
        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.model.train()
            train_loss = 0.0
            train_passage_score = 0.0
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    query_embedding, query_mask = self.model.query(batch['query']['input_ids'], batch['query']['attention_mask'])
                    passage_embedding, passage_mask = self.model.doc(batch['passage']['input_ids'], batch['passage']['attention_mask'])
                    neg_passage_embedding, neg_passage_mask = self.model.doc(batch['neg_passage']['input_ids'], batch['neg_passage']['attention_mask'])

                    passage_scores = self.model.score_pair(
                        query_embedding=query_embedding,
                        query_mask=query_mask,
                        doc_embedding=passage_embedding,
                        doc_mask=passage_mask,
                    )
                    negative_passage_scores = self.model.score_pair(
                        query_embedding=query_embedding,
                        query_mask=query_mask,
                        doc_embedding=neg_passage_embedding,
                        doc_mask=neg_passage_mask,
                    )
                    
                    if self.args.ir_loss_type == 'pairwise_cross_entropy':
                        paiwise_cross_entropy_loss = pairwise_softmax_cross_entropy_loss(
                            passage_scores,
                            negative_passage_scores,
                            targets
                        )
                        loss = paiwise_cross_entropy_loss
                    elif self.args.ir_loss_type == 'online_contrastive_loss':
                        positive_scores = passage_scores
                        negative_scores = negative_passage_scores
                        scores = torch.cat([positive_scores, negative_scores], dim=0)
                        labels = torch.cat([targets+1, targets], dim=0)
                        loss = online_contrastive_loss(scores=scores, labels=labels)
                    else:
                        positive_passage_in_batch_score = self.model.score_cart(
                            query_embeddings=query_embedding,
                            query_masks=query_mask,
                            doc_embeddings=passage_embedding,
                            doc_masks=passage_mask
                        )
                        negative_passage_in_batch_score = self.model.score_cart(
                            query_embeddings=query_embedding,
                            query_masks=query_mask,
                            doc_embeddings=neg_passage_embedding,
                            doc_masks=neg_passage_mask
                        )
                        loss = margin_loss(positive_score=passage_scores, negative_score=negative_passage_in_batch_score.diagonal(), targets=targets.float() + 1) \
                            + (in_batch_negative_constrastive_loss(positive_score=passage_scores, negative_scores=negative_passage_in_batch_score, targets=targets) \
                            + in_batch_negative_constrastive_loss(positive_score=passage_scores, negative_scores=positive_passage_in_batch_score, targets=targets)).mean() \
                            + (1/math.sqrt(global_step + 1))*positive_pair_loss(positive_score=passage_scores, targets=targets.float() + 1)
                    # Gather the losses and positive passages scores across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.args.batch_size)).mean()
                    avg_passage_score = self.accelerator.gather(passage_scores.repeat(self.args.batch_size)).mean()
                    train_loss += avg_loss.item() / self.args.gradient_accumulation_steps
                    train_passage_score += avg_passage_score.item() / self.args.gradient_accumulation_steps

                    # Backpropagate
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.accelerator.log({'train_loss': train_loss, 'train_passage_score': train_passage_score, 'grad_norm': grad_norm}, step=global_step)
                    train_loss = 0.0
                    train_passage_score = 0.0

                    self.checkpointing_check(global_step=global_step)
                    if global_step >= self.args.max_train_steps:
                        break

            self.model.eval()
            samples_seen = 0
            total_score = 0
            for step, batch in enumerate(self.val_dataloader):
                with torch.no_grad():
                    query_embedding, query_mask = self.model.query(batch['query']['input_ids'], batch['query']['attention_mask'])
                    passage_embedding, passage_mask = self.model.doc(batch['passage']['input_ids'], batch['passage']['attention_mask'])
                    scores = self.model.score_pair(
                        query_embedding=query_embedding,
                        doc_embedding=passage_embedding,
                        doc_mask=passage_mask,
                        query_mask=query_mask
                    )
                    scores_gathered = self.accelerator.gather(scores)
                    # If we are in a multiprocess environment, the last batch has duplicates
                    if self.accelerator.num_processes > 1:
                        if step == len(self.val_dataloader) - 1:
                            scores_gathered = scores_gathered[: len(self.val_dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += scores_gathered.shape[0]
                    total_score += scores_gathered.sum().item()
            val_avg_score = total_score/len(self.val_dataloader.dataset)
            self.accelerator.print(f'epoch {epoch}, avg score:', val_avg_score)
            self.accelerator.log({'val_avg_score': val_avg_score})
        self.accelerator.end_training()


class COILIRTripletTrainer(BertTrainer):
    def __init__(self, args):
        super(COILIRTripletTrainer, self).__init__(args)

    def fit(self, model: COIL, data):
        self.prepare(model=model, data=data)
        first_epoch, progress_bar, global_step = self.setup()
        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.model.train()
            train_loss = 0.0
            train_passage_score = 0.0
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    query_cls_emb, query_tok_embs = self.model.embed(
                        input_ids=batch['query']['input_ids'], attention_mask=batch['query']['attention_mask']
                    )
                    passage_cls_emb, passage_tok_embs = self.model.embed(
                        input_ids=batch['passage']['input_ids'], attention_mask=batch['passage']['attention_mask']
                    )
                    neg_passage_cls_emb, neg_passage_tok_embs = self.model.embed(
                        input_ids=batch['neg_passage']['input_ids'], attention_mask=batch['neg_passage']['attention_mask']
                    )

                    passage_scores = self.model.score_pair(
                        query_tok_embs=query_tok_embs,
                        doc_tok_embs=passage_tok_embs,
                        doc_input_ids=batch['passage']['input_ids'],
                        query_input_ids=batch['query']['input_ids'],
                        query_attention_mask=batch['query']['attention_mask'],
                        query_cls_emb=query_cls_emb,
                        doc_cls_emb=passage_cls_emb
                    )
                    neg_passage_scores_cart = self.model.score_cart(
                        query_tok_embs=query_tok_embs,
                        doc_tok_embs=neg_passage_tok_embs,
                        doc_input_ids=batch['neg_passage']['input_ids'],
                        query_input_ids=batch['query']['input_ids'],
                        query_attention_mask=batch['query']['attention_mask'],
                        query_cls_emb=query_cls_emb,
                        doc_cls_emb=neg_passage_cls_emb
                    )

                    loss = coil_online_constrastive_loss(
                        positive_scores=passage_scores,
                        negative_scores_cart=neg_passage_scores_cart,
                        query_attention_mask=batch['query']['attention_mask']
                    )
    
                    # Gather the losses and positive passages scores across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.args.batch_size)).mean()
                    avg_passage_score = self.accelerator.gather(passage_scores.repeat(self.args.batch_size)).mean()
                    train_loss += avg_loss.item() / self.args.gradient_accumulation_steps
                    train_passage_score += avg_passage_score.item() / self.args.gradient_accumulation_steps

                    # Backpropagate
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.accelerator.log({'train_loss': train_loss, 'train_passage_score': train_passage_score, 'grad_norm': grad_norm}, step=global_step)
                    train_loss = 0.0
                    train_passage_score = 0.0

                    self.checkpointing_check(global_step=global_step)
                    if global_step >= self.args.max_train_steps:
                        break
            
            self.model.eval()
            samples_seen = 0
            total_score = 0
            for step, batch in enumerate(self.val_dataloader):
                with torch.no_grad():
                    query_cls_emb, query_tok_embs = self.model.embed(
                        input_ids=batch['query']['input_ids'], attention_mask=batch['query']['attention_mask']
                    )
                    passage_cls_emb, passage_tok_embs = self.model.embed(
                        input_ids=batch['passage']['input_ids'], attention_mask=batch['passage']['attention_mask']
                    )
                    scores = self.model.score_pair(
                        query_tok_embs=query_tok_embs,
                        doc_tok_embs=passage_tok_embs,
                        doc_input_ids=batch['passage']['input_ids'],
                        query_input_ids=batch['query']['input_ids'],
                        query_attention_mask=batch['query']['attention_mask'],
                        query_cls_emb=query_cls_emb,
                        doc_cls_emb=passage_cls_emb
                    )
                    scores_gathered = self.accelerator.gather(scores)
                    # If we are in a multiprocess environment, the last batch has duplicates
                    if self.accelerator.num_processes > 1:
                        if step == len(self.val_dataloader) - 1:
                            scores_gathered = scores_gathered[: len(self.val_dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += scores_gathered.shape[0]
                    total_score += scores_gathered.sum().item()
            val_avg_score = total_score/len(self.val_dataloader.dataset)
            self.accelerator.print(f'epoch {epoch}, avg score:', val_avg_score)
            self.accelerator.log({'val_avg_score': val_avg_score})
        self.accelerator.end_training()

class CITADELIRTripletTrainer(BertTrainer):
    def __init__(self, args):
        super(CITADELIRTripletTrainer, self).__init__(args)

    def fit(self, model: CITADEL, data):
        self.prepare(model=model, data=data)
        first_epoch, progress_bar, global_step = self.setup()
        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.model.train()
            train_loss = 0.0
            train_passage_score = 0.0
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    query_emb = self.model.embed(
                        input_ids=batch['query']['input_ids'], attention_mask=batch['query']['attention_mask']
                    )
                    passage_emb = self.model.embed(
                        input_ids=batch['passage']['input_ids'], attention_mask=batch['passage']['attention_mask'], topk=5
                    )
                    neg_passage_emb = self.model.embed(
                        input_ids=batch['neg_passage']['input_ids'], attention_mask=batch['neg_passage']['attention_mask'], topk=5
                    )

                    passage_scores = self.model.score_pair(query_emb, passage_emb)
                    neg_passage_scores_cart = self.model.score_cart(query_emb, neg_passage_emb) 

                    constrastive_loss, router_constrastive_loss, regularization_loss, load_balancing_loss = citadel_loss(
                        positive_scores=passage_scores,
                        negative_scores_cart=neg_passage_scores_cart,
                        citadel_query_emb=query_emb,
                        citadel_doc_emb=passage_emb,
                        citadel_neg_doc_emb=neg_passage_emb,
                        query_attention_mask=batch['query']['attention_mask'],
                        alpha=self.args.expert_regularization_coef,
                        beta=self.args.expert_load_balancing_coef
                    )
                    loss = constrastive_loss + router_constrastive_loss + regularization_loss + load_balancing_loss
                    # print(constrastive_loss, router_constrastive_loss, regularization_loss, load_balancing_loss)
    
                    # Gather the losses and positive passages scores across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.args.batch_size)).mean()
                    avg_passage_score = self.accelerator.gather(passage_scores.repeat(self.args.batch_size)).mean()
                    train_loss += avg_loss.item() / self.args.gradient_accumulation_steps
                    train_passage_score += avg_passage_score.item() / self.args.gradient_accumulation_steps

                    # Backpropagate
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.accelerator.log({'train_loss': train_loss, 'train_passage_score': train_passage_score, 'grad_norm': grad_norm}, step=global_step)
                    train_loss = 0.0
                    train_passage_score = 0.0

                    self.checkpointing_check(global_step=global_step)
                    if global_step >= self.args.max_train_steps:
                        break
            
            self.model.eval()
            samples_seen = 0
            total_score = 0
            for step, batch in enumerate(self.val_dataloader):
                with torch.no_grad():
                    query_emb = self.model.embed(
                        input_ids=batch['query']['input_ids'], attention_mask=batch['query']['attention_mask']
                    )
                    passage_emb = self.model.embed(
                        input_ids=batch['passage']['input_ids'], attention_mask=batch['passage']['attention_mask']
                    )
                    scores = self.model.score_pair(query_emb, passage_emb)
                    scores_gathered = self.accelerator.gather(scores)
                    # If we are in a multiprocess environment, the last batch has duplicates
                    if self.accelerator.num_processes > 1:
                        if step == len(self.val_dataloader) - 1:
                            scores_gathered = scores_gathered[: len(self.val_dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += scores_gathered.shape[0]
                    total_score += scores_gathered.sum().item()
            val_avg_score = total_score/len(self.val_dataloader.dataset)
            self.accelerator.print(f'epoch {epoch}, avg score:', val_avg_score)
            self.accelerator.log({'val_avg_score': val_avg_score})
        self.accelerator.end_training()

class ColBERTwKWMaskTripletTrainer(BertTrainer):
    def __init__(self, args):
        super(ColBERTwKWMaskTripletTrainer, self).__init__(args)
    
    def prepare(self, model: Module, data: IRTripletDataModule):
        super(ColBERTwKWMaskTripletTrainer, self).prepare(model, data)
        self.cur_regularize_coef = 0

    def kw_regularize_schedule(self, global_step):
        if global_step >= self.args.kw_warmup_steps or self.args.kw_warmup_steps == 0:
            self.cur_regularize_coef = self.args.regularize_coef
        else:
            self.cur_regularize_coef = (global_step / self.args.kw_warmup_steps) * self.args.regularize_coef

    def fit(self, model: ColBERTwKWMask, data):
        self.prepare(model=model, data=data)
        first_epoch, progress_bar, global_step = self.setup()
        targets = torch.zeros(self.args.batch_size, dtype=torch.long, device=self.accelerator.device)
        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.model.train()
            train_loss = 0.0
            regularize_loss = 0.0
            train_passage_score = 0.0
            train_acc = 0.0
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    query_embedding, query_mask = self.model.query(batch['query']['input_ids'], batch['query']['attention_mask'])
                    passage_embedding, passage_mask, passage_weight_mask = self.model.doc(batch['passage']['input_ids'], batch['passage']['attention_mask'])
                    neg_passage_embedding, neg_passage_mask, neg_passage_weight_mask = self.model.doc(batch['neg_passage']['input_ids'], batch['neg_passage']['attention_mask'])

                    positive_scores = self.model.score_pair(
                        query_embedding=query_embedding,
                        query_mask=query_mask,
                        doc_embedding=passage_embedding,
                        doc_mask=passage_mask,
                        doc_weight_mask=passage_weight_mask,
                    )
                    negative_scores = self.model.score_pair(
                        query_embedding=query_embedding,
                        query_mask=query_mask,
                        doc_embedding=neg_passage_embedding,
                        doc_mask=neg_passage_mask,
                        doc_weight_mask=neg_passage_weight_mask,
                    )
                    
                    acc = (positive_scores > negative_scores).sum() / positive_scores.size(0)

                    scores = torch.cat([positive_scores, negative_scores], dim=0)
                    labels = torch.cat([targets+1, targets], dim=0)
                    loss = online_contrastive_loss(scores=scores, labels=labels)

                    if self.args.kw_regularize == 'unnormalize':
                        # Sum everything, do not count for passage length
                        kw_regularize_loss = torch.cat([passage_weight_mask.squeeze().sum(-1), neg_passage_weight_mask.squeeze().sum(-1)], dim=0).sum()
                        loss += self.cur_regularize_coef * kw_regularize_loss
                    elif self.args.kw_regularize == 'normalized':
                        pos_kw_regularize_loss = passage_weight_mask.squeeze().sum(-1) / passage_mask.squeeze().sum(-1)
                        neg_kw_regularize_loss = neg_passage_weight_mask.squeeze().sum(-1) / neg_passage_mask.squeeze().sum(-1)
                        kw_regularize_loss = self.cur_regularize_coef * (pos_kw_regularize_loss.sum() + neg_kw_regularize_loss.sum())
                        loss +=  kw_regularize_loss
                    # Gather the losses and positive passages scores across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.args.batch_size)).mean()
                    avg_regularize_loss = self.accelerator.gather(kw_regularize_loss.repeat(self.args.batch_size)).mean()
                    avg_passage_score = self.accelerator.gather(positive_scores.repeat(self.args.batch_size)).mean()
                    avg_acc = self.accelerator.gather(acc.repeat(self.args.batch_size)).mean()

                    train_loss += avg_loss.item() / self.args.gradient_accumulation_steps
                    regularize_loss += avg_regularize_loss.item() / self.args.gradient_accumulation_steps
                    train_passage_score += avg_passage_score.item()
                    train_acc = avg_acc.item()

                    # Backpropagate
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.kw_regularize_schedule(global_step=global_step)
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.accelerator.log({
                        'train_loss': train_loss,
                        'regularize_loss': regularize_loss,
                        'train_passage_score': train_passage_score,
                        'train_acc': train_acc,
                        'grad_norm': grad_norm
                    }, step=global_step)
                    train_loss = 0.0
                    regularize_loss = 0.0
                    train_passage_score = 0.0

                    self.checkpointing_check(global_step=global_step)
                    if global_step >= self.args.max_train_steps:
                        break

            self.model.eval()
            samples_seen = 0
            total_score = 0
            for step, batch in enumerate(self.val_dataloader):
                with torch.no_grad():
                    query_embedding, query_mask = self.model.query(batch['query']['input_ids'], batch['query']['attention_mask'])
                    passage_embedding, passage_mask, passage_weight_mask = self.model.doc(batch['passage']['input_ids'], batch['passage']['attention_mask'])
                    scores = self.model.score_pair(
                        query_embedding=query_embedding,
                        doc_embedding=passage_embedding,
                        doc_mask=passage_mask,
                        query_mask=query_mask,
                        doc_weight_mask=passage_weight_mask,
                    )
                    scores_gathered = self.accelerator.gather(scores)
                    # If we are in a multiprocess environment, the last batch has duplicates
                    if self.accelerator.num_processes > 1:
                        if step == len(self.val_dataloader) - 1:
                            scores_gathered = scores_gathered[: len(self.val_dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += scores_gathered.shape[0]
                    total_score += scores_gathered.sum().item()
            val_avg_score = total_score/len(self.val_dataloader.dataset)
            self.accelerator.print(f'epoch {epoch}, avg score:', val_avg_score)
            self.accelerator.log({'val_avg_score': val_avg_score})
        self.accelerator.end_training()


class ColBERTwKWMaskDistillTrainer(ColBERTwKWMaskTripletTrainer):
    def __init__(self, args):
        super(ColBERTwKWMaskDistillTrainer, self).__init__(args)   

    def prepare(self, model: Module, data: IRTripletDataModule):
        super().prepare(model, data)
        # Decide to freeze ColBERT embedding layer
        for param in self.model.embedding_head.parameters():
            param.requires_grad = self.args.train_colbert_embedding_head

    def get_maxsim_mask_pair(self, query_embedding, doc_embedding, doc_mask, query_mask):
        scores_no_masking = query_embedding @ doc_embedding.permute(0, 2, 1) # B * LQ * LD

        mask = doc_mask.permute(0, 2, 1).repeat([1, query_embedding.size(1), 1])
        masked_scores = scores_no_masking * mask.float()
        indices_non_mask = masked_scores.max(-1).indices
        all_first_index_indices = indices_non_mask[:,0].unsqueeze(-1).expand(-1, indices_non_mask.size(1))
        # masked tokens in query will not count
        indices_non_mask[query_mask.squeeze() == 0] = all_first_index_indices[query_mask.squeeze() == 0]   # this is now masked
        return indices_non_mask

    def fit(self, model: ColBERTwKWMask, data: IRTripletDataModule):
        self.prepare(model=model, data=data)
        self.model: ColBERTwKWMask

        first_epoch, progress_bar, global_step = self.setup()
        targets = torch.zeros(self.args.batch_size, dtype=torch.long, device=self.accelerator.device)
        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.model.train()
            train_loss = 0.0
            regularize_loss = 0.0
            train_passage_score = 0.0
            train_acc = 0.0
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    query_embedding, query_mask = self.model.query(batch['query']['input_ids'], batch['query']['attention_mask'])
                    passage_embedding, passage_mask, passage_weight_mask = self.model.doc(batch['passage']['input_ids'], batch['passage']['attention_mask'])
                    neg_passage_embedding, neg_passage_mask, neg_passage_weight_mask = self.model.doc(batch['neg_passage']['input_ids'], batch['neg_passage']['attention_mask'])

                    # IR
                    positive_scores = self.model.score_pair(
                        query_embedding=query_embedding,
                        query_mask=query_mask,
                        doc_embedding=passage_embedding,
                        doc_mask=passage_mask,
                        doc_weight_mask=passage_weight_mask,
                    )
                    negative_scores = self.model.score_pair(
                        query_embedding=query_embedding,
                        query_mask=query_mask,
                        doc_embedding=neg_passage_embedding,
                        doc_mask=neg_passage_mask,
                        doc_weight_mask=neg_passage_weight_mask,
                    )
                    scores = torch.cat([positive_scores, negative_scores], dim=0)
                    labels = torch.cat([targets+1, targets], dim=0)
                    ir_loss = online_contrastive_loss(scores=scores, labels=labels)
                    
                    # Distill
                    positive_maxsim_indices = self.get_maxsim_mask_pair(
                        query_embedding=query_embedding,
                        query_mask=query_mask,
                        doc_embedding=passage_embedding,
                        doc_mask=passage_mask
                    )
                    distill_loss = mask_distill_loss(passage_weight_mask.squeeze(), positive_maxsim_indices)
                    if self.args.distill_negative:
                        negative_maxsim_indices = self.get_maxsim_mask_pair(
                            query_embedding=query_embedding,
                            query_mask=query_mask,
                            doc_embedding=neg_passage_embedding,
                            doc_mask=neg_passage_mask
                        )
                        distill_loss += mask_distill_loss(neg_passage_weight_mask.squeeze(), negative_maxsim_indices)

                    # Regularize
                    if self.args.kw_regularize == 'unnormalize':
                        # Sum everything, do not count for passage length
                        kw_regularize_loss = torch.cat([passage_weight_mask.squeeze().sum(-1), neg_passage_weight_mask.squeeze().sum(-1)], dim=0).sum()
                    elif self.args.kw_regularize == 'normalized':
                        pos_kw_regularize_loss = passage_weight_mask.squeeze().sum(-1) / passage_mask.squeeze().sum(-1)
                        neg_kw_regularize_loss = neg_passage_weight_mask.squeeze().sum(-1) / neg_passage_mask.squeeze().sum(-1)
                        kw_regularize_loss = pos_kw_regularize_loss.sum() + neg_kw_regularize_loss.sum()

                    loss = ir_loss \
                            + self.args.distill_coeff * distill_loss \
                            + self.cur_regularize_coef * kw_regularize_loss
                    acc = (positive_scores > negative_scores).sum() / positive_scores.size(0)

                    # Gather the losses and positive passages scores across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.args.batch_size)).mean()
                    avg_regularize_loss = self.accelerator.gather(kw_regularize_loss.repeat(self.args.batch_size)).mean()
                    avg_passage_score = self.accelerator.gather(positive_scores.repeat(self.args.batch_size)).mean()
                    avg_acc = self.accelerator.gather(acc.repeat(self.args.batch_size)).mean()

                    train_loss += avg_loss.item() / self.args.gradient_accumulation_steps
                    regularize_loss += avg_regularize_loss.item() / self.args.gradient_accumulation_steps
                    train_passage_score += avg_passage_score.item() / self.args.gradient_accumulation_steps
                    train_acc += avg_acc.item() / self.args.gradient_accumulation_steps

                    # Backpropagate
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.kw_regularize_schedule(global_step=global_step)
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.accelerator.log({
                        'train_loss': train_loss,
                        'regularize_loss': regularize_loss,
                        'train_passage_score': train_passage_score,
                        'train_acc': train_acc,
                        'grad_norm': grad_norm
                    }, step=global_step)
                    train_loss = 0.0
                    regularize_loss = 0.0
                    train_passage_score = 0.0
                    train_acc = 0.0

                    self.checkpointing_check(global_step=global_step)
                    if global_step >= self.args.max_train_steps:
                        break

            self.model.eval()
            samples_seen = 0
            total_score = 0
            for step, batch in enumerate(self.val_dataloader):
                with torch.no_grad():
                    query_embedding, query_mask = self.model.query(batch['query']['input_ids'], batch['query']['attention_mask'])
                    passage_embedding, passage_mask, passage_weight_mask = self.model.doc(batch['passage']['input_ids'], batch['passage']['attention_mask'])
                    scores = self.model.score_pair(
                        query_embedding=query_embedding,
                        doc_embedding=passage_embedding,
                        doc_mask=passage_mask,
                        query_mask=query_mask,
                        doc_weight_mask=passage_weight_mask,
                    )
                    scores_gathered = self.accelerator.gather(scores)
                    # If we are in a multiprocess environment, the last batch has duplicates
                    if self.accelerator.num_processes > 1:
                        if step == len(self.val_dataloader) - 1:
                            scores_gathered = scores_gathered[: len(self.val_dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += scores_gathered.shape[0]
                    total_score += scores_gathered.sum().item()
            val_avg_score = total_score/len(self.val_dataloader.dataset)
            self.accelerator.print(f'epoch {epoch}, avg score:', val_avg_score)
            self.accelerator.log({'val_avg_score': val_avg_score})
        self.accelerator.end_training()