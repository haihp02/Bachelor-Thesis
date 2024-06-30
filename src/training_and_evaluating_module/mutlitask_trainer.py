from torch.nn.modules import Module
from accelerate.logging import get_logger

from training_and_evaluating_module.ir_trainer import ColBERTwKWMaskTripletTrainer, ColBERTwKWMaskDistillTrainer
from training_and_evaluating_module.evaluate import *
from data_module import DataModule
from data_module.bert_data_module import BertNERDataModule
from data_module.ir_data_module import IRTripletDataModule
from data_module.multitask_datamodule import IRTripletWithNERDataModule, IRTripletWithPOSDataModule
from modeling_module.colbert import ColBERT, ColBERTwKWMask
from modeling_module.loss import *


logger = get_logger(__name__)

class MultitaskNERColBERTwKWMaskTrainer(ColBERTwKWMaskTripletTrainer):
    def __init__(self, args):
        super(MultitaskNERColBERTwKWMaskTrainer, self).__init__(args)
    
    def prepare(self, model: Module, ir_data: IRTripletDataModule, ner_data: BertNERDataModule):
        super().prepare(model, ir_data)
        self.ner_dataloader = self.accelerator.prepare(ner_data.train_dataloader())
        self.ner_dataloader_iter = iter(self.ner_dataloader)

    def fit(self, model: ColBERTwKWMask, ir_data: IRTripletDataModule, ner_data: BertNERDataModule):
        self.prepare(model, ir_data, ner_data)
        first_epoch, progress_bar, global_step = self.setup()
        targets = torch.zeros(self.args.batch_size, dtype=torch.long, device=self.accelerator.device)
        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.model.train()
            train_loss = 0.0
            regularize_loss = 0.0
            guide_loss = 0.0
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
                        loss += kw_regularize_loss

                    # Named entity guide
                    try:
                        ner_batch = next(self.ner_dataloader_iter)
                    except StopIteration:
                        self.ner_dataloader_iter = iter(self.ner_dataloader)
                        ner_batch = next(self.ner_dataloader_iter)
                    _, _, ner_weight_mask = self.model.doc(ner_batch['input_ids'], ner_batch['attention_mask'])
                    ne_guide_loss = ((1 - ner_weight_mask.squeeze())*(ner_batch['labels'] > 0).float()).sum()

                    loss += self.args.distill_coeff * ne_guide_loss

                    # Gather the losses and positive passages scores across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.args.batch_size)).mean()
                    avg_regularize_loss = self.accelerator.gather(kw_regularize_loss.repeat(self.args.batch_size)).mean()
                    avg_guide_loss = self.accelerator.gather(ne_guide_loss.repeat(self.args.batch_size)).mean()
                    avg_passage_score = self.accelerator.gather(positive_scores.repeat(self.args.batch_size)).mean()
                    avg_acc = self.accelerator.gather(acc.repeat(self.args.batch_size)).mean()

                    train_loss += avg_loss.item() / self.args.gradient_accumulation_steps
                    regularize_loss += avg_regularize_loss.item() / self.args.gradient_accumulation_steps
                    guide_loss += avg_guide_loss.item() / self.args.gradient_accumulation_steps
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
                        'guide_loss': guide_loss,
                        'train_passage_score': train_passage_score,
                        'train_acc': train_acc,
                        'grad_norm': grad_norm
                    }, step=global_step)
                    train_loss = 0.0
                    regularize_loss = 0.0
                    guide_loss = 0.0
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

class ColBERTwKWMaskDistillWithNERTrainer(ColBERTwKWMaskDistillTrainer):
    def __init__(self, args):
        super(ColBERTwKWMaskDistillWithNERTrainer, self).__init__(args)

    def prepare(self, model: Module, data: IRTripletWithNERDataModule):
        return super().prepare(model, data)
    
    def fit(self, model: ColBERTwKWMask, data: IRTripletWithNERDataModule):
        self.prepare(model=model, data=data)
        self.model: ColBERTwKWMask

        first_epoch, progress_bar, global_step = self.setup()
        targets = torch.zeros(self.args.batch_size, dtype=torch.long, device=self.accelerator.device)
        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.model.train()
            train_loss, regularize_loss, guide_loss, train_passage_score, train_acc = 0.0, 0.0, 0.0, 0.0, 0.0
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

                    # Distill combine NER guide
                    positive_maxsim_indices = self.get_maxsim_mask_pair(
                        query_embedding=query_embedding,
                        query_mask=query_mask,
                        doc_embedding=passage_embedding,
                        doc_mask=passage_mask
                    )
                    distill_and_ner_guide_loss = mask_distill_loss_with_ner_guide(passage_weight_mask, positive_maxsim_indices, batch['passage']['ner_tags_ids'])
                    if self.args.distill_negative:
                        negative_maxsim_indices = self.get_maxsim_mask_pair(
                            query_embedding=query_embedding,
                            query_mask=query_mask,
                            doc_embedding=neg_passage_embedding,
                            doc_mask=neg_passage_mask
                        )
                        distill_and_ner_guide_loss += mask_distill_loss_with_ner_guide(neg_passage_weight_mask, negative_maxsim_indices, batch['neg_passage']['ner_tags_ids'])
                    
                    # Regularize
                    if self.args.kw_regularize == 'unnormalize':
                        # Sum everything, do not count for passage length
                        kw_regularize_loss = torch.cat([passage_weight_mask.squeeze().sum(-1), neg_passage_weight_mask.squeeze().sum(-1)], dim=0).sum()
                    elif self.args.kw_regularize == 'normalized':
                        pos_kw_regularize_loss = passage_weight_mask.squeeze().sum(-1) / passage_mask.squeeze().sum(-1)
                        neg_kw_regularize_loss = neg_passage_weight_mask.squeeze().sum(-1) / neg_passage_mask.squeeze().sum(-1)
                        kw_regularize_loss = pos_kw_regularize_loss.sum() + neg_kw_regularize_loss.sum()

                    loss = ir_loss \
                        + self.args.distill_coeff * distill_and_ner_guide_loss \
                        + self.cur_regularize_coef * kw_regularize_loss
                    acc = (positive_scores > negative_scores).sum() / positive_scores.size(0)
                    
                    # Gather the losses and positive passages scores across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.args.batch_size)).mean()
                    avg_regularize_loss = self.accelerator.gather(kw_regularize_loss.repeat(self.args.batch_size)).mean()
                    avg_guide_loss = self.accelerator.gather(distill_and_ner_guide_loss.repeat(self.args.batch_size)).mean()
                    avg_passage_score = self.accelerator.gather(positive_scores.repeat(self.args.batch_size)).mean()
                    avg_acc = self.accelerator.gather(acc.repeat(self.args.batch_size)).mean()

                    train_loss += avg_loss.item() / self.args.gradient_accumulation_steps
                    regularize_loss += avg_regularize_loss.item() / self.args.gradient_accumulation_steps
                    guide_loss += avg_guide_loss.item() / self.args.gradient_accumulation_steps
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
                        'guide_loss': guide_loss,
                        'train_passage_score': train_passage_score,
                        'train_acc': train_acc,
                        'grad_norm': grad_norm
                    }, step=global_step)
                    train_loss = 0.0
                    regularize_loss = 0.0
                    guide_loss = 0.0
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


class ColBERTwKWMaskDistillWithPOSTrainer(ColBERTwKWMaskDistillTrainer):
    def __init__(self, args):
        super(ColBERTwKWMaskDistillWithPOSTrainer, self).__init__(args)

    def prepare(self, model: Module, data: IRTripletWithPOSDataModule):
        return super().prepare(model, data)
    
    def fit(self, model: ColBERTwKWMask, data: IRTripletWithPOSDataModule):
        self.prepare(model=model, data=data)
        self.model: ColBERTwKWMask

        first_epoch, progress_bar, global_step = self.setup()
        targets = torch.zeros(self.args.batch_size, dtype=torch.long, device=self.accelerator.device)
        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.model.train()
            train_loss, regularize_loss, guide_loss, train_passage_score, train_acc = 0.0, 0.0, 0.0, 0.0, 0.0
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

                    # Distill combine POS guide
                    positive_maxsim_indices = self.get_maxsim_mask_pair(
                        query_embedding=query_embedding,
                        query_mask=query_mask,
                        doc_embedding=passage_embedding,
                        doc_mask=passage_mask
                    )
                    distill_and_pos_guide_loss = mask_distill_loss_with_pos_guide(passage_weight_mask, positive_maxsim_indices, batch['passage']['pos_tags_ids'])
                    if self.args.distill_negative:
                        negative_maxsim_indices = self.get_maxsim_mask_pair(
                            query_embedding=query_embedding,
                            query_mask=query_mask,
                            doc_embedding=neg_passage_embedding,
                            doc_mask=neg_passage_mask
                        )
                        distill_and_pos_guide_loss += mask_distill_loss_with_pos_guide(neg_passage_weight_mask, negative_maxsim_indices, batch['neg_passage']['pos_tags_ids'])
                    
                    # Regularize
                    if self.args.kw_regularize == 'unnormalize':
                        # Sum everything, do not count for passage length
                        kw_regularize_loss = torch.cat([passage_weight_mask.squeeze().sum(-1), neg_passage_weight_mask.squeeze().sum(-1)], dim=0).sum()
                    elif self.args.kw_regularize == 'normalized':
                        pos_kw_regularize_loss = passage_weight_mask.squeeze().sum(-1) / passage_mask.squeeze().sum(-1)
                        neg_kw_regularize_loss = neg_passage_weight_mask.squeeze().sum(-1) / neg_passage_mask.squeeze().sum(-1)
                        kw_regularize_loss = pos_kw_regularize_loss.sum() + neg_kw_regularize_loss.sum()

                    loss = ir_loss \
                        + self.args.distill_coeff * distill_and_pos_guide_loss \
                        + self.cur_regularize_coef * kw_regularize_loss
                    acc = (positive_scores > negative_scores).sum() / positive_scores.size(0)
                    
                    # Gather the losses and positive passages scores across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.args.batch_size)).mean()
                    avg_regularize_loss = self.accelerator.gather(kw_regularize_loss.repeat(self.args.batch_size)).mean()
                    avg_guide_loss = self.accelerator.gather(distill_and_pos_guide_loss.repeat(self.args.batch_size)).mean()
                    avg_passage_score = self.accelerator.gather(positive_scores.repeat(self.args.batch_size)).mean()
                    avg_acc = self.accelerator.gather(acc.repeat(self.args.batch_size)).mean()

                    train_loss += avg_loss.item() / self.args.gradient_accumulation_steps
                    regularize_loss += avg_regularize_loss.item() / self.args.gradient_accumulation_steps
                    guide_loss += avg_guide_loss.item() / self.args.gradient_accumulation_steps
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
                        'guide_loss': guide_loss,
                        'train_passage_score': train_passage_score,
                        'train_acc': train_acc,
                        'grad_norm': grad_norm
                    }, step=global_step)
                    train_loss = 0.0
                    regularize_loss = 0.0
                    guide_loss = 0.0
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