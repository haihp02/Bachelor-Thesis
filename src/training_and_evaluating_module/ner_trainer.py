import math
from tqdm.auto import tqdm

from torch.nn.modules import Module
from accelerate.logging import get_logger

from training_and_evaluating_module.trainer import BertTrainer
from training_and_evaluating_module.evaluate import *
from data_module import DataModule
from modeling_module.loss import *

logger = get_logger(__name__)

class BertNERTrainer(BertTrainer):
    def __init__(self, args):
        super(BertNERTrainer, self).__init__(args=args)

    def prepare(self, model: Module, data: DataModule):
        super().prepare(model, data)
        self.metric = evaluate.load('seqeval')

    def fit(self, model, data):
        self.prepare(model=model, data=data)
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        total_batch_size = self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(data.train_dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")

        first_epoch = 0
        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            global_step = self.resume_from_checkpoint()
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            global_step = 0

        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=global_step,
            desc='Steps',
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )
        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.model.train()
            train_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    ner_logits = self.model(batch['input_ids'], batch['attention_mask'])
                    loss = bert_ner_cross_entroly_loss(
                        logits=ner_logits,
                        targets=batch['labels'],
                        ignore_index=self.args.dummy_label_id
                    )
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.args.batch_size)).mean()
                    train_loss += avg_loss.item() / self.args.gradient_accumulation_steps

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
                    self.accelerator.log({'train_loss': train_loss, 'grad_norm': grad_norm}, step=global_step)
                    train_loss = 0.0

                    if isinstance(self.args.checkpointing_steps, int):
                        if global_step % self.args.checkpointing_steps == 0:
                            output_dir = f'step_{global_step}'
                            if self.args.output_dir is not None:
                                output_dir = os.path.join(self.args.output_dir, output_dir)
                            self.accelerator.save_state(output_dir)
                    if global_step >= self.args.max_train_steps:
                        break
                
            self.model.eval()
            samples_seen = 0
            for step, batch in enumerate(self.val_dataloader):
                with torch.no_grad():
                    ner_logits = self.model(batch['input_ids'], batch['attention_mask'])
                predictions = torch.max(ner_logits, 2)[1]
                labels = batch['labels']
                # Pad preditions and labels for being gathered
                predictions = self.accelerator.pad_across_processes(predictions, dim=1, pad_index=self.args.dummy_label_id)
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=self.args.dummy_label_id)
                predictions_gathered, labels_gathered = self.accelerator.gather((predictions, labels))
                # If we are in a multiprocess environment, the last batch has duplicates
                if self.accelerator.num_processes > 1:
                    if step == len(self.val_dataloader) - 1:
                        predictions_gathered = predictions_gathered[: len(self.val_dataloader.dataset) - samples_seen]
                        labels_gathered = labels_gathered[: len(self.val_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += labels_gathered.shape[0]
                preds, refs = get_ner_labels(predictions_gathered, labels_gathered, self.args)
                self.metric.add_batch(
                    predictions=preds,
                    references=refs,
                )  # predictions and preferences are expected to be a nested list of labels, not label_ids
            
            eval_metric = compute_metrics_ner(metric=self.metric)
            self.accelerator.print(f'epoch {epoch}:', eval_metric)
            self.accelerator.log(eval_metric)
        self.accelerator.end_training()