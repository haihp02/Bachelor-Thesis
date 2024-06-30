import os
import math
from tqdm.auto import tqdm

import torch
import transformers
import datasets
from transformers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model

from training_and_evaluating_module import Trainer
from training_and_evaluating_module.evaluate import *
from data_module import DataModule
from modeling_module.colbert import ColBERT
from modeling_module.loss import *
from args import Args

logger = get_logger(__name__)

class BertTrainer(Trainer):
    def __init__(self, args: Args):
        super(BertTrainer, self).__init__(args=args)
        self.args: Args

    def prepare(self, model: torch.nn.Module, data: DataModule):
        self.data = data
        # Preprate everything needed for training and validating
        if self.args.seed is not None:
            set_seed(self.args.seed)
        # Decide to train or freeze BERT weight
        for param in model.bert.parameters():
            param.requires_grad = self.args.train_bert
        # Prepare LoRA model
        if self.args.lora_config is not None:
            lora_config = LoraConfig(**self.args.lora_config)
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        # Prepare accelerator
        accelerator_project_configuration = ProjectConfiguration(
            project_dir=self.args.output_dir,
            logging_dir=os.path.join(self.args.output_dir, 'log')
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            split_batches=self.args.split_batches,
            log_with=self.args.log_with,
            project_config=accelerator_project_configuration
        )
        self.accelerator.init_trackers(
            project_name=self.args.exp_name,
            config={
                'learning_rate': self.args.lr,
                'dropout': self.args.dropout_rate,
                'batch_size': self.args.batch_size,
                'weight_decay': self.args.weight_decay
            },
            init_kwargs={
                self.args.log_with: {
                    'entity': 'trunghainguyenhp02',
                    'name': self.args.run_name
                }
            }
        )
        # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
        # to INFO for the main process only.
        if self.accelerator.is_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
        # Prepare optimizer
        if self.args.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                  "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]    
        optimizer = optimizer_cls(
            params=optimizer_grouped_parameters,
            lr=self.args.lr,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_eps
        )
        # Prepare learning rate scheduler
        lr_scheduler = get_scheduler(
            name=self.args.scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.args.max_train_steps * self.accelerator.num_processes,
        )
        # Gradient checkpoingting or not
        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        self.model, self.train_dataloader, self.val_dataloader, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            model,
            data.train_dataloader(),
            data.val_dataloader(),
            optimizer,
            lr_scheduler
        )

    def setup(self):
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        total_batch_size = self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.data.train_dataset)}")
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
        return first_epoch, progress_bar, global_step
    
    def checkpointing_check(self, global_step):
        # Chekck if its time to checkpoint and perform if it is
        if isinstance(self.args.checkpointing_steps, int):
            if global_step % self.args.checkpointing_steps == 0:
                output_dir = f'step_{global_step}'
                if self.args.output_dir is not None:
                    output_dir = os.path.join(self.args.output_dir, output_dir)
                self.accelerator.save_state(output_dir)

    def resume_from_checkpoint(self):
        if self.args.resume_from_checkpoint != 'latest':
            path = os.path.basename(self.args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(self.args.output_dir)
            dirs = [d for d in dirs if d.startswith('checkpoint')]
            dirs = sorted(dirs, key=lambda x: int(x.split('_')[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            self.accelerator.print(
                f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            self.args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            self.accelerator.print(f"Resuming from checkpoint {path}")
            if self.args.resume_path is not None:
                self.accelerator.load_state(self.args.resume_path, strict=False)
            else:
                self.accelerator.load_state(os.path.join(self.args.output_dir, path), )
            initial_global_step = int(path.split("_")[1])
        return initial_global_step