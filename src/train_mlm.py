from transformers import AutoModelForMaskedLM, AutoTokenizer

from data_module.bert_data_module import BertMLMDataModule
from training_and_evaluating_module.mlm_trainer import BertMLMTrainer 
from args import Args

# Setup args
args = Args()
# args.processed_data_dir = '../data/zalo_qa/processed_mlm/'
args.train_data_path = '../data/zalo_qa/all_passages_line_by_line.txt'
args.val_data_path = None
args.mlm_chunk_size = None
args.concat_mlm_chunks = True
args.mask_whole_word = True
args.batch_size = 4
args.output_dir = '../output/'
args.seed = 42
args.num_train_epochs = 5
args.max_train_steps = 1_000_000
args.checkpointing_steps = 10
args.mixed_precision = 'fp16'
args.lr_warmup_steps = 1000
args.scheduler = 'linear'
args.scheduler = 'linear'
args.gradient_accumulation_steps = 2
args.split_batches = False
args.lr = 3e-6
args.pre_mask_test = True
args.log_with = 'wandb'
args.exp_name = 'test_qazalo_mlm'

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base')
data_module = BertMLMDataModule(args=args, tokenizer=tokenizer)
trainer = BertMLMTrainer(args=args)
trainer.fit(model=model, data=data_module)