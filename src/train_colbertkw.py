from transformers import AutoConfig

from modeling_module.utils import *
from args import Args
from modeling_module.colbert import ColBERTwKWMask
from data_module.ir_data_module import IRTripletDataModule
from training_and_evaluating_module.ir_trainer import ColBERTwKWMaskDistillTrainer

args = Args()
args.name_or_path = 'vinai/phobert-base-v2'
args.model_max_length = 256
args.query_max_length = 32
# args.processed_data_dir = '/kaggle/input/processed-zaloqa-ir/processed_data'
args.queries_path = '../data/zalo_qa/ms_marco_format/queries_ws.tsv'
args.collection_path = '../data/zalo_qa/ms_marco_format/collection_ws.tsv'
args.train_triples_path = '../data/zalo_qa/ms_marco_format/train_triples.tsv'
args.val_qrels_path = '../data/zalo_qa/ms_marco_format/dev_qrels.tsv'
args.embedding_dim = 128
args.mask_punctuation = True
args.similarity_metric = 'cosine'
args.dataloader_num_workers = 2
args.batch_size = 64
args.add_special_tokens = False
args.query_pad_token = None
args.doc_marker_token = 'Đáp'
args.query_marker_token = 'Hỏi'
args.seed = 42
args.num_train_epochs = 10
args.max_train_steps = 500_000
args.output_dir = '../output'
args.checkpointing_steps = 12000
args.mixed_precision = 'fp16'
args.lr_warmup_steps = 12000
args.scheduler = 'linear'
args.gradient_accumulation_steps = 1
args.use_8bit_adam = True
args.split_batches = False
args.lr = 6e-6
args.resume_from_checkpoint = None
args.resume_path = None
args.max_grad_norm = 1
args.dropout_rate = 0.05
args.ir_loss_type = 'online_contrastive_loss'
args.kw_regularize = 'normalized'
args.distill_coeff = 0.1
args.regularize_coef = 0.5
args.kw_warmup_steps = 1
# args.kw_threshold = 0.5
args.weight_mask = -1
args.train_bert = False
args.log_with = 'wandb'
args.exp_name = 'qazalo_ir_kw_distill'
args.run_name = 'from_best_colbert_regularize'

model_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=args.name_or_path,
    output_hidden_states=True
)
model = ColBERTwKWMask.from_pretrained('/kaggle/input/colbert-best-ckpt/best-fintuned-model', args=args, config=model_config)
data_module = IRTripletDataModule(
    args=args,
    tokenizer=(model.query_tokenizer, model.doc_tokenizer)
)
trainer = ColBERTwKWMaskDistillTrainer(args=args)

trainer.fit(model=model, data=data_module)