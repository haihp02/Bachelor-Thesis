from safetensors.torch import load_model

from args import Args
from modeling_module.colbert import ColBERTwKWMask
from modeling_module.tokenizer import SlowTokenizerWrapper
from transformers import AutoConfig
from data_module.bert_data_module import ColBERTNERDataModule
from data_module.multitask_datamodule import IRTripletWithNERDataModule
from training_and_evaluating_module.mutlitask_trainer import ColBERTwKWMaskDistillWithNERTrainer

args = Args()
args.name_or_path = 'vinai/phobert-base-v2'
args.model_max_length = 256
args.query_max_length = 32
# args.processed_data_dir = '/kaggle/input/processed-zaloqa-ir/processed_data'
args.queries_path = '../data/zalo_qa/ms_marco_format/queries_ws.tsv'
args.collection_path = '../data/zalo_qa/ms_marco_format/collection_ws.tsv'
args.train_triples_path = '../data/zalo_qa/ms_marco_format/train_triples.tsv'
args.val_qrels_path = '../data/zalo_qa/ms_marco_format/dev_qrels.tsv'
args.train_data_path = '../data/zalo_qa/ms_marco_format/collection_ws_ner_tags.json'
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
args.lora_config = None
args.kw_regularize = 'normalized'
args.distill_coeff = 0.4
args.regularize_coef = 0.11
args.kw_warmup_steps = 1
args.weight_mask = -1
args.train_bert = False
args.train_colbert_embedding_head = False
args.distill_negative = True
args.log_with = 'wandb'
args.exp_name = 'qazalo_ir_kw_distill_with_ner_guide'
args.run_name = 'from_best_colbert_phobert_regularize_freeze_embedding_head_l2'

model_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=args.name_or_path,
    output_hidden_states=True
)
model = ColBERTwKWMask(args=args, config=model_config)
load_model(model, '/kaggle/input/colbert-kw-distill-ner/step_96000/model.safetensors', strict=False)
model.doc_tokenizer.raw_tokenizer = SlowTokenizerWrapper(model.doc_tokenizer.raw_tokenizer)
data_module = IRTripletWithNERDataModule(args=args, tokenizer=(model.query_tokenizer, model.doc_tokenizer))
trainer = ColBERTwKWMaskDistillWithNERTrainer(args)

trainer.fit(model=model, data=data_module)