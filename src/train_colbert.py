from transformers import AutoConfig
from safetensors.torch import load_model

# from data_module.ir_data_module import ColBertIRTripletDataModule
from data_module.ir_data_module import IRTripletDataModule
from modeling_module.colbert import ColBERT
from training_and_evaluating_module.ir_trainer import ColBERTTripletTrainer
from args import Args

# Setup args
args = Args()
args.name_or_path = 'xlm-roberta-base'
args.model_max_length = 512
args.query_max_length = 32
# args.processed_data_dir = '../data/zalo_qa/ms_marco_format/preprocessed'
args.queries_path = '../data/zalo_qa/ms_marco_format/queries.tsv'
args.collection_path = '../data/zalo_qa/ms_marco_format/collection.tsv'
args.train_triples_path = '../data/zalo_qa/ms_marco_format/train_triples.tsv'
args.val_qrels_path = '../data/zalo_qa/ms_marco_format/dev_qrels.tsv'
args.embedding_dim = 128
args.mask_punctuation = True
args.similarity_metric = 'cosine'
args.dataloader_num_workers = 2
args.batch_size = 8
args.add_special_tokens = False
args.query_pad_token = None
args.doc_marker_token = 'Antwort'
args.query_marker_token = 'Frage'
args.seed = 42
args.num_train_epochs = 80
args.max_train_steps = 600_000
args.output_dir = '/kaggle/working/output'
args.checkpointing_steps = 6000
args.mixed_precision = 'fp16'
args.lr_warmup_steps = 6000
args.scheduler = 'linear'
args.gradient_accumulation_steps = 4
args.use_8bit_adam = True
args.split_batches = False
args.lr = 2e-6
args.resume_from_checkpoint = None
args.resume_path = None
args.max_grad_norm = 1
args.dropout_rate = 0.05
args.ir_loss_type = 'online_contrastive_loss'
args.lora_config = None
args.log_with = 'wandb'
args.exp_name = 'qazalo_ir'
args.run_name = 'harder_triples'

model_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=args.name_or_path,
    output_hidden_states=True
)
model = ColBERT(args=args, config=model_config)
load_model(model, '/kaggle/input/colbert-best-ckpt/best-fintuned-model/model.safetensors', strict=False)
data_module = IRTripletDataModule(
    args=args,
    tokenizer=(model.query_tokenizer, model.doc_tokenizer)
)
trainer = ColBERTTripletTrainer(args=args)

trainer.fit(model=model, data=data_module)