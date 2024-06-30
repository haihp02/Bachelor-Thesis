from torch.utils.data import (
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler
)
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
    load_from_disk
)
from data_module import DataModule
from data_module.data_processor import IRTripletProcessor
from data_module.collate_fn import *
from data_module.dataset import *

class IRTripletDataModule(DataModule):

    def __init__(
            self,
            args,
            tokenizer,
            data_processor_cls=IRTripletProcessor,
            lazy_eval=False
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.dataset_builded = False
        self.processor = data_processor_cls(args=args)
        if not lazy_eval:
            self._build_dataset()

    def _build_dataset(self):
        if self.args.processed_data_dir:
            self.dataset_dict = load_from_disk(self.args.processed_data_dir)
            self.msmarco_dataset = self.dataset_dict['msmarco'] if 'msmarco' in self.dataset_dict else None
            self.train_dataset = self.dataset_dict['train'] if 'train' in self.dataset_dict else None
            self.val_dataset = self.dataset_dict['val'] if 'val' in self.dataset_dict else None
        else:
            collection, queries, train_triples, val_qrels = self.processor.load_data()
            self.msmarco_dataset = MSMarcoDataset(args=self.args, collection=collection, queries=queries)
            self.msmarco_dataset.tokenize_data(tokenizer=self.tokenizer)
            self.train_dataset = MSMarcoTripletDataset(args=self.args, triples=train_triples, msmarco_dataset=self.msmarco_dataset)
            self.val_dataset = MSMarcoQrelDataset(args=self.args, qrels=val_qrels, msmarco_dataset=self.msmarco_dataset)
            self.dataset_dict = DatasetDict({
                    'msmarco': self.msmarco_dataset,
                    'train': self.train_dataset,
                    'val': self.val_dataset
                })
        self.dataset_builded = True
    
    def get_dataloader(self, mode):
        if not self.dataset_builded:
            self._build_dataset()
        if mode == 'train':
            data_source = self.train_dataset
            sampler = RandomSampler(data_source=data_source)
        elif mode =='val':
            data_source = self.val_dataset
            sampler = SequentialSampler(data_source=data_source)
        return DataLoader(
            dataset=data_source,
            sampler=sampler,
            batch_size=self.args.batch_size,
            collate_fn=lambda batch: collate_fn_ir(batch, tokenizer=self.tokenizer),
            num_workers=self.args.dataloader_num_workers,
            drop_last=True
        )

# class ColBertIRTripletDataModule(IRTripletDataModule):
#     '''
#     Base class for information retrieval triplet DataModules
#     Each training example is a triple of (query, positive_passage, negative_passage)
#     Each val example is a pair (query, positive_passage)
#     No test set
#     '''
#     def __init__(
#             self,
#             args,
#             passage_tokenizer,
#             query_tokenizer,
#             data_processor_cls=IRTripletProcessor,
#             lazy_eval=False
#     ):
#         self.args = args
#         self.passage_tokenizer = passage_tokenizer
#         self.query_tokenizer = query_tokenizer
#         self.dataset_builded = False
#         self.processor = data_processor_cls(args=args)
#         if not lazy_eval:
#             self._build_dataset()

#     def _build_dataset(self):
#         # Always load precessed dataset if possible
#         if self.args.processed_data_dir:
#             self.dataset_dict = load_from_disk(self.args.processed_data_dir)
#             self.train_dataset = self.dataset_dict['train'] if 'train' in self.dataset_dict else None
#             self.val_dataset = self.dataset_dict['val'] if 'val' in self.dataset_dict else None
#         else:
#             # Preprocess to get text triples from raw dataset
#             self.processor.load_data()
#             self.train_triples = self.processor.get_examples(mode='train')
#             self.val_pairs = self.processor.get_examples(mode='val')

#             self.train_dataset = self.raw_text_to_dataset(
#                 mode='train', raw_text=self.train_triples, tokenize_func=self._tokenize_triples
#             )
#             self.val_dataset = self.raw_text_to_dataset(
#                 mode='val', raw_text=self.val_pairs, tokenize_func=self._tokenize_pairs
#             )
#             self.dataset_dict = datasets.DatasetDict({
#                 'train': self.train_dataset,
#                 'val': self.val_dataset
#             })
#         self.dataset_builded = True

#     def raw_text_to_dataset(self, mode, raw_text, tokenize_func):
#         raw_dataset = datasets.Dataset.from_list(raw_text)
#         processed_dataset = raw_dataset.map(
#             function=tokenize_func,
#             batched=True,
#             remove_columns=raw_dataset.column_names,
#             desc=f'Running tokenizer on {mode} dataset'
#         ).with_format('torch')
#         return processed_dataset

#     def _tokenize_query(self, query):
#         return self.query_tokenizer(query)

#     def _tokenize_passage(self, passage):
#         return self.passage_tokenizer(passage)

#     def _tokenize_triples(self, examples):
#         tokenized_query = self._tokenize_query(examples['query'])
#         tokenized_passage = self._tokenize_passage(examples['passage'])
#         tokenized_negative_passage = self._tokenize_passage(examples['negative_passage'])

#         return {
#             'query_input_ids': tokenized_query['input_ids'],
#             'query_attention_mask': tokenized_query['attention_mask'],
#             'passage_input_ids': tokenized_passage['input_ids'],
#             'passage_attention_mask': tokenized_passage['attention_mask'],
#             'negative_passage_input_ids': tokenized_negative_passage['input_ids'],
#             'negative_passage_attention_mask': tokenized_negative_passage['attention_mask'],
#         }
    
#     def _tokenize_pairs(self, examples):
#         tokenized_query = self._tokenize_query(examples['query'])
#         tokenized_passage = self._tokenize_passage(examples['passage'])

#         return {
#             'query_input_ids': tokenized_query['input_ids'],
#             'query_attention_mask': tokenized_query['attention_mask'],
#             'passage_input_ids': tokenized_passage['input_ids'],
#             'passage_attention_mask': tokenized_passage['attention_mask']
#         }

#     def get_dataloader(self, mode):
#         if not self.dataset_builded:
#             self._build_dataset()
#         if mode == 'train':
#             data_source = self.train_dataset
#             sampler = RandomSampler(data_source=data_source)
#         elif mode =='val':
#             data_source = self.val_dataset
#             sampler = SequentialSampler(data_source=data_source)
#         return DataLoader(
#             dataset=data_source,
#             sampler=sampler,
#             batch_size=self.args.batch_size,
#             collate_fn=lambda batch: colbert_collate_fn_ir(batch, tokenizer=(self.query_tokenizer, self.passage_tokenizer)),
#             num_workers=self.args.dataloader_num_workers,
#             drop_last=True
#         )
        

# class COILIRTripletDataModule(IRTripletDataModule):

#     def __init__(
#             self,
#             args,
#             tokenizer,
#             data_processor_cls=IRTripletProcessor,
#             lazy_eval=False
#     ):
#         self.args = args
#         self.tokenizer = tokenizer
#         self.dataset_builded = False
#         self.processor = data_processor_cls(args=args)
#         if not lazy_eval:
#             self._build_dataset()

#     def _build_dataset(self):
#         if self.args.processed_data_dir:
#             self.dataset_dict = load_from_disk(self.args.processed_data_dir)
#             self.msmarco_dataset = self.dataset_dict['msmarco'] if 'msmarco' in self.dataset_dict else None
#             self.train_dataset = self.dataset_dict['train'] if 'train' in self.dataset_dict else None
#             self.val_dataset = self.dataset_dict['val'] if 'val' in self.dataset_dict else None
#         else:
#             collection, queries, train_triples, val_qrels = self.processor.load_data()
#             self.msmarco_dataset = MSMarcoDataset(args=self.args, collection=collection, queries=queries)
#             self.msmarco_dataset.tokenize_data(tokenizer=self.tokenizer)
#             self.train_dataset = MSMarcoTripletDataset(args=self.args, triples=train_triples, msmarco_dataset=self.msmarco_dataset)
#             self.val_dataset = MSMarcoQrelDataset(args=self.args, qrels=val_qrels, msmarco_dataset=self.msmarco_dataset)
#             self.dataset_dict = DatasetDict({
#                     'msmarco': self.msmarco_dataset,
#                     'train': self.train_dataset,
#                     'val': self.val_dataset
#                 })
#         self.dataset_builded = True
    
#     def get_dataloader(self, mode):
#         if not self.dataset_builded:
#             self._build_dataset()
#         if mode == 'train':
#             data_source = self.train_dataset
#             sampler = RandomSampler(data_source=data_source)
#         elif mode =='val':
#             data_source = self.val_dataset
#             sampler = SequentialSampler(data_source=data_source)
#         return DataLoader(
#             dataset=data_source,
#             sampler=sampler,
#             batch_size=self.args.batch_size,
#             collate_fn=lambda batch: collate_fn_ir(batch, tokenizer=self.tokenizer),
#             num_workers=self.args.dataloader_num_workers,
#             drop_last=True
#         )

# class CITADELIRTripletDataModule(DataModule):
    
#     def __init__(
#             self,
#             args,
#             tokenizer,
#             data_processor_cls=IRTripletProcessor,
#             lazy_eval=False
#     ):
#         self.args = args
#         self.tokenizer = tokenizer
#         self.dataset_builded = False
#         self.processor = data_processor_cls(args=args)
#         if not lazy_eval:
#             self._build_dataset()

#     def _build_dataset(self):
#         if self.args.processed_data_dir:
#             self.dataset_dict = load_from_disk(self.args.processed_data_dir)
#             self.msmarco_dataset = self.dataset_dict['msmarco'] if 'msmarco' in self.dataset_dict else None
#             self.train_dataset = self.dataset_dict['train'] if 'train' in self.dataset_dict else None
#             self.val_dataset = self.dataset_dict['val'] if 'val' in self.dataset_dict else None
#         else:
#             collection, queries, train_triples, val_qrels = self.processor.load_data()
#             self.msmarco_dataset = MSMarcoDataset(args=self.args, collection=collection, queries=queries)
#             self.msmarco_dataset.tokenize_data(tokenizer=self.tokenizer)
#             self.train_dataset = MSMarcoTripletDataset(args=self.args, triples=train_triples, msmarco_dataset=self.msmarco_dataset)
#             self.val_dataset = MSMarcoQrelDataset(args=self.args, qrels=val_qrels, msmarco_dataset=self.msmarco_dataset)
#             self.dataset_dict = DatasetDict({
#                     'msmarco': self.msmarco_dataset,
#                     'train': self.train_dataset,
#                     'val': self.val_dataset
#                 })
#         self.dataset_builded = True
        
#     def get_dataloader(self, mode):
#         if not self.dataset_builded:
#             self._build_dataset()
#         if mode =='train':
#             data_source = self.train_dataset
#             sampler = RandomSampler(data_source=data_source)
#         elif mode == 'val':
#             data_source = self.val_dataset
#             sampler = SequentialSampler(data_source=data_source)
#         return DataLoader(
#             dataset=data_source,
#             sampler=sampler,
#             batch_size=self.args.batch_size,
#             collate_fn=lambda batch: collate_fn_ir(batch, tokenizer=self.tokenizer),
#             num_workers=self.args.dataloader_num_workers,
#             drop_last=True
#         )