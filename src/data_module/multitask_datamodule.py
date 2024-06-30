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
from data_module.collate_fn import *
from data_module.dataset import *
from data_module.data_processor import IRTripletWithNERProcessor, IRTripletWithPOSProcessor
from args import Args

class IRTripletWithNERDataModule(DataModule):
    
    def __init__(
            self,
            args: Args,
            tokenizer,
            data_processor_cls=IRTripletWithNERProcessor,
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
            self.msmarco_dataset = self.dataset_dict['msmarco_w_ner'] if 'msmarco_w_ner' in self.dataset_dict else None
            self.train_dataset = self.dataset_dict['train'] if 'train' in self.dataset_dict else None
            self.val_dataset = self.dataset_dict['val'] if 'val' in self.dataset_dict else None
        else:
            collection, queries, train_triples, val_qrels, ner_tags = self.processor.load_data()
            self.msmarco_dataset = MSMarcoWithNERDataset(args=self.args, collection=collection, queries=queries, ner_tags=ner_tags)
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
            collate_fn = lambda batch: collate_fn_ir_ner(batch, tokenizer=self.tokenizer, dummy_label_id=self.args.dummy_label_id)
        elif mode =='val':
            data_source = self.val_dataset
            sampler = SequentialSampler(data_source=data_source)
            collate_fn = lambda batch: collate_fn_ir(batch, tokenizer=self.tokenizer)
        return DataLoader(
            dataset=data_source,
            sampler=sampler,
            batch_size=self.args.batch_size,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True
        )


class IRTripletWithPOSDataModule(DataModule):
    
    def __init__(
            self,
            args: Args,
            tokenizer,
            data_processor_cls=IRTripletWithPOSProcessor,
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
            self.msmarco_dataset = self.dataset_dict['msmarco_w_pos'] if 'msmarco_w_pos' in self.dataset_dict else None
            self.train_dataset = self.dataset_dict['train'] if 'train' in self.dataset_dict else None
            self.val_dataset = self.dataset_dict['val'] if 'val' in self.dataset_dict else None
        else:
            collection, queries, train_triples, val_qrels, pos_tags = self.processor.load_data()
            self.msmarco_dataset = MSMarcoWithPOSDataset(args=self.args, collection=collection, queries=queries, pos_tags=pos_tags)
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
            collate_fn = lambda batch: collate_fn_ir_pos(batch, tokenizer=self.tokenizer, dummy_label_id=self.args.dummy_label_id)
        elif mode =='val':
            data_source = self.val_dataset
            sampler = SequentialSampler(data_source=data_source)
            collate_fn = lambda batch: collate_fn_ir(batch, tokenizer=self.tokenizer)
        return DataLoader(
            dataset=data_source,
            sampler=sampler,
            batch_size=self.args.batch_size,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True
        )
