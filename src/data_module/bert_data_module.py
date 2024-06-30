import os
import copy
import json
import logging

import torch
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler
)
from transformers import (
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
    default_data_collator
)
from datasets import (
    DatasetDict,
    load_dataset,
    load_from_disk,
    Dataset
)

from data_module import DataModule
from data_module.utils import (
    InputFeatures,
    VLSP_NER_TAGS,
    VLSP_NER_TAG_TO_ID
)
from args import Args
from data_module.collate_fn import collate_fn_mlm_whole_word_masking
    
class BertDataModule(DataModule):
    '''
    Base class for BERT based models 's DataModules
    '''
    def __init__(
            self,
            args: Args,
            tokenizer,
            lazy_eval=False
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.dataset_builded = False
        if not lazy_eval:
            self._build_dataset()
    
    def _build_dataset(self):
        # Always load precessed dataset if possible
        if self.args.processed_data_dir:
            self.dataset_dict = load_from_disk(self.args.processed_data_dir)
            self.train_dataset = self.dataset_dict['train'] if 'train' in self.dataset_dict else None
            self.val_dataset = self.dataset_dict['val'] if 'val' in self.dataset_dict else None
            self.test_dataset = self.dataset_dict['test'] if 'test' in self.dataset_dict else None 
        else:
            # Load data from files
            self.train_dataset = self.load_file_to_dataset(
                input_file_path=self.args.train_data_path,
                mode='train'
            ) if self.args.train_data_path else None
            self.val_dataset = self.load_file_to_dataset(
                input_file_path=self.args.val_data_path,
                mode='val'
            ) if self.args.val_data_path else None
            self.test_dataset = self.load_file_to_dataset(
                input_file_path=self.args.test_data_path,
                mode='test'
            ) if self.args.test_data_path else None
            self.dataset_dict = DatasetDict({
                'train': self.train_dataset,
                'val': self.val_dataset,
                'test': self.test_dataset
            })
        self.dataset_builded = True

class BertNERDataModule(BertDataModule):
    '''
    DataModule for NER task using BERT based models.
    '''
    def __init__(self, args: Args, tokenizer, lazy_eval=False):
        super(BertNERDataModule, self).__init__(args=args, tokenizer=tokenizer, lazy_eval=lazy_eval)

    def _tokenize_and_align_labels(self, examples):
        assert self.tokenizer.is_fast, 'This method is for Fast tokenizer only!'
        tokenized_inputs = self.tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, add_special_tokens=self.args.add_special_tokens)
        
        labels = []
        for i, ner_tags in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i) # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(self.args.dummy_label_id)
                elif word_idx != previous_word_idx:
                    label_ids.append(VLSP_NER_TAG_TO_ID[ner_tags[word_idx]])
                else:
                    if self.args.ner_first_token: label_ids.append(self.args.dummy_label_id)
                    else: label_ids.append(VLSP_NER_TAG_TO_ID[ner_tags[word_idx]])
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    
    def _slow_tokenize_and_align_labels(self):
        pass
        
    def load_file_to_dataset(self, input_file_path, mode):
        # Load dataset
        file_type = os.path.splitext(input_file_path)[1]
        raw_dataset = load_dataset(file_type[1:], data_files=input_file_path)['train']
        # Process dataset
        # Only keeps non-text columns with torch format
        processed_dataset = raw_dataset.map(
            function=self._tokenize_and_align_labels if self.tokenizer.is_fast else self._slow_tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_dataset.column_names,
            desc='Running tokenizer on dataset'
        ).with_format('torch')
        return processed_dataset

    def get_dataloader(self, mode):
        if not self.dataset_builded:
            self._build_dataset()
        data_collator = DataCollatorForTokenClassification(
                tokenizer=self.tokenizer,
                label_pad_token_id=self.args.dummy_label_id,
                padding=True
            )
        if mode == 'train':
            data_source = self.train_dataset 
            sampler = RandomSampler(data_source=data_source)
        elif mode == 'val':
            data_source = self.val_dataset
            sampler = SequentialSampler(data_source=data_source)
        elif mode == 'test':
            data_source = self.test_dataset
            sampler = SequentialSampler(data_source=data_source)
        else:
            raise Exception('Invalid mode!')
        return DataLoader(
            dataset=data_source,
            sampler=sampler,
            batch_size=self.args.batch_size,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers
        )
    

class ColBERTNERDataModule(BertNERDataModule):
    def __init__(self, args: Args, tokenizer,lazy_eval=False):
        super(BertNERDataModule, self).__init__(args=args, tokenizer=tokenizer, lazy_eval=lazy_eval)
    
    def _tokenize_and_align_labels(self, examples):
        assert self.tokenizer.is_fast, 'This method is for Fast tokenizer only!'
        tokenized_inputs = self.tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, add_special_tokens=self.args.add_special_tokens)

        labels = []
        for i, ner_tags in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i) # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            # Add dummy label if use marker
            if self.args.doc_marker_token is not None:
                ner_tags.insert(0, VLSP_NER_TAGS[0])
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(self.args.dummy_label_id)
                elif word_idx != previous_word_idx:
                    label_ids.append(VLSP_NER_TAG_TO_ID[ner_tags[word_idx]])
                else:
                    if self.args.ner_first_token: label_ids.append(self.args.dummy_label_id)
                    else: label_ids.append(VLSP_NER_TAG_TO_ID[ner_tags[word_idx]])
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs['labels'] = labels  
        return tokenized_inputs
    
    def _slow_tokenize_and_align_labels(self, examples):
        assert not self.tokenizer.is_fast, 'This method is for Non fast tokenizer only!'
        def tokenize_sequence(tokenizer, sequence: list[str], add_special_tokens):
            assert isinstance(sequence, list), 'Input must be a list of strings/ words!'
            subwords_sequence = [tokenizer(word, add_special_tokens=False, padding=False, add_marker=False) for word in sequence]
            all_input_ids = sum([subwords['input_ids'] for subwords in subwords_sequence], [])
            all_attention_mask =  sum([subwords['attention_mask'] for subwords in subwords_sequence], [])
            all_word_ids = sum([[i]*len(subwords_sequence[i]['input_ids']) for i in range(len(sequence))], [])
            if self.args.doc_marker_token is not None:
                all_input_ids = [self.tokenizer.marker_token_id] + all_input_ids
                all_attention_mask = [1] + all_attention_mask
                all_word_ids = [None] +  all_word_ids
            if add_special_tokens:
                all_input_ids = [tokenizer.cls_token_id] + all_input_ids + [self.sep_token_id]
                all_attention_mask = [1] + all_attention_mask + [1]
                all_word_ids = [None] + all_word_ids + [None]
            return {'input_ids': all_input_ids, 'attention_mask': all_attention_mask, 'word_ids': all_word_ids}

        # is examples['tokens'] always a list? CHECK LATER
        tokenized_sequences = [tokenize_sequence(self.tokenizer, sequence, add_special_tokens=self.args.add_special_tokens) for sequence in examples['tokens']]
        tokenized_inputs = {}
        tokenized_inputs['input_ids'] = [sequence['input_ids'] for sequence in tokenized_sequences]
        tokenized_inputs['attention_mask'] = [sequence['attention_mask'] for sequence in tokenized_sequences]

        labels = []
        for i, ner_tags in enumerate(examples['ner_tags']):
            word_ids = tokenized_sequences[i]['word_ids'] # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            # Add dummy label if use marker
            if self.args.doc_marker_token is not None:
                ner_tags.insert(0, VLSP_NER_TAGS[0])
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(self.args.dummy_label_id)
                elif word_idx != previous_word_idx:
                    label_ids.append(VLSP_NER_TAG_TO_ID[ner_tags[word_idx]])
                else:
                    if self.args.ner_first_token: label_ids.append(self.args.dummy_label_id)
                    else: label_ids.append(VLSP_NER_TAG_TO_ID[ner_tags[word_idx]])
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs['labels'] = labels  
        return tokenized_inputs

class BertMLMDataModule(BertDataModule):
    '''
    Datamodule to do Bert MLM
    '''
    def __init__(
            self,
            args: Args,
            tokenizer,
            lazy_eval=False
    ):
        super(BertMLMDataModule, self).__init__(args=args, tokenizer=tokenizer, lazy_eval=lazy_eval)

    def _build_dataset(self):
        # Get data collator
        if self.args.mask_whole_word:
            self.data_collator = lambda x: collate_fn_mlm_whole_word_masking(
                batch=x,
                tokenizer=self.tokenizer,
                label_pad_token=self.args.dummy_label_id,
            )
        else:
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm_probability=0.15
            )
        # Always load precessed dataset if possible
        if self.args.processed_data_dir:
            self.dataset_dict = load_from_disk(self.args.processed_data_dir)
            self.train_dataset = self.dataset_dict['train'] if 'train' in self.dataset_dict else None
            self.val_dataset = self.dataset_dict['val'] if 'val' in self.dataset_dict else None
        else:
            # Load data from files
            if self.args.train_data_path and self.args.val_data_path:
                self.train_dataset = self.load_file_to_dataset(input_file_path=self.args.train_data_path)
                self.val_dataset = self.load_file_to_dataset(input_file_path=self.args.val_data_path)
            elif self.args.train_data_path and self.args.val_data_path is None:
                dataset = self.load_file_to_dataset(input_file_path=self.args.train_data_path).train_test_split(test_size=0.1)
                self.train_dataset = dataset['train']
                self.val_dataset = dataset['test']
                if self.args.pre_mask_test:
                    self.val_dataset = self.val_dataset.map(
                        function=self._premask_test_set,
                        batched=True,
                        remove_columns=self.val_dataset.column_names,
                        desc='Premasking validation set'
                    )
                    self.val_dataset = self.val_dataset.rename_columns({
                        'masked_input_ids': 'input_ids',
                        'masked_attention_mask': 'attention_mask',
                        'masked_labels': 'labels'
                    })
            else:
                raise Exception('Train data path must be provided!')   
            self.dataset_dict = DatasetDict({
                'train': self.train_dataset,
                'val': self.val_dataset
            })
        self.dataset_builded = True

    def _premask_test_set(self, batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = self.data_collator(features)
        # Create a new "masked" column for each column in the dataset
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}


    def _group_text(self, examples):
        if self.args.mlm_chunk_size is None:
            chunk_size = self.tokenizer.model_max_length
        else:
            chunk_size = self.args.mlm_chunk_size
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Drop last chunk:
        total_length = (total_length // chunk_size) * chunk_size
        # Split by max_len
        result = {
            k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result['labels'] = result['input_ids'].copy()
        return result

    def _tokenize(self, examples, truncate=False):
        result = self.tokenizer(examples['text'], truncation=truncate)
        result['labels'] = result['input_ids'].copy()
        if self.tokenizer.is_fast:
            result['word_ids'] = [result.word_ids(i) for i in range(len(result['input_ids']))]
        return result

    def load_file_to_dataset(self, input_file_path):
        # Load dataset
        file_type = os.path.splitext(input_file_path)[1]
        assert file_type == '.txt', 'Unsupported data file extension!'

        with open(input_file_path, 'r', encoding='utf8') as f:
            data = [{'text': line} for line in f.readlines()]
        raw_dataset = Dataset.from_list(data)
        if self.args.concat_mlm_chunks:
            tokenized_dataset = raw_dataset.map(
                function=self._tokenize,
                batched=True,
                remove_columns=raw_dataset.column_names,
                desc='Running tokenizer on dataset'
            )
            processed_dataset = tokenized_dataset.map(self._group_text, batched=True)
        else:
            processed_dataset = raw_dataset.map(
                function=lambda x: self._tokenize(x, truncate=True),
                batched=True,
                remove_columns=raw_dataset.column_names,
                desc='Running tokenizer on dataset'
            )
        return processed_dataset
    
    def get_dataloader(self, mode):
        if not self.dataset_builded:
            self._build_dataset()
        if mode == 'train':
            data_source = self.train_dataset 
            sampler = RandomSampler(data_source=data_source)
            return DataLoader(
                dataset=data_source,
                sampler=sampler,
                batch_size=self.args.batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers
            )
        elif mode == 'val':
            data_source = self.val_dataset
            sampler = SequentialSampler(data_source=data_source)
            if self.args.pre_mask_test:
                return DataLoader(
                    dataset=self.val_dataset,
                    sampler=sampler,
                    batch_size=self.args.batch_size,
                    collate_fn=default_data_collator,
                    num_workers=self.args.dataloader_num_workers
                )
            else:
                return DataLoader(
                    dataset=data_source,
                    sampler=sampler,
                    batch_size=self.args.batch_size,
                    collate_fn=self.data_collator,
                    num_workers=self.args.dataloader_num_workers
                )
        
    


