import os
from typing import List

from torch.utils.data import Dataset
import datasets

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from data_module.utils import VLSP_NER_TAGS, VLSP_NER_TAG_TO_ID, VLSP_POS_TAGS, VLSP_POS_TAG_TO_ID

class MSMarcoDataset(Dataset):
    def __init__(self, args, collection: dict, queries: dict):
        super(MSMarcoDataset, self).__init__()
        self.args = args
        self.collection = collection
        self.queries = queries
        self.tokenized_collection = None
        self.tokenized_queries = None

    def tokenize_data(self, tokenizer):
        if isinstance(tokenizer, tuple) and len(tokenizer) == 2:
            query_tokenizer, passage_tokenizer = tokenizer
        else:
            query_tokenizer, passage_tokenizer = tokenizer, tokenizer
        hf_collection_dataset = datasets.Dataset.from_dict({'pid': self.collection.keys(), 'text': self.collection.values()})
        hf_collection_dataset = hf_collection_dataset.map(
            function=self.tokenize_text,
            batched=True,
            desc=f'Running tokenizer on collection',
            fn_kwargs={'tokenizer': passage_tokenizer}
        )
        hf_queries_dataset = datasets.Dataset.from_dict({'qid': self.queries.keys(), 'text': self.queries.values()})
        hf_queries_dataset = hf_queries_dataset.map(
            function=self.tokenize_text,
            batched=True,
            desc=f'Running tokenizer on queries',
            fn_kwargs={'tokenizer': query_tokenizer}
        )
        self.tokenized_collection = {
            data['pid']: {
                'text': data['text'],
                'input_ids': data['input_ids'],
                'attention_mask': data['attention_mask']
            } for data in hf_collection_dataset
        }
        self.tokenized_queries = {
            data['qid']: {
                'text': data['text'],
                'input_ids': data['input_ids'],
                'attention_mask': data['attention_mask']
            } for data in hf_queries_dataset
        }
        return self.tokenized_collection, self.tokenized_queries

    def tokenize_text(self, examples, tokenizer):
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            tokenized_text = tokenizer(
                examples['text'],
                padding=self.args.padding,
                truncation=self.args.truncation,
                add_special_tokens=self.args.add_special_tokens,
                max_length=self.args.model_max_length
            )
        else:
            # Customized tokenized (ColBERT)
            tokenized_text = tokenizer(examples['text'])
        return tokenized_text
    
    def __len__(self):
        return {
            'colection': len(self.collection),
            'queries': len(self.queries)
        }
    
    def __getitem__(self, index):
        raise Exception('Use get_passage() and get_query() instead!')

    def get_passage(self, pid):
        if self.tokenized_collection is not None:
            return self.tokenized_collection[pid]
        else:
            return self.collection[pid]
    
    def get_query(self, qid):
        if self.tokenized_queries is not None:
            return self.tokenized_queries[qid]
        else:
            return self.queries[qid]
        

class MSMarcoTripletDataset(Dataset):
    def __init__(self, args, triples: List[tuple], msmarco_dataset: MSMarcoDataset):
        super(MSMarcoTripletDataset, self).__init__()
        self.args = args
        self.triples = triples
        self.msmarco_dataset = msmarco_dataset

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        qid, pid, n_pid = self.triples[index]
        return {
            'query': self.msmarco_dataset.get_query(qid),
            'passage': self.msmarco_dataset.get_passage(pid),
            'neg_passage': self.msmarco_dataset.get_passage(n_pid)
        }

class MSMarcoQrelDataset(Dataset):
    def __init__(self, args, qrels: dict, msmarco_dataset: MSMarcoDataset):
        super(MSMarcoQrelDataset, self).__init__()
        # qrel to list
        list_qrel = []
        for qid in qrels:
            list_qrel.extend([(qid, pid) for pid in qrels[qid]])
        self.args = args
        self.qrels = list_qrel
        self.msmarco_dataset = msmarco_dataset

    def __len__(self):
        return len(self.qrels)
    
    def __getitem__(self, index):
        qid, pid = self.qrels[index]
        return {
            'query': self.msmarco_dataset.get_query(qid),
            'passage': self.msmarco_dataset.get_passage(pid)
        }
        
class MSMarcoWithNERDataset(MSMarcoDataset):
    def __init__(self, args, collection: dict, queries: dict, ner_tags: dict):
        super(MSMarcoWithNERDataset, self).__init__(args=args, collection=collection, queries=queries)
        self.ner_tags = ner_tags
        self.data_dict = self.build_multitask_data_dict(collection=collection, ner_tags=ner_tags)

    def build_multitask_data_dict(self, collection: dict, ner_tags: dict):
        # buil data dict for both ir and ner
        assert collection.keys() == ner_tags.keys(), 'Collection and ner tags files are not compatible!'
        all_pids = list(collection.keys())
        all_text = list(collection.values())
        all_ner_tags = [ner_tags[pid] for pid in all_pids]
        return {'pid': all_pids, 'text': all_text, 'ner_tags': all_ner_tags}

    def tokenize_data(self, tokenizer):
        if isinstance(tokenizer, tuple) and len(tokenizer) == 2:
            query_tokenizer, passage_tokenizer = tokenizer
        else:
            query_tokenizer, passage_tokenizer = tokenizer, tokenizer
        
        hf_collection_dataset = datasets.Dataset.from_dict(self.data_dict)
        hf_collection_dataset = hf_collection_dataset.map(
            function=self._tokenize_and_align_labels,
            batched=True,
            desc=f'Running tokenizer and aligning NE tags on collection',
            fn_kwargs={'tokenizer': passage_tokenizer}
        )
        hf_queries_dataset = datasets.Dataset.from_dict({'qid': self.queries.keys(), 'text': self.queries.values()})
        hf_queries_dataset = hf_queries_dataset.map(
            function=self.tokenize_text,
            batched=True,
            desc=f'Running tokenizer on queries',
            fn_kwargs={'tokenizer': query_tokenizer}
        )
        self.tokenized_collection = {
            data['pid']: {
                'text': data['text'],
                'ner_tags': data['ner_tags'],
                'ner_tags_ids': data['ner_tags_ids'],
                'input_ids': data['input_ids'],
                'attention_mask': data['attention_mask']
            } for data in hf_collection_dataset
        }
        self.tokenized_queries = {
            data['qid']: {
                'text': data['text'],
                'input_ids': data['input_ids'],
                'attention_mask': data['attention_mask']
            } for data in hf_queries_dataset
        }
        return self.tokenized_collection, self.tokenized_queries

    def _tokenize_and_align_labels(self, examples, tokenizer):
        if self.args.doc_marker_token is not None:
            examples['ner_tags'] = [[VLSP_NER_TAGS[0]] + ner_tags for ner_tags in examples['ner_tags']]
        tokenized_inputs = tokenizer(examples['text'], truncation=True, add_special_tokens=self.args.add_special_tokens)

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

        tokenized_inputs['ner_tags_ids'] = labels
        return tokenized_inputs
    
    def get_passage(self, pid):
        if self.tokenized_collection is not None:
            return self.tokenized_collection[pid]
        else:
            return self.collection[pid]
    

class MSMarcoWithPOSDataset(MSMarcoDataset):
    def __init__(self, args, collection: dict, queries: dict, pos_tags: dict):
        super(MSMarcoWithPOSDataset, self).__init__(args=args, collection=collection, queries=queries)
        self.pos_tags = pos_tags
        self.data_dict = self.build_multitask_data_dict(collection=collection, pos_tags=pos_tags)

    def build_multitask_data_dict(self, collection: dict, pos_tags: dict):
        # buil data dict for both ir and pos
        assert collection.keys() == pos_tags.keys(), 'Collection and pos tags files are not compatible!'
        all_pids = list(collection.keys())
        all_text = list(collection.values())
        all_pos_tags = [pos_tags[pid] for pid in all_pids]
        return {'pid': all_pids, 'text': all_text, 'pos_tags': all_pos_tags}

    def tokenize_data(self, tokenizer):
        if isinstance(tokenizer, tuple) and len(tokenizer) == 2:
            query_tokenizer, passage_tokenizer = tokenizer
        else:
            query_tokenizer, passage_tokenizer = tokenizer, tokenizer
        
        hf_collection_dataset = datasets.Dataset.from_dict(self.data_dict)
        hf_collection_dataset = hf_collection_dataset.map(
            function=self._tokenize_and_align_labels,
            batched=True,
            desc=f'Running tokenizer and aligning NE tags on collection',
            fn_kwargs={'tokenizer': passage_tokenizer}
        )
        hf_queries_dataset = datasets.Dataset.from_dict({'qid': self.queries.keys(), 'text': self.queries.values()})
        hf_queries_dataset = hf_queries_dataset.map(
            function=self.tokenize_text,
            batched=True,
            desc=f'Running tokenizer on queries',
            fn_kwargs={'tokenizer': query_tokenizer}
        )
        self.tokenized_collection = {
            data['pid']: {
                'text': data['text'],
                'pos_tags': data['pos_tags'],
                'pos_tags_ids': data['pos_tags_ids'],
                'input_ids': data['input_ids'],
                'attention_mask': data['attention_mask']
            } for data in hf_collection_dataset
        }
        self.tokenized_queries = {
            data['qid']: {
                'text': data['text'],
                'input_ids': data['input_ids'],
                'attention_mask': data['attention_mask']
            } for data in hf_queries_dataset
        }
        return self.tokenized_collection, self.tokenized_queries

    def _tokenize_and_align_labels(self, examples, tokenizer):
        def keep_chosen_pos_tags(pos_tag_id):
            if pos_tag_id in self.args.chosen_pos_tags_id: return 1
            else: return 0

        if self.args.doc_marker_token is not None:
            examples['pos_tags'] = [[VLSP_POS_TAGS[0]] + pos_tags for pos_tags in examples['pos_tags']]
        tokenized_inputs = tokenizer(examples['text'], truncation=True, add_special_tokens=self.args.add_special_tokens)

        labels = []
        for i, pos_tags in enumerate(examples['pos_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i) # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(self.args.dummy_label_id)
                elif word_idx != previous_word_idx:
                    label_ids.append(
                        keep_chosen_pos_tags(VLSP_POS_TAG_TO_ID[pos_tags[word_idx]]))
                else:
                    if self.args.pos_first_token: label_ids.append(self.args.dummy_label_id)
                    else: label_ids.append(
                        keep_chosen_pos_tags(VLSP_POS_TAG_TO_ID[pos_tags[word_idx]]))
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs['pos_tags_ids'] = labels
        return tokenized_inputs
    
    def get_passage(self, pid):
        if self.tokenized_collection is not None:
            return self.tokenized_collection[pid]
        else:
            return self.collection[pid]
