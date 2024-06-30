from collections import defaultdict

import numpy as np
import torch
from transformers import default_data_collator


def collate_fn_mlm_whole_word_masking(batch, tokenizer, label_pad_token=-100):
    for example in batch:
        word_ids = example.pop('word_ids')

        # Create a map between words and corresponding token indices
        mapping = defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)
        # Randomly mask words
        mask = np.random.binomial(n=1, p=0.2, size=(len(mapping), ))
        input_ids = example['input_ids']
        labels = example['labels']
        new_labels = [label_pad_token] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        example['labels'] = new_labels
    return default_data_collator(batch)

def colbert_collate_fn_ir(batch, tokenizer):
    if isinstance(tokenizer, tuple):
        query_tokenizer, passage_tokenizer = tokenizer
    else:
        query_tokenizer, passage_tokenizer = tokenizer, tokenizer

    collected_features = {
        'query': {
            'input_ids': [_['query_input_ids'] for _ in batch],
            'attention_mask': [_['query_attention_mask'] for _ in batch]
        },
        'passage': {
            'input_ids': [_['passage_input_ids'] for _ in batch],
            'attention_mask': [_['passage_attention_mask'] for _ in batch]
        }
    }
    if 'negative_passage_input_ids' in batch[0].keys():
        collected_features['negative_passage'] = {
            'input_ids': [_['negative_passage_input_ids'] for _ in batch],
            'attention_mask': [_['negative_passage_attention_mask'] for _ in batch]
        }

    for key in collected_features.keys():
        if key == 'query':
            collected_features[key] = query_tokenizer.pad(collected_features[key], padding=True, return_tensors='pt')
        elif key == 'passage' or key == 'negative_passage':
            collected_features[key] = passage_tokenizer.pad(collected_features[key], padding=True, return_tensors='pt')
    return collected_features

def collate_fn_ir(batch, tokenizer):
    if isinstance(tokenizer, tuple):
        query_tokenizer, passage_tokenizer = tokenizer
    else:
        query_tokenizer, passage_tokenizer = tokenizer, tokenizer

    collected_features = {
        k: {
            'input_ids': [example[k]['input_ids'] for example in batch],
            'attention_mask': [example[k]['attention_mask'] for example in batch]
        } for k in batch[0].keys()
    }

    for key in collected_features.keys():
        if key == 'query':
            collected_features[key] = query_tokenizer.pad(collected_features[key], padding=True, return_tensors='pt')
        elif key == 'passage' or key == 'neg_passage':
            collected_features[key] = passage_tokenizer.pad(collected_features[key], padding=True, return_tensors='pt')
    return collected_features

def collate_fn_ir_ner(batch, tokenizer, dummy_label_id: int):
    all_passage_ner_tags = [example['passage']['ner_tags_ids'] for example in batch]
    all_neg_passage_ner_tags = [example['neg_passage']['ner_tags_ids'] for example in batch]

    collected_features = collate_fn_ir(batch, tokenizer)
    passage_sequence_length = collected_features['passage']['input_ids'].size(1)
    neg_passage_sequence_length = collected_features['neg_passage']['input_ids'].size(1)

    all_passage_ner_tags = [
        ner_tags + [dummy_label_id]*(passage_sequence_length - len(ner_tags)) for ner_tags in all_passage_ner_tags
    ]
    all_neg_passage_ner_tags = [
        ner_tags + [dummy_label_id]*(neg_passage_sequence_length - len(ner_tags)) for ner_tags in all_neg_passage_ner_tags
    ]

    collected_features['passage']['ner_tags_ids'] = torch.tensor(all_passage_ner_tags, dtype=torch.long)
    collected_features['neg_passage']['ner_tags_ids'] = torch.tensor(all_neg_passage_ner_tags, dtype=torch.long)

    return collected_features

def collate_fn_ir_pos(batch, tokenizer, dummy_label_id: int):
    all_passage_pos_tags = [example['passage']['pos_tags_ids'] for example in batch]
    all_neg_passage_pos_tags = [example['neg_passage']['pos_tags_ids'] for example in batch]

    collected_features = collate_fn_ir(batch, tokenizer)
    passage_sequence_length = collected_features['passage']['input_ids'].size(1)
    neg_passage_sequence_length = collected_features['neg_passage']['input_ids'].size(1)

    all_passage_pos_tags = [
        pos_tags + [dummy_label_id]*(passage_sequence_length - len(pos_tags)) for pos_tags in all_passage_pos_tags
    ]
    all_neg_passage_pos_tags = [
        pos_tags + [dummy_label_id]*(neg_passage_sequence_length - len(pos_tags)) for pos_tags in all_neg_passage_pos_tags
    ]

    collected_features['passage']['pos_tags_ids'] = torch.tensor(all_passage_pos_tags, dtype=torch.long)
    collected_features['neg_passage']['pos_tags_ids'] = torch.tensor(all_neg_passage_pos_tags, dtype=torch.long)

    return collected_features
