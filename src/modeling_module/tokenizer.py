from typing import Any, Union
import itertools

from transformers import (
    AutoTokenizer, 
    PreTrainedTokenizer, 
    PreTrainedTokenizerFast
)
from transformers.tokenization_utils import BatchEncoding
from transformers import PhobertTokenizer

from args import Args

class QueryTokenizer():
    def __init__(self, args: Args, **kwargs):
        self.args = args
        self.raw_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        if args.query_pad_token is not None:
            self.raw_tokenizer = AutoTokenizer.from_pretrained(self.args.name_or_path, pad_token=self.args.query_pad_token, **kwargs)
        else:
            self.raw_tokenizer = AutoTokenizer.from_pretrained(self.args.name_or_path, **kwargs)
        if self.raw_tokenizer.model_max_length > 999999:
            self.raw_tokenizer.model_max_length = 256
        if self.args.query_marker_token is not None:
            self.marker_token_id = self.raw_tokenizer.convert_tokens_to_ids(self.args.query_marker_token)

    def __call__(self, batch_text: Union[str, list[str], list[list[str]]], **kwargs):
        if self.args.query_pad_token:
            padding = 'max_length'
            truncation = True
            max_length = self.args.query_max_length
        else:
            padding = kwargs.pop('padding', False)
            truncation = kwargs.pop('truncation', True)
            max_length = kwargs.pop('max_length', self.raw_tokenizer.model_max_length)
        add_special_tokens = kwargs.pop('add_special_tokens', self.args.add_special_tokens)
        is_split_into_words = kwargs.pop('is_split_into_words', False)

        add_marker = kwargs.pop('add_marker', True)
        if add_marker and self.args.doc_marker_token is not None:
            # Prepend
            if isinstance(batch_text, str):
                batch_text = f'{self.args.doc_marker_token} {batch_text}'
            elif isinstance(batch_text, list):
                if not is_split_into_words:
                    batch_text = [f'{self.args.doc_marker_token} {text}' for text in batch_text]
                else:
                    batch_text = [[self.args.doc_marker_token] + token_list for token_list in batch_text]

        kwargs['padding'] = padding
        kwargs['add_special_tokens'] = add_special_tokens
        kwargs['truncation'] = truncation
        kwargs['is_split_into_words'] = is_split_into_words
        kwargs['max_length'] = max_length
        return self.raw_tokenizer(batch_text, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # If can't find attribute in class, look for it from raw_tokenizer
        return getattr(self.raw_tokenizer, name)
    
class DocTokenizer():
    def __init__(self, args: Args, **kwargs):
        self.args = args
        self.raw_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        self.raw_tokenizer = AutoTokenizer.from_pretrained(self.args.name_or_path, **kwargs)
        if self.raw_tokenizer.model_max_length > 999999:
            self.raw_tokenizer.model_max_length = 256
        if self.args.doc_marker_token is not None:
            self.marker_token_id = self.raw_tokenizer.convert_tokens_to_ids(self.args.doc_marker_token)

    def __call__(self, batch_text: Union[str, list[str], list[list[str]]], **kwargs):
        padding = kwargs.pop('padding', False)
        add_special_tokens = kwargs.pop('add_special_tokens', self.args.add_special_tokens)
        truncation = kwargs.pop('truncation', True)
        is_split_into_words = kwargs.pop('is_split_into_words', False)

        add_marker = kwargs.pop('add_marker', True)
        if add_marker and self.args.doc_marker_token is not None:
            # Prepend
            if isinstance(batch_text, str):
                batch_text = f'{self.args.doc_marker_token} {batch_text}'
            elif isinstance(batch_text, list):
                if not is_split_into_words:
                    batch_text = [f'{self.args.doc_marker_token} {text}' for text in batch_text]
                else:
                    batch_text = [[self.args.doc_marker_token] + token_list for token_list in batch_text]
        
        kwargs['padding'] = padding
        kwargs['add_special_tokens'] = add_special_tokens
        kwargs['truncation'] = truncation
        kwargs['is_split_into_words'] = is_split_into_words
        return self.raw_tokenizer(batch_text, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        # If can't find attribute in class, look for it from raw_tokenizer
        return getattr(self.raw_tokenizer, name)
    

class SlowTokenizerWrapper():
    def __init__(self, tokenizer: PreTrainedTokenizer):
        assert not tokenizer.is_fast, 'This wrapper is only for slow tokenizer!'
        self.tokenizer = tokenizer

    def sequence_to_ids(self, sequence: list[str], add_special_tokens):
        # Make sure to split sequence by space before input to the this
        input_ids = []
        word_ids = []
        for i, word in enumerate(sequence):
            subword_tokens_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))
            if isinstance(subword_tokens_ids, int):
                input_ids.append(subword_tokens_ids)
                word_ids.append(i)
            else:
                input_ids.extend(subword_tokens_ids)
                word_ids.extend([i] * len(subword_tokens_ids))
        assert len(word_ids) == len(word_ids), 'Error word_ids length!'
        if add_special_tokens:
            word_ids = [None] + word_ids + [None]
        return input_ids, word_ids

    def get_input_ids(self, text: Union[str, list[str]], is_split_into_words=False, add_special_tokens=True):
        assert (isinstance(text, str) and not is_split_into_words) or (isinstance(text, list) and isinstance(text[0], str) and is_split_into_words), 'Inavalid input!'
        if isinstance(text, str):
            return self.sequence_to_ids(text.split(), add_special_tokens)
        else:
            return self.sequence_to_ids(text, add_special_tokens)
    
    # currently only support __call__
    def __call__(self, text, **kwargs):
        is_split_into_words = kwargs.pop('is_split_into_words', False)
        add_special_tokens = kwargs.pop('add_special_tokens', True)
        kwargs['is_split_into_words'] = is_split_into_words
        kwargs['add_special_tokens'] = add_special_tokens

        if is_split_into_words:
            is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
        else:
            is_batched = isinstance(text, (list, tuple))

        if is_batched:
            all_input_and_word_ids_pairs = [self.get_input_ids(t, is_split_into_words, add_special_tokens) for t in text]
            all_input_ids = [pair[0] for pair in all_input_and_word_ids_pairs]
            all_word_ids = [pair[1] for pair in all_input_and_word_ids_pairs]
            padding = kwargs.pop('padding', False)
            batch_outputs = {}
            for input_ids in all_input_ids:
                outputs = self.tokenizer.prepare_for_model(
                    ids=input_ids,
                    return_attention_mask=True,
                    **kwargs
                )
                for key, value in outputs.items():
                    if key not in batch_outputs:
                        batch_outputs[key] = []
                    batch_outputs[key].append(value)
            batch_outputs = self.tokenizer.pad(
                batch_outputs,
                padding=padding,
                max_length=self.tokenizer.model_max_length,
                return_attention_mask=True,
            )
            batch_encoding = BatchEncoding(batch_outputs)
            all_word_ids = self._refine_word_ids(batch_encoding, all_word_ids, add_special_tokens)
        else:
            input_ids, word_ids = self.get_input_ids(text, is_split_into_words, add_special_tokens)
            batch_encoding = self.tokenizer.prepare_for_model(
                ids=input_ids,
                return_attention_mask=True,
                **kwargs
            )
            all_word_ids = self._refine_word_ids(batch_encoding, word_ids, add_special_tokens)
        return SlowBatchEncodingWrapper(batch_encoding, all_word_ids)

    def _refine_word_ids(self, batch_encoding, all_word_ids, add_special_tokens: bool):
        if isinstance(all_word_ids[0], list):
            for i in range(len(all_word_ids)):
                all_word_ids[i] = all_word_ids[i][:len(batch_encoding['input_ids'][i])]
                if add_special_tokens: all_word_ids[i][-1] = None
        else:
            all_word_ids = all_word_ids[:len(batch_encoding['input_ids'])]
            if add_special_tokens: all_word_ids[-1] = None
            all_word_ids = [all_word_ids]
        return all_word_ids

    def __getattr__(self, name: str) -> Any:
        # If can't find attribute in class, look for it from raw_tokenizer
        return getattr(self.tokenizer, name)


class SlowBatchEncodingWrapper(BatchEncoding):
    def __init__(self, batch_encoding: BatchEncoding, all_word_ids: list):
        assert 'word_ids' not in batch_encoding, 'This wrapper mean to provide access to word_ids for slow tokenizers!'
        self.batch_encoding = batch_encoding
        self.word_ids_list = all_word_ids

    def _refine_word_ids(self, all_word_ids):
        # Check if special token
        if all_word_ids[0][0] is None and all_word_ids[0][-1] is None:
            special_added = True
        else:
            special_added = False

        for i in range(len(all_word_ids)):
            all_word_ids[i] = all_word_ids[i][:len(self.batch_encoding['input_ids'][i])]
            if special_added: all_word_ids[i][-1] = None
        return all_word_ids

    def word_ids(self, batch_index=0):
        return self.word_ids_list[batch_index]
    
    def __getitem__(self, item: Union[int, str]):
        return self.batch_encoding.__getitem__(item)

    def __getattr__(self, name: str) -> Any:
        # Only reimplement word_ids
        return getattr(self.batch_encoding, name)