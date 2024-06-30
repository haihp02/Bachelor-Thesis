import json
from tqdm import tqdm
from collections import defaultdict

import numpy as np

from data_module.utils import InputExample, read_json, read_tsv
from args import Args

class DataProcessor(object):
    '''
    Base class for data processor
    '''
    def __init__(self, args: Args):
        self.args: Args = args
    
    def get_examples(self, mode):
        raise NotImplementedError
    
   
class IRTripletProcessor(DataProcessor):
    def __init__(self, args: Args):
        super(IRTripletProcessor, self).__init__(args)

    def load_data(self):
        collection = self.tsv_pair_to_dict(self.args.collection_path)
        queries = self.tsv_pair_to_dict(self.args.queries_path)
        if queries is None:
            train_queries = self.tsv_pair_to_dict(self.args.train_queries)
            val_queries = self.tsv_pair_to_dict(self.args.val_queries)
            queries = {**train_queries, **val_queries}
        train_triples = self.read_triples_to_tuples()
        val_qrels = self.tsv_qrels_to_dict(self.args.val_qrels_path)
        return collection, queries, train_triples, val_qrels

    def get_text_triples(self, collection, queries, triples):
        text_triples = [(queries[q_id], collection[p_pid], collection[n_pid]) for (q_id, p_pid, n_pid) in triples]
        text_triples =[{'query': q, 'passage': p, 'negative_passage': n_p} for (q, p, n_p) in text_triples]
        return text_triples
    
    def get_text_pairs(self, collection, queries, pairs):
        text_pairs = []
        for (q_id, p_id_list) in pairs.items():
            text_pairs.extend([(queries[q_id], collection[p_id]) for p_id in p_id_list])
        text_pairs = [{'query': q, 'passage': p} for (q, p) in text_pairs]
        return text_pairs

    def get_examples(self, mode):
        collection, queries, train_triples, val_qrels = self.load_data()
        if mode == 'train':
            return self.get_text_triples(collection, queries, train_triples)
        elif mode == 'val':
            return self.get_text_pairs(collection, queries, val_qrels)
        
    def tsv_qrels_to_dict(self, tsv_file_path):
        if tsv_file_path is None:
            return None
        pairs = read_tsv(tsv_file_path)
        result_dict = defaultdict(list)
        for pair in pairs:
            if int(pair[0]) in result_dict:
                result_dict[int(pair[0])].append(int(pair[2]))
            else:
                result_dict[int(pair[0])] = [int(pair[2])]
        return result_dict

    def tsv_pair_to_dict(self, tsv_file_path):
        if tsv_file_path is None:
            return None
        pairs = read_tsv(tsv_file_path)
        result_dict = {int(pair[0]): pair[1] for pair in pairs}
        return result_dict
    
    def read_triples_to_tuples(self):
        triples = read_tsv(self.args.train_triples_path)
        return [(int(q_id), int(p_pid), int(n_pid)) for (q_id, p_pid, n_pid) in triples]
    
    def read_qrels_to_dict(self):
        qrel_tuples = read_tsv(self.args.val_qrels_path)
        qrels = [(int(qid), int(pid))  for (qid, _1, pid, _2) in qrel_tuples]
        qrels_dict = defaultdict(list)
        for qrel in qrels:
            if qrel[0] in qrels_dict:
                qrels_dict[qrel[0]].append(qrel[1])
            else:
                qrels_dict[qrel[0]] = [qrel[1]]
        return qrels_dict
    
class IRTripletWithNERProcessor(IRTripletProcessor):
    
    def __init__(self, args):
        super(IRTripletWithNERProcessor, self).__init__(args)

    def load_data(self):
        collection, queries, train_triples, val_qrels = super(IRTripletWithNERProcessor, self).load_data()
        with open(self.args.train_data_path, 'r', encoding='utf8') as f:
            raw_ner_tags = json.load(f)
            ner_tags = {int(k): raw_ner_tags[k] for k in raw_ner_tags}
        return collection, queries, train_triples, val_qrels, ner_tags
    
class IRTripletWithPOSProcessor(IRTripletProcessor):
    
    def __init__(self, args):
        super(IRTripletWithPOSProcessor, self).__init__(args)

    def load_data(self):
        collection, queries, train_triples, val_qrels = super(IRTripletWithPOSProcessor, self).load_data()
        with open(self.args.train_data_path, 'r', encoding='utf8') as f:
            raw_pos_tags = json.load(f)
            pos_tags = {int(k): raw_pos_tags[k] for k in raw_pos_tags}
        return collection, queries, train_triples, val_qrels, pos_tags