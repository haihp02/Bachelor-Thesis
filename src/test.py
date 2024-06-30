import time
import json
import random
from tqdm import trange
from transformers import AutoConfig
from safetensors.torch import load_model

from indexing_and_retrieving_module.indexer import ColBERTPLAIDIndexer
from indexing_and_retrieving_module.retriever import ColBERTPLAIDRetriever
from training_and_evaluating_module.evaluate import InformationRetrievalEvaluator
from modeling_module.colbert import ColBERT, ColBERTwKWMask
from data_module.data_processor import IRTripletProcessor
from args import Args

args = Args()
args.name_or_path = 'vinai/phobert-base-v2'
args.query_max_length = 32
args.embedding_dim = 128
args.mask_punctuation = True
args.similarity_metric = 'cosine'
args.batch_size = 4
# args.query_pad_token = '<mask>'
args.query_pad_token = None
# args.query_marker_token = 'Frage'
# args.doc_marker_token = 'Antwort'
args.query_marker_token = 'Hỏi'
args.doc_marker_token = 'Đáp'
args.apply_weight_mask = True
args.weight_mask = -1
args.kw_threshold = 0.7

args.queries_path = '../data/zalo_qa/ms_marco_format/queries_ws.tsv'
args.collection_path = '../data/zalo_qa/ms_marco_format/collection_ws.tsv'
args.train_triples_path = '../data/zalo_qa/ms_marco_format/train_triples.tsv'
args.val_qrels_path = '../data/zalo_qa/ms_marco_format/dev_qrels.tsv'
args.top_k_tokens = 100    #top 100 cho mỗi token cho khoảng 100 doc cho pha rerank
args.ncells = 4
args.centroid_score_threshold = 0.45
args.plaid_num_partitions_coef = 1
args.ndocs = 1000

args.seed = 42
args.accuracy_at_k = [1, 3, 5, 10, 30, 50, 100]
args.precision_recall_at_k = [1, 3, 5, 10, 30, 50, 100]
args.mrr_at_k = [1, 3, 5, 10, 30, 50, 100]
args.ndcg_at_k = [1, 3, 5, 10, 30, 50, 100]
args.map_at_k = [1, 3, 5, 10, 30, 50, 100]
args.eval_name = 'test'
args.corpus_chunk_size = 2

data_processer = IRTripletProcessor(args=args)
collection, queries, _, val_qrels = data_processer.load_data()
queries_list = list(queries.values())

model_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=args.name_or_path,
    output_hidden_states=True
)
# model = ColBERT.from_pretrained('../output/ColBERT/best-phobert-finetuned-model', args=args, config=model_config)
# model = ColBERT.from_pretrained('../output/ColBERT/best-xlm-roberta-fintuned-model/', args=args, config=model_config)
model = ColBERTwKWMask.from_pretrained('../output/ColBERTwKW/PhoBERT/distill-l2/step_36000', args=args, config=model_config)

indexer = ColBERTPLAIDIndexer(args=args, model=model)
indexer.load_index_db_from_disk('../vector_dbs/colbert-plaid-phobert-based')
# indexer.collection = collection
# indexer.finalize()
# indexer.save_index_db_to_disk(save_dir='../vector_dbs/colbert-kw-plaid-phobert-based-l2-0.7-threshold')
# indexer =ColBERTPLAIDIndexer(args=args, model=model)
# indexer.load_index_db_from_disk('../vector_dbs/colbert-kw-plaid-phobert-based-l2-0.7-threshold')
retriever = ColBERTPLAIDRetriever(args, model)
retriever.prepare(indexer)

import random
from tqdm import trange

# retriever.prepare(indexer)

num_execution = 10
queries_batch_size = 1

total_time = 0
for i in trange(num_execution):
    sampled_list = random.sample(queries_list, queries_batch_size)
    start = time.time()
    result = retriever.search(queries=sampled_list, indexer=indexer, return_sorted=True)
    end = time.time()
    total_time += end - start

print('Avg time: ', f'{total_time/num_execution * 1000}ms')