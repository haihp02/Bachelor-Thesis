import os

import py_vncorenlp
from transformers import AutoConfig

from args import Args
from indexing_and_retrieving_module.retriever import ColBERTRetriever, COILRetriever, ColBERTPLAIDRetriever
from indexing_and_retrieving_module.indexer import ColBERTIndexer, ColBERTwKWIndexer, COILIndexer, ColBERTPLAIDIndexer
from modeling_module.coil import COIL
from modeling_module.colbert import ColBERT, ColBERTwKWMask
from data_module.utils import read_tsv

# Load collection
pairs = read_tsv('../data/zalo_qa/ms_marco_format/collection.tsv')
collection = {int(pair[0]): pair[1] for pair in pairs}

# Set ColBERT_model_args
ColBERT_model_args = Args()
ColBERT_model_args.name_or_path = 'vinai/phobert-base-v2'
ColBERT_model_args.model_max_length = 256
ColBERT_model_args.no_sep = True
ColBERT_model_args.query_marker_token = 'Hỏi'
ColBERT_model_args.doc_marker_token = 'Đáp'
ColBERT_model_args.top_k_tokens = 100
# ColBERT_model
model_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=ColBERT_model_args.name_or_path,
    output_hidden_states=True
)
ColBERT_model = ColBERT.from_pretrained('../output/ColBERT/best-phobert-finetuned-model/', args=ColBERT_model_args, config=model_config)
# Load ColBERT Index, Retriever
ColBERT_indexer = ColBERTIndexer(args=ColBERT_model_args, model=ColBERT_model)
ColBERT_indexer.load_index_db_from_disk('../vector_dbs/colbert-phobert-based')
ColBERT_retriever = ColBERTRetriever(args=ColBERT_model_args, model=ColBERT_model)



# Load ColBERTwKW Index
ColBERTwKW_indexer = ColBERTwKWIndexer(args=ColBERT_model_args, model=ColBERT_model)
ColBERTwKW_indexer.load_index_db_from_disk('../vector_dbs/colbert-kw-phobert-based-l2-0.7-threshold')

# Set PLAID_ColBERT_args
PLAID_ColBERT_args = Args()
PLAID_ColBERT_args.ncells = 4
PLAID_ColBERT_args.centroid_score_threshold = 0.45
PLAID_ColBERT_args.plaid_num_partitions_coef = 1
PLAID_ColBERT_args.ndocs = 1000

# Load ColBERT_PLAID Index, Retriever
PLAID_ColBERT_indexer = ColBERTPLAIDIndexer(args=PLAID_ColBERT_args, model=ColBERT_model)
PLAID_ColBERT_indexer.load_index_db_from_disk('../vector_dbs/colbert-plaid-phobert-based')
PLAID_ColBERT_retriever = ColBERTPLAIDRetriever(args=PLAID_ColBERT_args, model=ColBERT_model)
PLAID_ColBERT_retriever.prepare(indexer=PLAID_ColBERT_indexer)

# Load ColBERTwKW_PLAID Index, Retriever
PLAID_ColBERTwKW_indexer = ColBERTPLAIDIndexer(args=PLAID_ColBERT_args, model=ColBERT_model)
PLAID_ColBERTwKW_indexer.load_index_db_from_disk('../vector_dbs/colbert-kw-plaid-phobert-based-l2-0.7-threshold')
PLAID_ColBERTwKW_retriever = ColBERTPLAIDRetriever(args=PLAID_ColBERT_args, model=ColBERT_model)
PLAID_ColBERTwKW_retriever.prepare(indexer=PLAID_ColBERTwKW_indexer)


# Set COIL_model_args
COIL_model_args = Args()
COIL_model_args.name_or_path = 'vinai/phobert-base-v2'
COIL_model_args.model_max_length = 256
COIL_model_args.tok_embedding_dim = 32
COIL_model_args.cls_embedding_dim = 768
COIL_model_args.coil_score_type = 'full'
# COIL_model
COIL_model_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=COIL_model_args.name_or_path,
    output_hidden_states=True
)
COIL_model = COIL.from_pretrained('../output/COIL/best-phobert-finetuned-model', args=COIL_model_args, config=model_config)
# Load COIL Index, Retriever
COIL_indexer = COILIndexer(args=COIL_model_args, model=COIL_model)
COIL_indexer.load_finalized_index_db_from_disk(load_dir='../vector_dbs/coil-phobert-based/')
COIL_retriever = COILRetriever(args=COIL_model_args, model=COIL_model)

current_working_directory = os.getcwd()
ws_model = py_vncorenlp.VnCoreNLP(save_dir='D:\VnCoreNLP', annotators=['wseg'])
os.chdir(current_working_directory)