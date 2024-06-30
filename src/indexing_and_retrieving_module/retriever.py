import pathlib
import itertools
from collections import defaultdict
from typing import Callable

import torch
import numpy as np
from torch.utils.cpp_extension import load

from modeling_module.colbert import ColBERT
from modeling_module.coil import COIL
from indexing_and_retrieving_module.indexer import *
from indexing_and_retrieving_module.utils import *
from indexing_and_retrieving_module.residual import ResidualEmbeddingsStrided
from args import Args
try:
    from indexing_and_retrieving_module.retriever_ext import scatter as c_scatter
except ImportError:
    raise ImportError(
        'Cannot import scatter module.'
        ' Make sure you have compiled the retriever extension.'
    )

class Retriever:
    def __init__(self, args, model):
        self.args: Args = args
        self.model = model

    def search(self, queries: list[str] | str, indexer: Indexer, return_sorted: bool):
        pass


class ColBERTRetriever(Retriever):
    def __init__(self, args: Args, model: ColBERT):
        super(ColBERTRetriever, self).__init__(args, model)
        self.model: ColBERT

    def encode_queries(
            self, queries: list[str] | str, return_tensor=False
    ) -> tuple[Union[np.ndarray, torch.Tensor], list[list[int]], list[tuple[int, int]]]:
        if not isinstance(queries, list):
            queries = [queries]
        # encode queries
        self.model.eval()
        with torch.inference_mode():
            q_embeddings, q_masks = self.model.encode_query(queries)
        q_embs = []
        q_index_range = []
        start, end = 0, 0
        for i in range(len(queries)):
            if not return_tensor: q_emb = (q_embeddings[i])[q_masks[i].squeeze()].cpu().numpy()
            else: q_emb = (q_embeddings[i])[q_masks[i].squeeze()].cpu()
            q_embs.append(q_emb)
            start = end
            end = start + len(q_emb)
            q_index_range.append((start, end))
        if not return_tensor: q_embs = np.concatenate(q_embs, axis=0)
        else: q_embs = torch.cat(q_embs, dim=0)
        return q_embs, q_index_range

    def search(self, queries: list[str] | str, indexer: ColBERTIndexer, return_sorted=False):
        q_embs, q_index_range = self.encode_queries(queries=queries)
        # retrieve docs by token retrieval
        retrieved_doc_ids, all_retrieved_doc_ids = self.doc_retrieve(q_embs=q_embs, q_index_range=q_index_range, indexer=indexer)
        # return reranked score
        rerank_scores = self.doc_rerank(
            q_embs=q_embs,
            q_index_range=q_index_range,
            retrieved_doc_ids=retrieved_doc_ids,
            all_retrieved_doc_ids=all_retrieved_doc_ids,
            indexer=indexer)
        if return_sorted:
            rerank_scores = [sorted(q_result, key=lambda x: x['score'], reverse=True) for q_result in rerank_scores]
        return rerank_scores

    def doc_retrieve(self, q_embs, q_index_range, indexer: ColBERTIndexer):
        retrieved_doc_ids = []  # list of lists, each list contains id for docs retrived for a query
        all_retrieved_doc_ids = None
        # retrieve top k similar tokens for each query token
        _, I = indexer.index_db.search(q_embs, k=self.args.top_k_tokens)   # Q * top_k_tokens
        for (start, end) in q_index_range:
            q_retrieved_token_ids = np.unique(I[start:end])
            q_retrieved_doc_ids = np.unique([indexer.inverted_list[token_id] for token_id in q_retrieved_token_ids]).tolist()
            retrieved_doc_ids.append(q_retrieved_doc_ids)
        all_retrieved_doc_ids = set(itertools.chain(*retrieved_doc_ids))
        return retrieved_doc_ids, all_retrieved_doc_ids
    
    def doc_rerank(self, q_embs: np.ndarray, q_index_range, retrieved_doc_ids, all_retrieved_doc_ids, indexer: ColBERTIndexer):
        # reconstruct all tokens from retrieved and get token ids for retrieved doc
        all_retrieved_doc_token_ids = []
        new_index_range = {} # id -> (new_start, new_end) new_start and new_end are start and end correspond to reconstructed matrix 
        new_start, new_end = None, None

        for id in all_retrieved_doc_ids:
            ex_start, ex_end = indexer.index_range[id]
            all_retrieved_doc_token_ids.append([*range(ex_start, ex_end)])
            if new_end is None:
                new_start = 0
            else:
                new_start = new_end
            new_end = ex_end - ex_start + new_start
            new_index_range[id] = (new_start, new_end)

        all_retrieved_doc_token_ids = list(itertools.chain(*all_retrieved_doc_token_ids))
        all_retrieved_doc_tokens: np.ndarray = indexer.index_db.reconstruct_batch(all_retrieved_doc_token_ids)

        rerank_scores = [] # list, each item is a list of dict
        # calculate score for each query and doc pair:
        for i, (q_start, q_end) in enumerate(q_index_range):
            q_i_embs = q_embs[q_start:q_end]
            cur_rerank_scores = []
            for id in retrieved_doc_ids[i]:
                doc_embs = all_retrieved_doc_tokens[new_index_range[id][0]:new_index_range[id][1]]
                sim_scores = q_i_embs @ doc_embs.transpose()
                score = sim_scores.max(axis=1).sum()
                cur_rerank_scores.append({'corpus_id': id, 'score': score.item()})
            rerank_scores.append(cur_rerank_scores)
        return rerank_scores
    
# class ColBERTRetrieverAlter(Retriever):
#     def __init__(self, args: Args, model: ColBERT):
#         super(ColBERTRetrieverNew, self).__init__(args, model)
#         self.model: ColBERT

#     def encode_queries(
#             self, queries: list[str] | str, return_tensor=False
#     ) -> tuple[Union[np.ndarray, torch.Tensor], list[list[int]], list[tuple[int, int]]]:
#         if not isinstance(queries, list):
#             queries = [queries]
#         # encode queries
#         self.model.eval()
#         with torch.inference_mode():
#             q_embeddings, q_masks = self.model.encode_query(queries)
#         q_embs = []
#         q_index_range = []
#         start, end = 0, 0
#         for i in range(len(queries)):
#             if not return_tensor: q_emb = (q_embeddings[i])[q_masks[i].squeeze()].cpu().numpy()
#             else: q_emb = (q_embeddings[i])[q_masks[i].squeeze()].cpu()
#             q_embs.append(q_emb)
#             start = end
#             end = start + len(q_emb)
#             q_index_range.append((start, end))
#         if not return_tensor: q_embs = np.concatenate(q_embs, axis=0)
#         else: q_embs = torch.cat(q_embs, dim=0)
#         return q_embs, q_index_range

#     def search(self, queries: list[str] | str, indexer: ColBERTIndexer, return_sorted=False):
#         q_embs, q_index_range = self.encode_queries(queries=queries)
#         # retrieve docs by token retrieval
#         retrieved_doc_ids = self.doc_retrieve(q_embs=q_embs, q_index_range=q_index_range, indexer=indexer)
#         # return reranked score
#         rerank_scores = self.doc_rerank(
#             q_embs=q_embs,
#             q_index_range=q_index_range,
#             retrieved_doc_ids=retrieved_doc_ids,
#             indexer=indexer)
#         if return_sorted:
#             rerank_scores = [sorted(q_result, key=lambda x: x['score'], reverse=True) for q_result in rerank_scores]
#         return rerank_scores

#     def doc_retrieve(self, q_embs, q_index_range, indexer: ColBERTIndexer):
#         retrieved_doc_ids = []  # list of lists, each list contains id for docs retrived for a query
#         # retrieve top k similar tokens for each query token
#         _, I = indexer.index_db.search(q_embs, k=self.args.top_k_tokens)   # Q * top_k_tokens
#         for (start, end) in q_index_range:
#             q_retrieved_token_ids = np.unique(I[start:end])
#             q_retrieved_doc_ids = np.unique([indexer.inverted_list[token_id] for token_id in q_retrieved_token_ids]).tolist()
#             retrieved_doc_ids.append(q_retrieved_doc_ids)
#         return retrieved_doc_ids
    
#     def doc_rerank(self, q_embs: np.ndarray, q_index_range, retrieved_doc_ids, indexer: ColBERTIndexer):
#         rerank_scores = [] # list, each item is a list of dict
#         # calculate score for each query and doc pair:
#         for i, (q_start, q_end) in enumerate(q_index_range):
#             q_i_embs = q_embs[q_start:q_end]
#             cur_rerank_scores = []
#             for id in retrieved_doc_ids[i]:
#                 doc_embs = indexer.index_db.reconstruct_n(indexer.index_range[id][1], indexer.index_range[id][1]-indexer.index_range[id][0])
#                 sim_scores = q_i_embs @ doc_embs.transpose()
#                 score = sim_scores.max(axis=1).sum()
#                 cur_rerank_scores.append({'corpus_id': id, 'score': score})
#             rerank_scores.append(cur_rerank_scores)
#         return rerank_scores


class COILRetriever(Retriever):
    def __init__(self, args, model: COIL):
        super(COILRetriever, self).__init__(args, model)
        self.model: COIL

    def search(self, queries: list[str] | str, indexer: COILIndexer, return_sorted=False):
        if not isinstance(queries, list):
            queries = [queries]
        assert indexer._finalized, 'Cannot search on an index that is not finalized'
        # Encode queries
        self.model.eval()
        with torch.inference_mode():
            q_encode_outputs = self.model.encode_query(queries)
            q_clss, q_repss, q_input_idss, q_attention_masks = q_encode_outputs['cls_emb'], q_encode_outputs['tok_embs'], q_encode_outputs['input_ids'], q_encode_outputs['attention_mask']
            q_attention_masks = self.model.mask_sep(q_attention_masks)

        all_q_cls = []
        q_tok_rep_dict = defaultdict(list)
        tok_qid_dict = defaultdict(list)
        for qid, (q_cls, q_reps, q_input_ids, q_attention_mask) in enumerate(zip(q_clss, q_repss, q_input_idss, q_attention_masks)):
            token_indices = torch.where(q_attention_mask > 0)[0]
            # exclude CLS, SEP, MASK
            q_input_ids = q_input_ids[token_indices][1:]
            q_tok_embs = q_reps[token_indices][1:]
            assert len(q_input_ids) == len(q_tok_embs), "Number of query token ids should equal number of token embeddings!"

            all_q_cls.append(q_cls)
            for i, q_tok_id in enumerate(q_input_ids):
                q_tok_id = q_tok_id.item()
                if q_tok_id in indexer.tok_rep_dict:
                    q_tok_rep_dict[q_tok_id].append(q_tok_embs[i].cpu().numpy())
                    tok_qid_dict[q_tok_id].append(qid)
            
        q_cls_reps: np.ndarray = np.stack(all_q_cls, axis=0)
        for q_tok_id in q_tok_rep_dict:
            q_tok_rep_dict[q_tok_id] = np.stack(q_tok_rep_dict[q_tok_id], axis=0)

        # Calculate scores
        match_scores = (q_cls_reps @ indexer.cls_reps.transpose()).astype('float32')  # cls score
        # tok_scores_dict = {} # q_tok_id -> tok_scores matrix
        # xét từng q_tok_id
        for q_tok_id in q_tok_rep_dict:
            tok_scores = q_tok_rep_dict[q_tok_id] @ indexer.tok_rep_dict[q_tok_id].transpose()    # score matrix for this q_tok_id (#q_tok_id_in_qs * #q_tok_id_in_ps)
            ivl_scatter_map = indexer.ivl_scatter_maps[q_tok_id]
            scatter_map = indexer.scatter_maps[q_tok_id]

            ivl_maxed_scores = np.empty(len(scatter_map), dtype='float32')   # (số lượng passge có chứa token) (xét các token có tok_q_id)
            for i in range(tok_scores.shape[0]):
                ivl_maxed_scores[:] = 0
                c_scatter.scatter_max(tok_scores[i], ivl_scatter_map, ivl_maxed_scores)
                q_offset = tok_qid_dict[q_tok_id][i]
                # Cập nhật điểm cho cặp query - passage bằng cách cộng thêm điểm của token sim
                # q_offset xác định query và shard_scatter_map xác định các passage
                new_maxed_scores = np.take_along_axis(match_scores[q_offset], indices=scatter_map, axis=0) + ivl_maxed_scores
                np.put_along_axis(match_scores[q_offset], indices=scatter_map, values=new_maxed_scores, axis=0)
        top_ids, top_scores = numpy_topk(match_scores, k=self.args.top_k_tokens, axis=1)
        
        rerank_scores = []
        for i in range(top_scores.shape[0]):
            rerank_scores.append([
                {'corpus_id': indexer.all_pids[id], 'score': score.item()} for (id, score) in zip(top_ids[i], top_scores[i])
            ])
        if return_sorted:
            rerank_scores = [sorted(q_result, key=lambda x: x['score'], reverse=True) for q_result in rerank_scores]
        return rerank_scores


class SentenceTransformersRetriever(Retriever):
    def __init__(self, args, model):
        super().__init__(args, model) 
        self.model: SentenceTransformer

    def search(self, queries: list[str] | str, indexer: SentenceTransformersIndexer):
        if not isinstance(queries, list):
            queries = [queries]
        with torch.inference_mode():
            q_embs = self.model.encode(queries)
        D, I = indexer.index_db.search(q_embs, k=self.args.top_k_tokens)
        rerank_scores = []
        for distances, indexs in zip(D, I):
            rerank_scores.append([
                {'corpus_id': indexer.all_pids[i], 'score': d} for (d, i) in zip(distances, indexs)])
        return rerank_scores
    

class ColBERTPLAIDRetriever(ColBERTRetriever):
    def __init__(self, args: Args, model: ColBERT):
        super(ColBERTPLAIDRetriever, self).__init__(args, model)
        ColBERTPLAIDRetriever.try_load_torch_extensions()

    def prepare(self, indexer: ColBERTPLAIDIndexer):
        self.indexer: ColBERTPLAIDIndexer = indexer
        _, self.index_doclens = torch.unique_consecutive(torch.tensor(indexer.inverted_list), return_counts=True)
        self.index_embeddings_strided = ResidualEmbeddingsStrided(indexer.codec, indexer.embeddings, self.index_doclens)
        self.index_offsets = self.index_embeddings_strided.codes_strided.offsets

    def search(self, queries: list[str] | str, indexer: ColBERTPLAIDIndexer = None, return_sorted=False):
        if indexer is None: assert self.indexer is not None
        else: self.prepare(indexer=indexer)
        q_embs, q_index_range = self.encode_queries(queries=queries, return_tensor=True)
        rerank_scores = []
        # search for one query at a time
        with torch.inference_mode():
            for i, (start, end) in enumerate(q_index_range):
                cur_q_embs = q_embs[start:end]
                pids, centroid_scores = self.doc_retrieve(q_embs=cur_q_embs)
                scores, pids = self.score_pids(
                    q_embs=cur_q_embs, pids=pids, 
                    centroid_scores=centroid_scores)
                scores, pids = scores.tolist(), pids.tolist()
                rerank_scores.append([
                    {'corpus_id': pid, 'score': score} for (pid, score) in zip(pids, scores)
                ])
        if return_sorted:
            rerank_scores = [sorted(q_result, key=lambda x: x['score'], reverse=True) for q_result in rerank_scores]
        return rerank_scores

    def doc_retrieve(self, q_embs: torch.Tensor):
        # find top closest centroids
        centroid_scores = self.indexer.codec.centroids @ q_embs.T
        cells = centroid_scores.topk(self.args.ncells, dim=0, sorted=False).indices.permute(1, 0) # LQ * ncells
        cells = cells.flatten().contiguous() # (LQ * ncells)
        cells = cells.unique(sorted=False)
        # generate candidate pid
        pids, cell_lengths = self.indexer.ivf.lookup(cells)
        sorted_pids = pids.sort()
        pids = sorted_pids.values.int()
        pids, pids_counts = torch.unique_consecutive(pids, return_counts=True)
        pids: torch.Tensor; pids_counts: torch.Tensor
        return pids, centroid_scores
    
    def score_pids(
            self, q_embs: torch.Tensor, pids: torch.Tensor, 
            centroid_scores:torch.Tensor
    ):
        '''
        Single query at a time
            q_embs: LQ * D 
        '''
        assert q_embs.dim() == 2
        idx = centroid_scores.max(-1).values >= self.args.centroid_score_threshold

        pids = ColBERTPLAIDRetriever.filter_pids(
            pids, centroid_scores, self.indexer.embeddings.codes, self.index_doclens,
            self.index_offsets, idx, self.args.ndocs
        )
        D_packed: torch.Tensor = ColBERTPLAIDRetriever.decompress_residuals(
            pids,
            self.index_doclens,
            self.index_offsets,
            self.indexer.codec.bucket_weights,
            self.indexer.codec.reversed_bit_map,
            self.indexer.codec.decompression_lookup_table,
            self.indexer.embeddings.residuals,
            self.indexer.embeddings.codes,
            self.indexer.codec.centroids,
            self.args.embedding_dim,
            self.args.nbits
        )
        D_packed = torch.nn.functional.normalize(D_packed.to(torch.float32), p=2, dim=-1)
        D_mask = self.index_doclens[pids.long()]
        assert D_packed.dim() == 2
        scores = D_packed @ q_embs.to(dtype=D_packed.dtype).T
        return ColBERTPLAIDRetriever.segmented_maxsim(scores, D_mask), pids

    @classmethod
    def try_load_torch_extensions(cls):
        if hasattr(cls, "loaded_extensions"):
            return
        os.add_dll_directory(os.path.join(pathlib.Path(__file__).parent.resolve(), "cpp_extensions", "PTHREADS-BUILT", "bin"))
        pthread_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), "cpp_extensions", "PTHREADS-BUILT")
        pthread_include_dir = os.path.join(pthread_dir, "include")
        pthread_library_dir = os.path.join(pthread_dir, "lib")
        extra_ldflags = [f'/LIBPATH:{pthread_library_dir}', 'pthreadVC2.lib']
        extra_include_paths = [pthread_include_dir]
        extra_cflags = ["/O2"]

        print(f"Loading filter_pids_cpp extension")
        filter_pids_cpp = load(
            name="filter_pids_cpp",
            sources=[os.path.join(pathlib.Path(__file__).parent.resolve(), "cpp_extensions", "filter_pids.cpp"),],
            extra_ldflags=extra_ldflags, 
            extra_include_paths=extra_include_paths,
            extra_cflags=extra_cflags, verbose=True,
            with_cuda=False
        )
        cls.filter_pids: Callable[..., torch.Tensor] = filter_pids_cpp.filter_pids_cpp

        print(f"Loading decompress_residuals_cpp extension")
        decompress_residuals_cpp = load(
            name="decompress_residuals_cpp",
            sources=[os.path.join(pathlib.Path(__file__).parent.resolve(), "cpp_extensions", "decompress_residuals.cpp"),],
            extra_ldflags=extra_ldflags, 
            extra_include_paths=extra_include_paths,
            extra_cflags=extra_cflags, 
            verbose=True,
            with_cuda=False
        )
        cls.decompress_residuals: Callable[..., torch.Tensor] = decompress_residuals_cpp.decompress_residuals_cpp

        print(f"Loading segmented_maxsim_cpp extension")
        segmented_maxsim_cpp = load(
            name="segmented_maxsim_cpp",
            sources=[os.path.join(pathlib.Path(__file__).parent.resolve(), "cpp_extensions", "segmented_maxsim.cpp"),],
            extra_ldflags=extra_ldflags, extra_include_paths=extra_include_paths,
            extra_cflags=extra_cflags, verbose=True,
            with_cuda=False
        )
        cls.segmented_maxsim: Callable[..., torch.Tensor] = segmented_maxsim_cpp.segmented_maxsim_cpp
        cls.loaded_extensions = True
    