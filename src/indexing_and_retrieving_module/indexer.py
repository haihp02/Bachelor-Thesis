import os
import pickle
import math
import random
from tqdm import tqdm, trange
from collections import defaultdict
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import faiss
from sentence_transformers import SentenceTransformer

from modeling_module.colbert import ColBERT
from modeling_module.coil import COIL
from modeling_module.utils import get_model_device
from indexing_and_retrieving_module.utils import *
from indexing_and_retrieving_module.residual import ResidualCodec
from indexing_and_retrieving_module.strided import StridedTensor
from args import Args

class Indexer:
    def __init__(self, args, model):
        self.args: Args = args
        self.model = model

    def _encode(self, collection):
        ''' Encode a collection of docs/ passages to vectors '''
        raise NotImplemented
    
    def add(self, encoded):
        ''' Add vectors into index database '''
        raise NotImplemented
    
    def update_collection(self, current_collection: dict, new_collection: dict):
        if not current_collection:
            return new_collection
        # Check for common keys
        common_keys = set(current_collection.keys()) & set(new_collection.keys())
        if common_keys:
            raise ValueError(f"Error: Collections have overlapping keys: {common_keys}")
        return {**current_collection, **new_collection}

    def reset(self):
        pass
    
    def save_index_db_to_disk(self, save_dir):
        pass

    def load_index_db_from_disk(self, load_dir):
        pass


class ColBERTIndexer(Indexer):
    def __init__(self, args: Args, model: ColBERT):
        super(ColBERTIndexer, self).__init__(args, model)
        self.model: ColBERT

        self.index_db = faiss.IndexFlatIP(self.args.embedding_dim)
        self.inverted_list = []
        self.index_range = {} # pid -> (starts, ends), end not included [start:end] when retrieve
        self.collection = {}

    def _encode(self, collection: dict):
        encoded = {}

        keys = list(collection.keys())
        texts = [collection[k] for k in keys]

        batch_size = self.args.batch_size
        self.model.eval()
        for i in trange(0, len(keys), batch_size):
            if i + batch_size <= len(keys):
                keys_2_encode = keys[i:i+batch_size]
                texts_2_encode = texts[i:i+batch_size]
            else:
                keys_2_encode = keys[i:]
                texts_2_encode = texts[i:]
            with torch.inference_mode():
                embeddings, masks = self.model.encode_doc(texts_2_encode)
            for j, key in enumerate(keys_2_encode):
                encoded[key] = (embeddings[j])[masks[j].squeeze(-1)].cpu().numpy()
        return encoded
    
    def add(self, collection: dict):
        assert len(self.inverted_list) == self.index_db.ntotal, "Inverted list must have same size with index db!"
        # Update collection
        self.collection = self.update_collection(current_collection=self.collection, new_collection=collection)
        encoded: dict[object, np.ndarray] = self._encode(collection)
        for key, embeddings in encoded.items():
            assert embeddings.shape[1] == self.index_db.d, "Embeddings must have same size with index predefined vector size!"
            relative_ivl = [key] * embeddings.shape[0]
            start = self.index_db.ntotal
            self.index_db.add(embeddings)
            end = self.index_db.ntotal
            self.inverted_list.extend(relative_ivl)
            self.index_range[key] = (start, end)

    def reset(self):
        self.index_db.reset()
        self.inverted_list = []
        self.index_range = {}
        self.collection = {}

    def save_index_db_to_disk(self, save_dir, save_collection=False):
        faiss.write_index(self.index_db, os.path.join(save_dir, 'colbert_index.index'))
        save_pickle(self.inverted_list, os.path.join(save_dir, 'colbert_ivl.pkl'))
        save_pickle(self.index_range, os.path.join(save_dir, 'index_range.pkl'))      
        if save_collection:
            save_pickle(self.collection, os.path.join(save_dir, 'collection.pkl'))

    def load_index_db_from_disk(self, load_dir, load_collection=False):
        assert os.path.exists(os.path.join(load_dir, 'colbert_index.index')) and os.path.exists(os.path.join(load_dir, 'colbert_ivl.pkl')) and os.path.exists(os.path.join(load_dir, 'index_range.pkl')), "Load dir missing files!"
        self.index_db = faiss.read_index(os.path.join(load_dir, 'colbert_index.index'))
        self.inverted_list = load_pickle(os.path.join(load_dir, 'colbert_ivl.pkl'))
        self.index_range = load_pickle(os.path.join(load_dir, 'index_range.pkl'))
        if load_collection:
            collection_path = os.path.join(load_dir, 'collection.pkl')
            assert os.path.exists(collection_path), 'Collection is not in load directory!'
            self.collection = load_pickle(collection_path)


class ColBERTwKWIndexer(ColBERTIndexer):

    def _encode(self, collection: dict):
        encoded = {}

        keys = list(collection.keys())
        texts = [collection[k] for k in keys]

        batch_size = self.args.batch_size
        self.model.eval()
        for i in trange(0, len(keys), batch_size):
            if i + batch_size <= len(keys):
                keys_2_encode = keys[i:i+batch_size]
                texts_2_encode = texts[i:i+batch_size]
            else:
                keys_2_encode = keys[i:]
                texts_2_encode = texts[i:]
            with torch.inference_mode():
                embeddings, punctuation_mask, weight_mask = self.model.encode_doc(texts_2_encode)
            for j, key in enumerate(keys_2_encode):
                if not self.args.apply_weight_mask:
                    encoded[key] = (embeddings[j])[weight_mask[j].squeeze(-1) >= self.args.kw_threshold].cpu().numpy()
                else:
                    encoded[key] = (embeddings[j] * weight_mask[j])[weight_mask[j].squeeze(-1) >= self.args.kw_threshold].cpu().numpy()
        return encoded


class COILIndexer(Indexer):
    def __init__(self, args, model: COIL):
        super(COILIndexer, self).__init__(args, model)
        self.model: COIL

        self.all_pids: Union[list, None] = []
        self.cls_reps: Union[np.ndarray, None]  = None
        self.tok_rep_dict: dict[object, np.ndarray] = dict()
        self.tok_pid_dict: dict[object, list] = defaultdict(list)
        self.collection: dict = {}
        self._finalized = False
        self.ivl_scatter_maps: dict[object, np.ndarray]
        self.scatter_maps: dict[object, np.ndarray]

    def _encode(self, collection: dict):
        encoded = {}

        keys = list(collection.keys())
        texts = [collection[k] for k in keys]

        batch_size = self.args.batch_size
        self.model.eval()
        for i in trange(0, len(keys), batch_size):
            if i + batch_size <= len(keys):
                keys_2_encode = keys[i:i+batch_size]
                texts_2_encode = texts[i:i+batch_size]
            else:
                keys_2_encode = keys[i:]
                texts_2_encode = texts[i:]
            with torch.inference_mode():
                encode_outputs = self.model.encode_doc(texts_2_encode) # a dict
                clss, repss, input_idss, attention_masks = encode_outputs['cls_emb'], encode_outputs['tok_embs'], encode_outputs['input_ids'], encode_outputs['attention_mask']
                attention_masks = self.model.mask_sep(attention_masks)
                for cls, reps, input_ids, mask, key in zip(clss, repss, input_idss, attention_masks, keys_2_encode):
                    encoded[key] = {
                        'cls_emb': cls,
                        'tok_embs': reps,
                        'input_ids': input_ids,
                        'attention_mask': mask
                    }
        return encoded

    def add(self, collection: dict):
        assert self._finalized == False, 'Cannot add to finalized index!'
        # Update collection
        self.collection = self.update_collection(current_collection=self.collection, new_collection=collection)

        encoded: dict[object, dict[str, np.ndarray]] = self._encode(collection)
        pids = []
        all_cls = []
        # all_reps = []
        
        tok_rep_dict = defaultdict(list)    # id -> [tok reps]
        tok_pid_dict = defaultdict(list)    # id -> [pid list (pid cho từng tok rep tương ứng của tok_rep_dict)], dict of inverted list
        for pid, encoded_p in encoded.items():
            tokens_indices = torch.where(encoded_p['attention_mask'] > 0)[0]
            # exclude CLS, SEP, MASK
            input_ids = encoded_p['input_ids'][tokens_indices][1:]
            tok_embs = encoded_p['tok_embs'][tokens_indices][1:]
            assert len(input_ids) == len(tok_embs), "Number of token ids should equal number of token embeddings!"

            all_cls.append(encoded_p['cls_emb'].cpu().numpy())
            pids.append(pid)
            # all_reps.append(tok_embs.cpu())

            # Dict: token id -> [token reps] for each passage
            for i, tok_id in enumerate(input_ids):
                tok_id = tok_id.item()
                tok_rep_dict[tok_id].append(tok_embs[i].cpu().numpy())
                tok_pid_dict[tok_id].append(pid)

        # Update index
        if not self.cls_reps: self.cls_reps = np.stack(all_cls, axis=0)
        else: self.cls_reps = np.concatenate(self.cls_reps, np.stack(all_cls, axis=0))
        self.all_pids.extend(pids)

        for tok_id in tok_rep_dict:
            self.tok_pid_dict[tok_id].extend(tok_pid_dict[tok_id])
            tok_rep_dict[tok_id] = np.stack(tok_rep_dict[tok_id], axis=0)
            if tok_id in self.tok_rep_dict:
                self.tok_rep_dict[tok_id] = np.concatenate([self.tok_rep_dict[tok_id], tok_rep_dict[tok_id]], axis=0)
            else:
                self.tok_rep_dict[tok_id] = tok_rep_dict[tok_id]
                
    def reset(self):
        self.all_pids = []
        self.cls_reps = None
        self.tok_rep_dict = dict()
        self.tok_pid_dict = dict()
        self.collection = {}
        self._finalized = False

    def finalize(self):
        # Finalize the index by calculate the scatter map
        ivl_scatter_maps = {}
        scatter_maps = {}
        pid_to_id = dict(((pid, id) for id, pid in enumerate(self.all_pids)))
        for tok_id, pid_list in self.tok_pid_dict.items():
            '''
            scatter_map: danh sách index của các passage có chứa token
            ivl_scatter_map: danh sách index của các passage có chứa token trong scatter_map, các index được lại theo số lần token xuất hiện trong mỗi passage
            eg: scatter_map = [1, 3, 4], ivl_scatter_map = [0, 0, 0, 1, 1, 2] -> token xuất hiện trong các passage có index là 1, 3, 4 và xuất hiện 3 lần trong passage có index là 1, 3 lần với index 2 và 1 lần với index 4 ...
            index != pid
            '''
            scatter_map, ivl_scatter_map = self._build_scatter_map(
                pid_list=pid_list,
                pid_2_id_map=pid_to_id
            )
            ivl_scatter_maps[tok_id] = ivl_scatter_map
            scatter_maps[tok_id] = scatter_map
        self.ivl_scatter_maps = ivl_scatter_maps
        self.scatter_maps = scatter_maps
        self._finalized = True

    def _build_scatter_map(self, pid_list, pid_2_id_map):
        _scatter_map = [pid_list[0]]
        ivl_scatter_map = [0]
        for eid in pid_list[1:]:
            if eid != _scatter_map[-1]:
                _scatter_map.append(eid)
            ivl_scatter_map.append(len(_scatter_map) - 1)

        ivl_scatter_map = np.array(ivl_scatter_map)
        scatter_map = list(map(lambda x: pid_2_id_map[x], _scatter_map))
        scatter_map = np.array(scatter_map)
        return scatter_map, ivl_scatter_map

    def save_index_db_to_disk(self, save_dir, save_collection=False):
        np.save(os.path.join(save_dir, 'cls_reps.npy'), self.cls_reps)
        save_pickle(self.all_pids, os.path.join(save_dir, 'all_pids.pkl'))
        save_pickle(self.tok_rep_dict, os.path.join(save_dir, 'tok_rep_dict.pkl'))
        save_pickle(self.tok_pid_dict, os.path.join(save_dir, 'tok_pid_dict.pkl'))
        if save_collection:
            save_pickle(self.collection, os.path.join(save_dir, 'collection.pkl'))
    
    def load_index_db_to_disk(self, load_dir, load_collection=False):
        assert not self._finalized, 'Cannot load new for finalized index!'
        self.cls_reps = np.load(os.path.join(load_dir, 'cls_reps.npy'))
        self.all_pids = load_pickle(os.path.join(load_dir, 'all_pids.pkl'))
        self.tok_rep_dict = load_pickle(os.path.join(load_dir, 'tok_rep_dict.pkl'))
        self.tok_pid_dict = load_pickle(os.path.join(load_dir, 'tok_pid_dict.pkl'))
        if load_collection:
            collection_path = os.path.join(load_dir, 'collection.pkl')
            assert os.path.exists(collection_path), 'Collection is not in load directory!'
            self.collection = load_pickle(collection_path)

    def save_finalized_index_db_to_disk(self, save_dir, save_collection=False):
        if not self._finalized:
            self.finalize()
        self.save_index_db_to_disk(save_dir, save_collection)
        save_pickle(self.ivl_scatter_maps, os.path.join(save_dir, 'ivl_scatter_maps.pkl'))
        save_pickle(self.scatter_maps, os.path.join(save_dir, 'scatter_maps.pkl'))

    def load_finalized_index_db_from_disk(self, load_dir, load_collection=False):
        assert not self._finalized, 'Cannot load new for finalized index!'
        self.load_index_db_to_disk(load_dir, load_collection)
        self.ivl_scatter_maps = load_pickle(os.path.join(load_dir, 'ivl_scatter_maps.pkl'))
        self.scatter_maps = load_pickle(os.path.join(load_dir, 'scatter_maps.pkl'))
        self._finalized = True


class SentenceTransformersIndexer(Indexer):
    def __init__(self, args, model: SentenceTransformer):
        super().__init__(args, model)
        self.model: SentenceTransformer

        self.index_db = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
        self.all_pids: Union[list, None] = []
        self.collection: dict = {}
    
    def _encode(self, collection: dict):
        keys = list(collection.keys())
        texts = [collection[k] for k in keys]

        batch_size = self.args.batch_size
        self.model.eval()
        with torch.inference_mode():
            embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        encoded = {keys[i]: embeddings[i] for i in range(len(keys))}
        return encoded
    
    def add(self, collection: dict):
        assert len(self.all_pids) == self.index_db.ntotal, "Number of passages and embeddings must be the same!!"
        # Update collection
        self.collection = self.update_collection(current_collection=self.collection, new_collection=collection)
        encoded: dict[object, np.ndarray] = self._encode(collection)
        self.all_pids.extend(list(encoded.keys()))
        self.index_db.add(np.stack(list(encoded.values()), axis=0))

    def reset(self):
        self.index_db.reset()
        self.all_pids = []
        self.collection = {}

    def save_index_db_to_disk(self, save_dir, save_collection=False):
        faiss.write_index(self.index_db, os.path.join(save_dir, 'sentence_transformers_index.index'))
        save_pickle(self.all_pids, os.path.join(save_dir, 'all_pids.pkl'))     
        if save_collection:
            save_pickle(self.collection, os.path.join(save_dir, 'collection.pkl'))
    
    def load_index_db_from_disk(self, load_dir, load_collection=False):
        assert os.path.exists(os.path.join(load_dir, 'sentence_transformers_index.index')) and os.path.exists(os.path.join(load_dir, 'all_pids.pkl')), "Load dir missing files!"
        self.index_db = faiss.read_index(os.path.join(load_dir, 'sentence_transformers_index.index'))
        self.all_pids = load_pickle(os.path.join(load_dir, 'all_pids.pkl'))
        if load_collection:
            collection_path = os.path.join(load_dir, 'collection.pkl')
            assert os.path.exists(collection_path), 'Collection is not in load directory!'
            self.collection = load_pickle(collection_path)


class ColBERTPLAIDIndexer(Indexer):
    def __init__(self, args, model):
        super(ColBERTPLAIDIndexer, self).__init__(args=args, model=model)
        self.model: ColBERT
        self.inverted_list = []
        self.index_range = {}
        self.index_db: Union[torch.Tensor, None] = None
        self.collection = {}
        self._finalized: bool = False

    def _encode(self, collection):
        encoded = {}

        keys = list(collection.keys())
        texts = [collection[k] for k in keys]

        batch_size = self.args.batch_size
        self.model.eval()
        for i in trange(0, len(keys), batch_size):
            if i + batch_size <= len(keys):
                keys_2_encode = keys[i:i+batch_size]
                texts_2_encode = texts[i:i+batch_size]
            else:
                keys_2_encode = keys[i:]
                texts_2_encode = texts[i:]
            with torch.inference_mode():
                embeddings, masks = self.model.encode_doc(texts_2_encode)
            for j, key in enumerate(keys_2_encode):
                encoded[key] = (embeddings[j])[masks[j].squeeze(-1)].cpu()
        return encoded
    
    def add(self, collection: dict):
        assert self.index_db is None or len(self.inverted_list) == self.index_db.size(0), "Inverted list must have same size with index db!"
        # Update collection
        self.collection = self.update_collection(current_collection=self.collection, new_collection=collection)
        encoded: dict[object, torch.Tensor] = self._encode(collection)
        embeddings_list = []
        current_db_size = self.index_db.size(0) if self.index_db is not None else 0
        for key, embeddings in encoded.items():
            assert embeddings.dim() == 2, "Embedding dim must be 2!"
            assert self.index_db == None or embeddings.size(1) == self.index_db.size(1), "Embeddings must have same size with index vector size!"
            relative_ivl = [key for _ in range(embeddings.size(0))]
            start = current_db_size + len(embeddings_list)
            end = start + embeddings.size(0)
            embeddings_list.append(embeddings)
            self.inverted_list.extend(relative_ivl)
            self.index_range[key] = (start, end)
        if self.index_db is None:
            self.index_db = torch.cat(embeddings_list, dim=0)
        else:
            self.index_db = torch.cat([self.index_db] + embeddings_list, dim=0)

    def finalize(self):
        self._setup_centroids_and_embeddings()
        self._build_ivf()
        self._finalized = True

    def _setup_centroids_and_embeddings(self):
        # Sample embeddings to train K-mean
        sampled_embs, num_partitions = self._sample_embs_for_partition()
        self.num_partitions = num_partitions
        # Train K-mean, residuals, self.codec save centroids information
        self.codec: ResidualCodec = self._train(embs=sampled_embs, num_partitions=num_partitions)
        # Compress residual
        assert self.index_db.dtype == torch.float32
        self.index_db = self.index_db.half()
        # self.embeddings save compressed embs information, the codes (centroids) and residuals
        self.embeddings = self.codec.compress(embs=self.index_db)

    def _build_ivf(self):
        codes = self.embeddings.codes
        sorted_codes = codes.sort()
        ivf, values = sorted_codes.indices, sorted_codes.values
        ivf_lengths = torch.bincount(values, minlength=self.num_partitions)
        assert ivf_lengths.size(0) == self.num_partitions
        # Transforms centroid->embedding ivf to centroid->passage ivf
        optimized_ivf, optimized_ivf_lengths = optimize_ivf(ivf, ivf_lengths, self.inverted_list)
        # self.ivf = optimized_ivf; self.ivf_length = optimized_ivf_length
        # Optimized ivf và ivf_length cho biết một centroid có thể map tới các pid nào
        # (tồn tại token thuộc passage nằm trong cụm). 
        # ivf_length chứa số lượng pid mà mỗi map tới (centroid_id -> number of pid)
        # ivf chứa các pid mà các centroid map tới được lưu liên tục từ centroid 0 tới hết
        self.ivf = StridedTensor(packed_tensor=optimized_ivf, lengths=optimized_ivf_lengths)

    def _train(self, embs: torch.Tensor, num_partitions: int):
        # Split
        embs = embs[torch.randperm(embs.size(0))]
        heldout_size = int(0.05 * embs.size(0))
        embs, heldout_embs = embs.split([embs.size(0) - heldout_size, heldout_size], dim=0)

        # Train k-mean
        kmeans = faiss.Kmeans(self.args.embedding_dim, num_partitions, niter=20, verbose=True)
        kmeans.train(embs.float().numpy())
        centroids = torch.from_numpy(kmeans.centroids)
        centroids = torch.nn.functional.normalize(centroids, p=2, dim=-1).float()        
        
        bucket_cutoffs, bucket_weights, avg_residual = self._compute_avg_residual(centroids, heldout_embs)
        codec = ResidualCodec(args=self.args, centroids=centroids, avg_residual=avg_residual,
                              bucket_cutoffs=bucket_cutoffs, bucket_weights=bucket_weights)
        return codec

    def _compute_avg_residual(self, centroids: torch.Tensor, heldout_embs: torch.Tensor):
        compressor = ResidualCodec(args=self.args, centroids=centroids)

        heldout_codes = compressor.compress_into_code(heldout_embs)
        heldout_reconstruct = compressor.lookup_centroids(heldout_codes)
        heldout_avg_residual = heldout_embs - heldout_reconstruct
        avg_residual = torch.abs(heldout_avg_residual).mean(dim=0).cpu()

        num_options = 2 ** self.args.nbits
        quantiles = torch.arange(0, num_options, device=heldout_avg_residual.device) * (1 / num_options)
        bucket_cutoffs_quantiles, bucket_weights_quantiles = quantiles[1:], quantiles + (0.5 / num_options)

        bucket_cutoffs = heldout_avg_residual.float().quantile(bucket_cutoffs_quantiles)
        bucket_weights = heldout_avg_residual.float().quantile(bucket_weights_quantiles)
        return bucket_cutoffs, bucket_weights, avg_residual.mean()
    
    @classmethod
    def from_colbert_indexer(cls, args: Args, model: ColBERT, colbert_indexer: ColBERTIndexer = None, colbert_index_dir: str = None):
        assert colbert_indexer or colbert_index_dir, 'Must provide ColBERTIndexer or save path!'
        if colbert_indexer is None and colbert_index_dir is not None:
            colbert_indexer = ColBERTIndexer(args, model)
            colbert_indexer.load_index_db_from_disk(load_dir=colbert_index_dir)
        colbert_plaid_indexer = ColBERTPLAIDIndexer(args, model)
        colbert_plaid_indexer.inverted_list = colbert_indexer.inverted_list
        colbert_plaid_indexer.index_range = colbert_indexer.index_range
        colbert_plaid_indexer.index_db = colbert_indexer.index_db.reconstruct_n()
        colbert_plaid_indexer.index_db = torch.tensor(colbert_plaid_indexer.index_db)
        return colbert_plaid_indexer

    def _sample_embs_for_partition(self):
        # passages
        num_passages = len(self.collection)
        num_sampled_pids = min(1 + int(16 * np.sqrt(self.index_db.size(0))), num_passages) # Simple alternative: < 100k: 100%, < 1M: 15%, < 10M: 7%, < 100M: 3%, > 100M: 1%
        sampled_pids = random.sample(self.collection.keys(), num_sampled_pids)
        # embeddings
        sampled_embs = torch.cat([self.index_db[self.index_range[i][0]:self.index_range[i][1]] for i in sampled_pids], dim=0)
        avg_doclen_est = sampled_embs.size(0) / num_sampled_pids
        # torch.save(sampled_embs.half(), os.path.join(self.config.index_path_, f'sample.{self.rank}.pt'))
        # select number of partitions (centroids)
        num_embeddings_est = num_passages * avg_doclen_est
        num_partitions = int(2 ** np.floor(np.log2(self.args.plaid_num_partitions_coef * np.sqrt(num_embeddings_est))))
        return sampled_embs, num_partitions
    
    def save_index_db_to_disk(self, save_dir, save_collection=False):
        assert self._finalized == True, 'PLAID Indexer must be finalized before saving!'
        save_pickle(self.inverted_list, os.path.join(save_dir, 'inverted_list.pkl'))
        save_pickle(self.ivf, os.path.join(save_dir, 'ivf.pkl'))
        save_pickle(self.codec, os.path.join(save_dir, 'codec.pkl'))
        save_pickle(self.embeddings, os.path.join(save_dir, 'embeddings.pkl'))
        if save_collection:
            save_pickle(self.collection, os.path.join(save_dir, 'collection.pkl'))

    def load_index_db_from_disk(self, load_dir, load_collection=False):
        assert self._finalized == False, 'PLAID Indexer already finalized! Can\'t load!'
        self.inverted_list = load_pickle(os.path.join(load_dir, 'inverted_list.pkl'))
        self.ivf = load_pickle(os.path.join(load_dir, 'ivf.pkl'))
        self.codec = load_pickle(os.path.join(load_dir, 'codec.pkl'))
        self.embeddings = load_pickle(os.path.join(load_dir, 'embeddings.pkl'))
        if load_collection:
            collection_path = os.path.join(load_dir, 'collection.pkl')
            assert os.path.exists(collection_path), 'Collection is not in load directory!'
            self.collection = load_pickle(collection_path)

        

class ColBERTFisrtTokensIndexer(ColBERTIndexer):
    def __init__(self, args: Args, model: ColBERT):
        super(ColBERTFisrtTokensIndexer, self).__init__(args, model)

    def _encode(self, collection: dict):
        encoded = {}

        keys = list(collection.keys())
        texts = [collection[k] for k in keys]

        batch_size = self.args.batch_size
        self.model.eval()
        for i in trange(0, len(keys), batch_size):
            if i + batch_size <= len(keys):
                keys_2_encode = keys[i:i+batch_size]
                texts_2_encode = texts[i:i+batch_size]
            else:
                keys_2_encode = keys[i:]
                texts_2_encode = texts[i:]
            with torch.inference_mode():
                embeddings, masks = self.model.encode_doc(texts_2_encode)
                masks = masks.squeeze(-1)
                num_tokens_to_keep = (masks.sum(dim=-1) * self.args.post_pruning_threshold).int()
                range_tensor = torch.arange(masks.size(1)).expand_as(masks)
                masks[range_tensor >= num_tokens_to_keep.unsqueeze(1)] = False
            for j, key in enumerate(keys_2_encode):
                encoded[key] = (embeddings[j])[masks[j]].cpu().numpy()
        return encoded

class ColBERTTopAttentionIndexer(ColBERTIndexer):
    def __init__(self, args: Args, model: ColBERT):
        super(ColBERTTopAttentionIndexer, self).__init__(args, model)

    def _encode(self, collection: dict):
        encoded = {}

        keys = list(collection.keys())
        texts = [collection[k] for k in keys]

        batch_size = self.args.batch_size
        self.model.eval()
        for i in trange(0, len(keys), batch_size):
            if i + batch_size <= len(keys):
                keys_2_encode = keys[i:i+batch_size]
                texts_2_encode = texts[i:i+batch_size]
            else:
                keys_2_encode = keys[i:]
                texts_2_encode = texts[i:]
            with torch.inference_mode():
                embeddings, masks = self.model.encode_doc(texts_2_encode)
                masked_embeddings = embeddings * masks.float()
                masks = masks.squeeze(-1)
                num_tokens_to_keep = (masks.sum(dim=-1) * self.args.post_pruning_threshold).int()
                range_tensor = torch.arange(masks.size(1)).expand_as(masks)
                avg_embeddings = masked_embeddings.sum(dim=1) / masks.sum(dim=-1).unsqueeze(dim=-1)
                # avg self attetion score between embeddings of one dc
                avg_self_attention = (masked_embeddings @ avg_embeddings.unsqueeze(1).transpose(1, 2)).squeeze(-1)
                _, indices = avg_self_attention.sort(descending=True)
                all_first_index_indices = indices[:,0].unsqueeze(-1).expand(-1, indices.size(1))
                # get index of top attention tokens
                indices[range_tensor >= num_tokens_to_keep.unsqueeze(1)] = all_first_index_indices[range_tensor >= num_tokens_to_keep.unsqueeze(1)]
                masks = (torch.zeros_like(masks).scatter(dim=-1, index=indices, value=1)).bool()
            for j, key in enumerate(keys_2_encode):
                encoded[key] = (embeddings[j])[masks[j]].cpu().numpy()
        return encoded
    
class ColBERTTopIDFIndexer(ColBERTIndexer):
    def __init__(self, args: Args, model: ColBERT):
        super(ColBERTTopIDFIndexer, self).__init__(args, model)
        self.doc_count = defaultdict(int)
        self.all_encoded: Union[dict, None] = None

    def _encode(self, collection: dict):
        encoded = {}

        keys = list(collection.keys())
        texts = [collection[k] for k in keys]

        batch_size = self.args.batch_size
        self.model.eval()
        for i in trange(0, len(keys), batch_size):
            if i + batch_size <= len(keys):
                keys_2_encode = keys[i:i+batch_size]
                texts_2_encode = texts[i:i+batch_size]
            else:
                keys_2_encode = keys[i:]
                texts_2_encode = texts[i:]
            with torch.inference_mode():
                tokenized_docs = self.model.doc_tokenizer(
                    texts_2_encode,
                    padding=True,
                    add_special_tokens=self.args.add_special_tokens,
                    truncation=True,
                    return_tensors='pt'
                ).to(get_model_device(self.model))
                embeddings, masks = self.model.doc(tokenized_docs['input_ids'], tokenized_docs['attention_mask'])
                masks = masks.squeeze(-1)
            for j, key in enumerate(keys_2_encode):
                encoded[key] = {
                    'embeddings': (embeddings[j])[masks[j]].cpu(),
                    'token_ids': tokenized_docs['input_ids'][j][masks[j]].tolist()
                }
        return encoded
    
    def add(self, collection: dict):
        assert len(self.inverted_list) == self.index_db.ntotal, "Inverted list must have same size with index db!"
        # Update collection
        self.collection = self.update_collection(current_collection=self.collection, new_collection=collection)
        encoded: dict[object, dict[str, Union[torch.Tensor, list]]] = self._encode(collection)
        for _, encoded_dict in encoded.items():
            assert encoded_dict['embeddings'].size(1) == self.index_db.d, "Embeddings must have same size with index predefined vector size!"
        if self.all_encoded is None:
            self.all_encoded = encoded
        else:
            self.all_encoded.update(encoded)

    def finalize(self):
        doc_count = defaultdict(int)
        # build idf scores
        for key, encoded_dict in self.all_encoded.items():
            token_ids = encoded_dict['token_ids']
            unique_tokens = set(token_ids)
            for token in unique_tokens: doc_count[token] += 1
        idf_scores = {token: math.log(len(self.collection) / count) for token, count in doc_count.items()}

        for key, encoded_dict in self.all_encoded.items():
            embeddings, token_ids = encoded_dict['embeddings'], encoded_dict['token_ids']
            # filter token embeddnigs by top IDF
            idf_tensor = torch.tensor([idf_scores[token_id] for token_id in token_ids])
            num_tokens_to_keep = int(embeddings.size(0) * self.args.post_pruning_threshold)
            _, topk_idf_indices = idf_tensor.topk(k=num_tokens_to_keep)
            embeddings = embeddings[topk_idf_indices.tolist() + [0]].numpy()
            # add to index_db
            relative_ivl = [key] * embeddings.shape[0]
            start = self.index_db.ntotal
            self.index_db.add(embeddings)
            end = self.index_db.ntotal
            self.inverted_list.extend(relative_ivl)
            self.index_range[key] = (start, end)

        self._finalized = True