import os
import pickle

import numpy as np
import torch

def save_pickle(save_object, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(save_object, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        return pickle.load(f)
    
def array_slice(array: np.ndarray, axis: int, start: int, end:int, step: int=1):
    return array[(slice(None),) * (axis % array.ndim) + (slice(start, end, step),)]

def numpy_topk(input, k, axis=None, ascending=False):
    if not ascending:
        input *= -1
    ind = np.argpartition(input, k, axis=axis)
    ind = array_slice(ind, axis=axis, start=0, end=k)
    input = np.take_along_axis(input, ind, axis=axis) # k non-sorted values
    # sort within k elements
    ind_part = np.argsort(input, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind_part, axis=axis) 
    return ind, val
    

def optimize_ivf(orig_ivf: torch.Tensor, orig_ivf_lengths: torch.Tensor, emb2pid: torch.Tensor):
    if not torch.is_tensor(emb2pid):
        emb2pid = torch.tensor(emb2pid)

    ivf = emb2pid[orig_ivf]
    unique_pids_per_centroid = []
    ivf_lengths = []

    offset = 0
    for length in orig_ivf_lengths.tolist():
        pids: torch.Tensor = torch.unique(ivf[offset:offset+length])
        unique_pids_per_centroid.append(pids)
        ivf_lengths.append(pids.size(0))
        offset += length
    ivf = torch.cat(unique_pids_per_centroid)
    ivf_lengths = torch.tensor(ivf_lengths)

    return ivf, ivf_lengths

