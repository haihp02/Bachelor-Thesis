import os
import pathlib

import numpy as np
import torch
from torch.utils.cpp_extension import load


class StridedTensorCore:
    def __init__(self, packed_tensor: torch.Tensor, lengths: torch.Tensor, dim=None):
        self.dim = dim
        self.tensor = packed_tensor
        self.inner_dims = self.tensor.size()[1:]

        self.lengths = lengths.long() if torch.is_tensor(lengths) else torch.LongTensor(lengths)

        self.strides = _select_strides(self.lengths, [.5, .75, .9, .95]) + [self.lengths.max().item()]
        self.max_stride = self.strides[-1]

        zero = torch.zeros(1, dtype=torch.long, device=self.lengths.device)
        self.offsets = torch.cat((zero, torch.cumsum(self.lengths, dim=0)))
        if self.offsets[-2] + self.max_stride > self.tensor.size(0):
            padding = torch.zeros(self.max_stride, *self.inner_dims, dtype=self.tensor.dtype, device=self.tensor.device)
            self.tensor = torch.cat((self.tensor, padding))


class StridedTensor(StridedTensorCore):
    def __init__(self, packed_tensor: torch.Tensor, lengths: torch.Tensor, dim=None):
        super(StridedTensor, self).__init__(packed_tensor, lengths, dim)
        StridedTensor.try_load_torch_extensions()

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
        
        print(f"Loading segmented_lookup_cpp extension")
        segmented_lookup_cpp = load(
            name="segmented_lookup_cpp",
            sources=[os.path.join(pathlib.Path(__file__).parent.resolve(), "cpp_extensions", "segmented_lookup.cpp"),],
            extra_ldflags=extra_ldflags, 
            extra_include_paths=extra_include_paths,
            extra_cflags=extra_cflags, 
            verbose=True,
            with_cuda=False
        )
        cls.segmented_lookup = segmented_lookup_cpp.segmented_lookup_cpp
        cls.loaded_extensions = True

    def lookup(self, pids) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(pids, list):
            pids = torch.tensor(pids)
        assert pids.dim() == 1
        pids = pids.long().cpu()
        lengths = self.lengths[pids]
        offsets = self.offsets[pids]
        tensor = self.segmented_lookup(self.tensor, pids, lengths, offsets)
        return tensor, lengths
    


def _select_strides(lengths, quantiles):
    if lengths.size(0) < 5_000:
        return _get_quantiles(lengths, quantiles)
    sample = torch.randint(0, lengths.size(0), size=(2_000,))
    return _get_quantiles(lengths[sample], quantiles)

def _get_quantiles(lengths, quantiles):
    return torch.quantile(lengths.float(), torch.tensor(quantiles, device=lengths.device)).int().tolist()

def _create_view(tensor, stride, inner_dims):
    outdim = tensor.size(0) - stride + 1
    size = (outdim, stride, *inner_dims)

    inner_dim_prod = int(np.prod(inner_dims))
    multidim_stride = [inner_dim_prod, inner_dim_prod] + [1] * len(inner_dims)
    return torch.as_strided(tensor, size=size, stride=multidim_stride)


def _create_mask(lengths, stride, like=None):
    mask = torch.arange(stride) + 1
    mask = mask.unsqueeze(0) <= lengths.unsqueeze(-1)

    if like is not None:
        for _ in range(like.dim() - mask.dim()):
            mask = mask.unsqueeze(-1)
    return mask
