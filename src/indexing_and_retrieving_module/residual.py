import itertools
from typing import Union

import numpy as np
import torch

from args import Args
from indexing_and_retrieving_module.strided import StridedTensor

class ResidualEmbeddings:

    def __init__(self, codes: torch.Tensor, residuals: torch.Tensor):
        assert codes.size(0) == residuals.size(0), (codes.size(), residuals.size())
        assert codes.dim() == 1 and residuals.dim() == 2, (codes.size(), residuals.size())
        assert residuals.dtype == torch.uint8

        self.codes = codes.to(torch.int32)  # (num_embeddings,) int32
        self.residuals = residuals   # (num_embeddings, compressed_dim) uint8


class ResidualCodec:
    Embeddings = ResidualEmbeddings

    def __init__(
            self, args: Args, 
            centroids: torch.Tensor, 
            avg_residual: Union[torch.Tensor, None] = None, 
            bucket_cutoffs: Union[torch.Tensor, None] = None,
            bucket_weights: Union[torch.Tensor, None] = None
    ):
        self.args: Args = args
        self.centroids: torch.Tensor = centroids.float()
        self.avg_residual: torch.Tensor = avg_residual
        self.bucket_cutoffs: torch.Tensor = bucket_cutoffs
        self.bucket_weights: torch.Tensor = bucket_weights
        self.arange_bits = torch.arange(0, self.args.nbits, dtype=torch.uint8)

        # We reverse the residual bits because arange_bits as
        # currently constructed produces results with the reverse
        # of the expected endianness
        self.reversed_bit_map = []
        mask = (1 << self.args.nbits) - 1
        for i in range(256):
            # The reversed byte
            z = 0
            for j in range(8, 0, -self.args.nbits):
                # Extract a subsequence of length n bits
                x = (i >> (j - self.args.nbits)) & mask
                # Reverse the endianness of each bit subsequence (e.g. 10 -> 01)
                y = 0
                for k in range(self.args.nbits - 1, -1, -1):
                    y += ((x >> (self.args.nbits - k - 1)) & 1) * (2 ** k)
                # Set the corresponding bits in the output byte
                z |= y
                if j > self.args.nbits:
                    z <<= self.args.nbits
            self.reversed_bit_map.append(z)
        self.reversed_bit_map = torch.tensor(self.reversed_bit_map).to(torch.uint8)

        # A table of all possible lookup orders into bucket_weights
        # given n bits per lookup
        keys_per_byte = 8 // self.args.nbits
        if self.bucket_weights is not None:
            self.decompression_lookup_table = (
                torch.tensor(
                    list(itertools.product( 
                        list(range(len(self.bucket_weights))),
                        repeat=keys_per_byte
                    ))
                ).to(torch.uint8))
        else:
            self.decompression_lookup_table = None
        

    def compress_into_code(self, embs: torch.Tensor) -> torch.Tensor:
        indices = (self.centroids @ embs.T.float()).max(dim=0).indices
        return indices
    
    def lookup_centroids(self, codes: torch.Tensor) -> torch.Tensor:
        centroids = self.centroids[codes.long()]
        return centroids

    def compress(self, embs):
        codes = self.compress_into_code(embs)
        centroids = self.lookup_centroids(codes)
        residuals = embs - centroids
        residuals = self.binarize(residuals)
        return ResidualCodec.Embeddings(codes=codes, residuals=residuals)

    def binarize(self, residuals: torch.Tensor):
        residuals = torch.bucketize(residuals.float(), self.bucket_cutoffs).to(dtype=torch.uint8)
        residuals = residuals.unsqueeze(-1).expand(*residuals.size(), self.args.nbits)  # add a new nbits-wide dim
        residuals = residuals >> self.arange_bits  # divide by 2^bit for each bit position
        residuals = residuals & 1  # apply mod 2 to binarize

        assert self.args.embedding_dim % 8 == 0
        assert self.args.embedding_dim % (self.args.nbits * 8) == 0, (self.args.embedding_dim, self.args.nbits)

        residuals_packed = np.packbits(np.asarray(residuals.contiguous().flatten()))
        residuals_packed = torch.as_tensor(residuals_packed, dtype=torch.uint8)
        residuals_packed = residuals_packed.reshape(residuals.size(0), self.args.embedding_dim // 8 * self.args.nbits)

        return residuals_packed
    

class ResidualEmbeddingsStrided:
    def __init__(self, codec: ResidualCodec, embeddings: ResidualEmbeddings, doclens: torch.Tensor):
        self.codec = codec
        self.embeddings = embeddings
        self.codes = embeddings.codes
        self.residuals = embeddings.residuals
        self.codes_strided = StridedTensor(self.codes, doclens)
        self.residuals_strided = StridedTensor(self.residuals, doclens)

    def lookup_pids(self, passage_ids):
        codes_packed, codes_lengths = self.codes_strided.lookup(passage_ids)#.as_packed_tensor()
        residuals_packed, _ = self.residuals_strided.lookup(passage_ids)#.as_packed_tensor()
        embeddings_packed = self.codec.decompress(ResidualEmbeddings(codes_packed, residuals_packed))
        return embeddings_packed, codes_lengths

    def lookup_codes(self, passage_ids):
        return self.codes_strided.lookup(passage_ids)#.as_packed_tensor()