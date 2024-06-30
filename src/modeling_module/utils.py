from typing import Union, List

import torch
from torch import nn

def get_model_device(model: nn.Module):
    return next(model.parameters()).device

def to_device(device, tensors: Union[torch.Tensor, List[torch.Tensor]]):
    if not isinstance(tensors, list):
        tensors = [tensors]
    for tensor in tensors:
        if tensor is not None:
            tensor = tensor.to(device)
    return tensors
