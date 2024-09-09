import torch
import torch.nn as nn
from spacejam.models.transformers.sequence_transformer import SequenceTransformer
from spacejam.models.transformers.transformer import Transformer

class RecurrentSTN(nn.Module):
    def __init__(self, stn: nn.Module, num_warps: int, avoid_interpolations = False):
        super().__init__()
        self.stn = stn
        self.num_warps = num_warps
        self.avoid_interpolations = avoid_interpolations

    def forward(self, stn_input: torch.Tensor) -> Transformer:
        transformers = []
        curr_stn_input = stn_input
        for _ in range(self.num_warps):
            transformer = self.stn(curr_stn_input)
            transformers.append(transformer)

            if self.avoid_interpolations:   # warp the input image with all the transformations (wastful but avoids interpolation)
                curr_stn_input = SequenceTransformer(transformers, combine_transformations=True)(stn_input)
            else:
                curr_stn_input = transformer.transform(curr_stn_input)

        return SequenceTransformer(transformers, combine_transformations=True)