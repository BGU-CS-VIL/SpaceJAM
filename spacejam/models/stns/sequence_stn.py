import torch
import torch.nn as nn
from spacejam.models.transformers.sequence_transformer import SequenceTransformer
from spacejam.models.transformers.transformer import Transformer

class SequenceSTN(nn.Module):
    def __init__(self, stns: nn.ModuleList, avoid_interpolations = False):
        super().__init__()
        self.stns = stns
        self.avoid_interpolations = avoid_interpolations

    def forward(self, stn_input: torch.Tensor) -> Transformer:
        active_stns = [stn for stn in self.stns if stn.active]

        transformers = []
        curr_stn_input = stn_input
        for stn in active_stns:
            transformer = stn(curr_stn_input)
            transformers.append(transformer)

            if self.avoid_interpolations:   # warp the input image with all the transformations (wastful but avoids interpolation)
                curr_stn_input = SequenceTransformer(transformers, combine_transformations=True)(stn_input)
            else:
                curr_stn_input = transformer(curr_stn_input)

        return SequenceTransformer(transformers, combine_transformations=True)