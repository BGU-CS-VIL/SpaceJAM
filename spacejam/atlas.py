from __future__ import annotations
from typing import List, TYPE_CHECKING, Optional
import torch
import torch.nn.functional as F

from spacejam.models.inputs_reflector import InputsReflector
from spacejam.models.transformers.transformer import Transformer

class Atlas: 
    def __init__(self, device, inputs_reflector: Optional[InputsReflector], **kwargs): 
        self.device = device
        self.inputs_reflector = inputs_reflector
        
    def update_atlas(self, transformer: Transformer, indices: torch.Tensor, keys: torch.Tensor, encoded_keys: torch.Tensor, masks: torch.Tensor, frames: torch.Tensor, **kwargs):
        raise NotImplementedError()

    def get_atlas(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def _select_config_and_warp(self, transformer: Transformer, indices: torch.Tensor, encoded_keys: torch.Tensor, frames: torch.Tensor):
        if self.inputs_reflector is None:  # Training without reflections
            assert encoded_keys.dim() == 4 and frames.dim() == 4, f"Expected 4D tensors, got {encoded_keys.dim()} and {frames.dim()}"
            warped_encoded_keys = transformer(encoded_keys)
            warped_frames = transformer(frames)
        else:  # Training with reflections
            assert encoded_keys.dim() == 5 and frames.dim() == 5, f"Expected 5D tensors, got {encoded_keys.dim()} and {frames.dim()}"
            batch_size, ref_count = encoded_keys.shape[:2]
            warped_encoded_keys = transformer(encoded_keys.view(-1, *encoded_keys.shape[-3:]))  # [B*Ref, C, H, W]
            warped_frames = transformer(frames.view(-1, *frames.shape[-3:]))                    # [B*Ref, 1, H, W]

            config = self.inputs_reflector.get_reflection_configurations(indices)  # [B]
            warped_encoded_keys = warped_encoded_keys.view(batch_size, ref_count, *warped_encoded_keys.shape[-3:])[torch.arange(batch_size, device=self.device), config]  # [B, C, H, W]
            warped_frames = warped_frames.view(batch_size, ref_count, *warped_frames.shape[-3:])[torch.arange(batch_size, device=self.device), config]                    # [B, 1, H, W]

        return warped_encoded_keys, warped_frames


class SingleWarpedImageAtlas(Atlas):
    def __init__(self, device, inputs_reflector: Optional[InputsReflector], **kwargs): 
        super().__init__(device, inputs_reflector, **kwargs)
        self.atlas = None

    def update_atlas(self, transformer: Transformer, indices: torch.Tensor, keys: torch.Tensor, encoded_keys: torch.Tensor, masks: torch.Tensor, frames: torch.Tensor, index=0):
        warped_encoded_keys, _ = self._select_config_and_warp(transformer, indices, encoded_keys, frames)
        self.atlas = warped_encoded_keys[index]

    def get_atlas(self) -> torch.Tensor:
        assert self.atlas is not None, "Atlas is not initialized by update_atlas()"
        return self.atlas

