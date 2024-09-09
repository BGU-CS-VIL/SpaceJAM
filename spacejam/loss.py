import torch
import torch.nn as nn
import torch.nn.functional as F
from spacejam.models.autoencoder import Autoencoder
from spacejam.models.inputs_reflector import InputsReflector
from spacejam.models.transformers.transformer import Transformer


loss_to_func = {
    'L1': lambda x, y, dim: F.l1_loss(x, y, reduction='none').mean(dim=dim),
    'L2': lambda x, y, dim: F.mse_loss(x, y, reduction='none').mean(dim=dim),
    'smooth_L1': lambda x, y, dim: F.smooth_l1_loss(x, y, reduction='none').mean(dim=dim),
}   


class Loss:
    def __init__(self, loss_function: str, weight_loss_with_masks: bool, weight_keys_with_masks: bool):
        self.criterion = loss_to_func[loss_function]
        self.weight_loss_with_masks = weight_loss_with_masks
        self.weight_keys_with_masks = weight_keys_with_masks

    # ------------------------ Reflection Loss  ------------------------
    def _loss_with_reflections(self, batch_transformer: Transformer, decoder: nn.Module, atlas: torch.Tensor, keys, encoded_keys, masks, frames):
        # Finds the loss between the keys and the atlas unwarped for every reflected input X_ij, and reflected atlas A_k
        batch_size, n_reflections, high_dim_channels = keys.shape[:3]

        atlas = InputsReflector.apply_reflection(atlas).repeat(batch_size * n_reflections, 1, 1, 1, 1)                       # [B*Ref, Ref, low_dim_channels, H, W]
        repeated_keys = keys.view(batch_size, n_reflections, 1, *keys.shape[-3:]).repeat(1, 1, n_reflections, 1, 1, 1)       # [B, Ref, Ref, high_dim_channels, H, W]

        repeated_masks = masks.view(batch_size, n_reflections, 1, *masks.shape[-3:]).repeat(1, 1, n_reflections, 1, 1, 1)    # [B, Ref, Ref, 1, H, W]
        repeated_frames = frames.view(batch_size, n_reflections, 1, *frames.shape[-3:]).repeat(1, 1, n_reflections, 1, 1, 1) # [B, Ref, Ref, 1, H, W]

        loss_masks = repeated_masks if self.weight_loss_with_masks else repeated_frames
        keys_masks = repeated_masks if self.weight_keys_with_masks else repeated_frames

        atlas_unwarped = batch_transformer(atlas, inverse=True)                                                                    # [B*Ref, Ref, low_dim_channels, H, W]
        decoded_unwarped_atlas = decoder(atlas_unwarped.view(-1, *atlas_unwarped.shape[-3:]))                                      # [B*Ref*Ref, high_dim_channels, H, W]
        decoded_unwarped_atlas = decoded_unwarped_atlas.view(batch_size, n_reflections, n_reflections, *keys.shape[-3:])           # [B, Ref, Ref, high_dim_channels, H, W]
        repeated_keys = repeated_keys * keys_masks                                                                                 # [B, Ref, Ref, high_dim_channels, H, W]
        loss = self.criterion(decoded_unwarped_atlas, repeated_keys, dim=[-3])                                                     # [B, Ref, Ref, H, W]

        loss = torch.mean(loss * loss_masks.squeeze(-3), dim=[-2, -1])   # [B, Ref, Ref]
        return loss  # [B, Ref, Ref]
    
    
    def loss_first_atlas_reflection(self, batch_transformer: Transformer, decoder: nn.Module, atlas: torch.Tensor, keys, encoded_keys, masks, frames):
        # Finds the loss between the keys and the atlas unwarped for every reflected input X_ij, and atlas A (= A_0)
        batch_size, n_reflections = keys.shape[:2]
        atlas = atlas.repeat(batch_size * n_reflections, 1, 1, 1)            # [B*Ref,  low_dim_channels,  H, W]

        masks = masks.view(batch_size, n_reflections, *masks.shape[-3:])     # [B, Ref, 1, H, W]
        frames = frames.view(batch_size, n_reflections, *frames.shape[-3:])  # [B, Ref, 1, H, W]

        loss_masks = masks if self.weight_loss_with_masks else frames        
        keys_masks = masks if self.weight_keys_with_masks else frames

        atlas_unwarped = batch_transformer(atlas, inverse=True)  # [B*Ref, low_dim_channels, H, W]
        decoded_unwarped_atlas = decoder(atlas_unwarped).view(batch_size, n_reflections, *keys.shape[-3:])  # [B, Ref, high_dim_channels, H, W]
        loss = self.criterion(decoded_unwarped_atlas, keys * keys_masks, dim=[-3])  # [B, Ref, H, W]

        return torch.mean(loss * loss_masks.squeeze(-3), dim=[-2, -1])   # [B, Ref]
    

    def loss_with_best_atlas_reflections(self, batch_transformer: Transformer, decoder: nn.Module, atlas: torch.Tensor, keys, encoded_keys, masks, frames):
        # atlas: [low_dim_channels, H, W]
        # decoder (nn.Module): Decoder from [N, low_dim, H, W] tensors to [N, high_dim, H, W] tensors
        # keys, encoded_keys, masks, frames: [B, Ref, C, H, W] for C = (high_dim_channels, low_dim_channels, 1, 1) respectively
        # batch_transformer: Transformer for [B*Ref, C, H, W] shapes or [B*Ref, Ref, C, H, W] shapes (broadcasts dim=1)
        loss = self._loss_with_reflections(batch_transformer, decoder, atlas, keys, encoded_keys, masks, frames)  # [B, Ref, Ref]
        best_k_configs = torch.argmin(loss, dim=2)                                                               # [B, Ref] (given i and j, get ths best k)
        i_indices, j_indices = torch.meshgrid(torch.arange(best_k_configs.shape[0], device=loss.device), torch.arange(best_k_configs.shape[1], device=loss.device)) 
        loss = loss[i_indices, j_indices, best_k_configs]                   
        return loss  # [B, Ref]
    # ------------------------ Reflection Loss  ------------------------
    


    # ------------------------ Loss without Reflections  ------------------------
    def loss_without_reflections(self, batch_transformer: Transformer, decoder: nn.Module, atlas: torch.Tensor, keys, encoded_keys, masks, frames):
        batch_size = keys.shape[0]
        atlas = atlas.repeat(batch_size, 1, 1, 1)  # [B, low_dim_channels, H, W]

        keys_masks = masks if self.weight_keys_with_masks else frames   # [B, 1, H, W]
        loss_masks = masks if self.weight_loss_with_masks else frames   # [B, 1, H, W]

        atlas_unwarped = batch_transformer(atlas, inverse=True)  # [B, low_dim_channels, H, W]
        decoded_unwarped_atlas = decoder(atlas_unwarped)  # [B, high_dim_channels, H, W]
        loss = self.criterion(decoded_unwarped_atlas, keys * keys_masks, dim=[-3])  # [B, H, W]

        return torch.mean(loss * loss_masks.squeeze(-3), dim=[-2, -1])  # [B]
    # ------------------------ Loss without Reflections  ------------------------



    # ------------------------ Reconstruction Loss  ------------------------
    def reconstruction_loss(self, autoencoder: Autoencoder, keys, masks, frames):
        reconstruction = autoencoder(keys.view(-1, *keys.shape[-3:])).view(*keys.shape)  # [B, [ref], C, H, W]
        loss = self.criterion(keys, reconstruction, dim=[-3])                            # [B, [ref], H, W]
        loss = loss * masks.squeeze(-3)                                                  # [B, [ref], H, W] * [B, [ref], H, W]
        return loss.mean() 
    # ------------------------ Reconstruction Loss  ------------------------