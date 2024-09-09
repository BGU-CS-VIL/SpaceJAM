from typing import Tuple, Union
from dino_vit_features.vit_extractor_v2 import ViTExtractor
from spacejam.extractors.extractor import Extractor
from torchvision import transforms
import torch.nn.functional as F
import torch
import numpy as np

class FeatureMapExtractor(Extractor):
    def __init__(self, model_type: str = 'dino_vits8', device: torch.device = torch.device("cuda"), num_patches: int = 64, stride: int = 4):
        """
        Args:
            model_type (str, optional): Available models: 'dino_vits8', 'dino_vits16', 'dino_vitb8', 'dino_vitb16', 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'. Defaults to 'dino_vits8'. 
            device (torch.device, optional): Defaults to torch.device("cuda").
            num_patches (int, optional): the number of patches per row (input resolution will be num_patches * patch_size). Defaults to 32.
        """
        self.is_dinov2 = 'dinov2' in model_type 

        super().__init__(model_type, stride, device)
              
        self.extractor = ViTExtractor(self.model_type, self.stride, device=self.device)

        self.layer_to_extract = self.extractor.num_layers - 1
        self.facet = 'token' if self.is_dinov2 else 'key' 

        self.embedding_dim = self.extractor.embedding_dim
        self.patch_size = self.extractor.model.patch_embed.patch_size[0] if self.is_dinov2 else self.extractor.model.patch_embed.patch_size
        
        self.num_patches = num_patches
        self.input_res = self.stride * (self.num_patches - 1) + self.patch_size

        print(f"ViTNewExtractor: input_res: {self.input_res}, num_patches: {self.num_patches}, embedding_dim: {self.embedding_dim}, patch_size: {self.patch_size}, stride: {self.stride}")

        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)




    def get_feature_map(self, x, out_size: Union[Tuple[int, int], str]) -> torch.Tensor:
        """
        x: [batch, 3, H, W]
        out_size: (h, w) or 'same' (same size as input) or 'none' (no resizing)
        """
        with torch.no_grad():
            if out_size == 'same':
                feat_res = x.shape[2]
            elif out_size == 'none':
                feat_res = 0
            else:
                feat_res = out_size[0]

            # Resize to input_res and normalize input
            if x.shape[2] * 0.85 > self.input_res:
                print(f"Warning: the input resolution (for the dino extractor) is {self.input_res}, but the input image resolution is {x.shape[2]}. This unnecessary downsampling may affect the performance.")
            x = F.interpolate(x, size=self.input_res, mode='bilinear', align_corners=False)
            x = transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)(x)

            # Extract features
            feat_map = self.extractor.extract_descriptors(x, self.layer_to_extract, facet=self.facet) # [N, 1, registers(opt) + n_patches * n_patches, embedding_dim]
            feat_map = feat_map[:, :, -self.num_patches * self.num_patches:, :]                       # [N, 1, n_patches * n_patches, embedding_dim]
            assert feat_map.shape == (x.shape[0], 1, self.num_patches * self.num_patches, self.embedding_dim), f"feat_map.shape ({feat_map.shape}) != ({x.shape[0]}, 1, {self.num_patches * self.num_patches}, {self.embedding_dim})"
            feat_map = feat_map.permute(0, 3, 2, 1).reshape(x.shape[0], self.embedding_dim, self.num_patches, self.num_patches) # [N, embedding_dim, n_patches, n_patches]
            
            # interpolate to desired resolution
            if feat_res > 0:
                return F.interpolate(feat_map, size=feat_res, mode='bilinear', align_corners=False)
            else:
                return feat_map
            