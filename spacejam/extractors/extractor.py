from typing import Tuple, Union
import torch


class Extractor:
    def __init__(self, model_type: str = 'dino_vits8', stride: int = 4, device: torch.device = torch.device("cuda")):
        """
        Args:
            model_type (str, optional): A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224 |
                          dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14 | dinov2_vitg14]. 
                          Defaults to 'dino_vits8'.
            stride (int, optional): stride of first convolution layer. small stride -> higher resolution. 
                                    Defaults to 4.
            device (torch.device, optional): Defaults to torch.device("cuda").
        """
        models = ['dino_vits8', 'dino_vits16', 'dino_vitb8', 'dino_vitb16',
                  'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14']
        # assert model_type in models, f"Model type {model_type} not supported!"

        self.stride = stride
        self.model_type = model_type
        self.device = device
        
    
    def get_feature_map(self, x, out_size: Union[Tuple[int, int], str]) -> torch.Tensor: 
        """
        x: [batch, c, H, W]
        out_size: (h, w) or 'same' (same size as input) or 'none' (no resizing)
        """
        raise NotImplementedError
    

    def get_attn_map(self, x, out_size: Union[Tuple[int, int], str]) -> torch.Tensor: 
        """
        x: [batch, c, H, W]
        out_size: (h, w) or 'same' (same size as input) or 'none' (no resizing)
        """
        raise NotImplementedError
    
    
