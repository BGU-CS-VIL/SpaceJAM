from typing import Optional, Tuple, Union
from spacejam.models.transformers.homography_transformer import HomographyTransformer
from spacejam.models.transformers.transformer import Transformer
import torch


class ReflectionTransformer(HomographyTransformer):
    theta_to_homography = torch.tensor([
        [[+1, +0, 0], [+0, +1, 0], [0, 0, 1]], 
        [[-1, +0, 0], [+0, +1, 0], [0, 0, 1]], 
        [[+1, +0, 0], [+0, -1, 0], [0, 0, 1]], 
        [[-1, +0, 0], [+0, -1, 0], [0, 0, 1]], 
        [[+0, +1, 0], [+1, +0, 0], [0, 0, 1]], 
        [[+0, -1, 0], [-1, +0, 0], [0, 0, 1]], 
    ], dtype=torch.float32) # [6, 3, 3]
            
    def __init__(self, img_size: Tuple[int, int], theta):
        assert theta.dim() == 1, f"Expected theta to have 1 dimension, but got {theta.dim()} dimensions (shape: {theta.shape})"
        theta = ReflectionTransformer.theta_to_homography[theta.cpu()].to(theta.device)  # [N, 3, 3]
        super().__init__(img_size, theta, lie_algebra=False)

