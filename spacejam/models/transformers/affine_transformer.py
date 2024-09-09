from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from spacejam.models.transformers.transformer import Transformer
from utilities.run_utils import cached_field

class AffineTransformer(Transformer):
    def __init__(self, theta, img_size, align_corners=False):
        super().__init__()
        assert theta.dim() == 3 and theta.shape[1:] == (2, 3), "theta must be a 3D tensor of shape (batch_size, 2, 3)"
        self.theta = theta
        self.img_size = img_size
        self.theta_inv = self._invert_theta(self.theta)
        self.align_corners = align_corners
        self.grid = self.make_affine_grid(self.theta, [self.theta.shape[0], 1, *self.img_size])
        self.grid_inv = self.make_affine_grid(self.theta_inv, [self.theta.shape[0], 1, *self.img_size])


    def forward(self, x, inverse=False):
        has_reflections = x.dim() >= 5
        n_reflections = x.shape[1] if has_reflections else 1
        batch_size = x.shape[0]
        shape_before_reshape = x.shape

        grid = self.grid_inv if inverse else self.grid  # [Batch*num_reflections, H, W, 2] (if has_reflections) or [Batch, H, W, 2] (if no reflections)

        if has_reflections:
            x = x.reshape(-1, *x.shape[-3:])

            assert self.theta.shape[0] == batch_size * n_reflections, f"Expected theta to have batch_size * n_reflections = {batch_size * n_reflections} rows, but got {self.theta.shape[0]} rows"
            if len(shape_before_reshape) == 6:  # [Batch, num_reflections, num_reflections, C, H, W]
                grid = self._get_reflection_expanded_grid_inv(batch_size, n_reflections) if inverse else self._get_reflection_expanded_grid(batch_size, n_reflections)    # [Batch*num_reflections*num_reflections, H, W, 2]
            elif len(shape_before_reshape) > 6:   
                raise ValueError(f"Expected x to have 4 (no reflections), 5 (with reflections) or 6 (with double reflections) dimensions, but got {x.dim()} dimensions")

        res = F.grid_sample(x, grid, align_corners=self.align_corners)
        
        if has_reflections:
            res = res.reshape(shape_before_reshape)
            
        return res
    
    
    def get_transfomation_params(self):
        """
        :param affine_mat: (N, 2, 3) tensor, the predicted affine matrices.

        :return: A dictionary of outputs (N, 1) tensors for each parameter (determinant, rot, scale, shear_x, shear_y, tx, ty)
        """
        rot_mat = self.theta[:, :2, :2]
        determinants = torch.det(rot_mat) # (N, 1)
        
        affine_mat = self.theta.reshape(-1, 6)
        a, b, tx, c, d, ty = torch.chunk(affine_mat, chunks=6, dim=1)  
        return {
            'determinant': determinants,
            'scale': torch.sqrt(a ** 2 + b ** 2),  
            'rot': torch.rad2deg(torch.atan2(b, a)),
            'shear_x': torch.rad2deg(torch.atan2(c, d)),
            'shear_y': torch.rad2deg(torch.atan2(b, a)),
            'tx': tx,
            'ty': ty
        }
    
    
    @cached_field("_grid_reflection_expanded")
    def _get_reflection_expanded_grid(self, batch_size, n_reflections):
        return self._expand_grid_for_reflections(self.grid, batch_size, n_reflections)
        
    @cached_field("_grid_inv_reflection_expanded")
    def _get_reflection_expanded_grid_inv(self, batch_size, n_reflections):
        return self._expand_grid_for_reflections(self.grid_inv, batch_size, n_reflections)

    def _expand_grid_for_reflections(self, grid, batch_size, n_reflections):
        grid = grid.reshape(batch_size, n_reflections, 1, *grid.shape[-3:])                # [Batch, num_reflections, 1, H, W, 2]
        grid = grid.repeat(1, 1, n_reflections, 1, 1, 1)                                   # [Batch, num_reflections, num_reflections, H, W, 2]
        grid = grid.reshape(batch_size * n_reflections * n_reflections, *grid.shape[-3:])  # [Batch*num_reflections*num_reflections, H, W, 2]
        return grid

            
    @staticmethod
    def make_homogenous_grid(shape, device):
        N, _, H, W = shape
        y, x = torch.meshgrid([torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device)])
        y, x = y.flatten(), x.flatten()
        grid = torch.stack([x, y, torch.ones(x.shape[0], device=device)], dim=0)  # [3, H*W]
        grid = grid.unsqueeze(0).repeat(N, 1, 1)                                  # [N, 3, H*W]
        return grid

    @staticmethod
    def make_affine_grid(theta, size):
        N, _, H, W = size
        theta = theta.reshape(N, 2, 3)                                            # [N, 2, 3]
        grid = AffineTransformer.make_homogenous_grid(size, device=theta.device)  # [N, 3, H*W]
        grid_t = torch.bmm(theta, grid)                                           # ([N, 2, 3] @ [N, 3, H*W]) = [N, 2, H*W]
        grid_t = grid_t.permute(0, 2, 1).reshape(N, H, W, 2)                      # [N, H, W, 2]
        return grid_t
    
    
    @staticmethod
    def _invert_theta(theta):
        theta = torch.cat([theta, torch.tensor([[[0, 0, 1]]], device=theta.device).repeat(theta.shape[0], 1, 1)], dim=1)
        theta = torch.inverse(theta)
        theta = theta[:, :2, :]
        return theta
        
        
    @staticmethod
    def warp_image(x : torch.Tensor, img_size: Tuple[int, int], theta : torch.Tensor, inverse=False, align_corners=False) -> torch.Tensor:
        """map image to new space by applying transformation.
        This function should be used only when needed, as it not efficient (not caching computations) and does not support batch processing.
        
        Args:
            x (torch.tensor): expected shape (C, H, W) for image / feature-map
            img_size (tuple[int, int]): expected shape (H, W) for image
            theta (torch.tensor): expected shape (2, 3) for affine transformation
            inverse (bool, optional): forward or backward transformation. Defaults to False.
            align_corners (bool, optional): Defaults to False.
        
        Returns:
            torch.tensor: expected shape (C, H, W) for warped image
        """
        H, W = img_size
        
        if img_size != x.shape[-2:]:
            print(f"Warning: img_size {img_size} does not match x.shape[-2:] {x.shape[-2:]}")
        
        if inverse:
            theta = AffineTransformer._invert_theta(theta.unsqueeze(0)).squeeze(0)
            
        grid = F.affine_grid(theta.unsqueeze(0), [1, 1, H, W], align_corners=align_corners)
        x_transformed = F.grid_sample(x.unsqueeze(0), grid, align_corners=align_corners)
        
        return x_transformed.squeeze(0)
    
    
    @staticmethod
    def warp_points(x : torch.Tensor, img_size: Tuple[int, int], theta : torch.Tensor, inverse=False, normalize=True) -> torch.Tensor:
        """map points to new space by applying transformation.
        This function should be used only when needed, as it not efficient (not caching computations) and does not support batch processing.
    
        Args:
            x (torch.tensor): expected shape (N, 2) for points
            img_size (tuple[int, int]): expected shape (H, W) for image. Used to normalize points to [-1, 1] range.
            theta (torch.tensor): expected shape (2, 3) for affine transformation
            inverse (bool, optional): forward or backward transformation. Defaults to False.
            normalize (bool, optional): normalize points to [-1, 1] range and back. Defaults to True (points are expected to be in range [0, W]x[0, H]).
            
        Returns:
            torch.tensor: expected shape (N, 2) for warped points
        """
        N = x.shape[0]
        H, W = img_size
        
        x = x.clone()
        theta = theta.clone()
        
        inverse = not inverse  # "inversing the inverse" - For warping images we use the inverse transformation, for points, the forward 
        
        if inverse:
            theta = AffineTransformer._invert_theta(theta.unsqueeze(0)).squeeze(0)
    
        if normalize:   # from [0, W]x[0, H] to [-1, 1]x[-1, 1]
            assert torch.max(x) > 1, f"It seems that points are in range [0, 1], but it is expected that they are in range [0, {W}]x[0, {H}]. Got x between {torch.min(x[:, 0])} and {torch.max(x[:, 0])} and y between {torch.min(x[:, 1])} and {torch.max(x[:, 1])}"    
                        
            x[:, 0] = x[:, 0] / (W / 2) - 1
            x[:, 1] = x[:, 1] / (H / 2) - 1
                            
        pointsA_hom = torch.cat([x, torch.ones(N, 1, device=x.device)], dim=1) # [N, 3]
        theta_hom = torch.cat([theta, torch.tensor([[0, 0, 1]], device=theta.device)], dim=0) # [3, 3]
        
        points_transformed = pointsA_hom @ theta_hom.T # [N, 3] @ [3, 3] = [N, 3]
        points_transformed = points_transformed[:, :2] #                   [N, 2]
        
        if normalize:   # from [-1, 1]x[-1, 1] to [0, W]x[0, H]
            points_transformed[:, 0] = (points_transformed[:, 0] + 1) * (W / 2)
            points_transformed[:, 1] = (points_transformed[:, 1] + 1) * (H / 2)

        return points_transformed
    
