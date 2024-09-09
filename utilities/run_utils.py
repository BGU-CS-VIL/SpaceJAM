from collections import defaultdict
import re
import sys
from typing import List, Type, TypeVar
import imageio
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import torch
import logging
import time
import os
import torch.nn as nn

from torch.utils.data import DataLoader
from spacejam.dataset import Dataset
from spacejam.loss import Loss
from spacejam.models.autoencoder import Autoencoder
from spacejam.models.transformers.transformer import Transformer
from spacejam.models.transformers.sequence_transformer import SequenceTransformer
from spacejam.models.transformers.reflection_transformer import ReflectionTransformer
from spacejam.atlas import Atlas
from utilities.pca import PCA
from utilities.training_utils import process_inputs_dict, select_first_inputs_config

import numpy as np
import random
import PIL.Image 
from pathlib import Path



class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def seed_everything(seed):
    if seed is not None and seed >= 0:
        print(f"Setting seed to {seed}")
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore # needed for reproducibility
        torch.backends.cudnn.benchmark = False  # type: ignore
    else:
        print("No seed set, results will not be reproducible")


   

# decorator that if field is not none, return it
def cached_field(field_name):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if getattr(self, field_name, None) is None:
                setattr(self, field_name, func(self, *args, **kwargs))
            return getattr(self, field_name)
        return wrapper
    return decorator



def apply_pca(desc, pca_n_components, trained_pca=None):
    """ desc shape [N, embedding_size] """
    if isinstance(desc, torch.Tensor):
        desc = desc.detach()
    if trained_pca is None:
        if isinstance(desc, torch.Tensor):
            from utilities.pca import PCA
        else:
            print("Using sklearn PCA")
            from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_n_components).fit(desc)
        print(f'PCA trained. The explained variance ratio of the PCA({pca_n_components}) model is: ', pca.explained_variance_ratio_, 'The sum is: ', pca.explained_variance_ratio_.sum()) 
    else:
        pca = trained_pca

    pca_descriptors = pca.transform(desc)                        # [N, pca_n_components]
    return pca_descriptors



def str_to_tuple(types):
    def str_to_tuple_inner(arg):
        elements_str = [elem.strip() for elem in arg.split(',')]
        if len(types) != len(elements_str):
            raise ValueError(f'len(types)={len(types)} must be equal to len(elements_str)={len(elements_str)}')
        return tuple(type_(elem) for elem, type_ in zip(elements_str, types))
    return str_to_tuple_inner




@torch.no_grad()
def train_pca_on_features(tensor):
    # Assuming tensor shape is [B, C, H, W]
    B, C, H, W = tensor.shape
    device = tensor.device
    tensor_reshaped = tensor.permute(0, 2, 3, 1).reshape(-1, C)

    pca = PCA(n_components=3)
    transformed = pca.fit_transform(tensor_reshaped)
    transformed = transformed.to(device).view(B, H, W, 3).permute(0, 3, 1, 2)
    return pca, transformed.min(), transformed.max()


@torch.no_grad()
def transform_with_pca(pca, tensor):
    B, C, H, W = tensor.shape
    device = tensor.device
    tensor_reshaped = tensor.permute(0, 2, 3, 1).reshape(-1, C)

    transformed = pca.transform(tensor_reshaped)
    transformed = transformed.to(device).view(B, H, W, 3).permute(0, 3, 1, 2)
    return transformed




@torch.no_grad()
def forward_inference_all(dataset: Dataset, autoencoder: Autoencoder, stn: nn.Module, atlas: Atlas, loss: Loss,
                          train_with_reflections=False, test_dataset=False, batch_size=1, num_of_batches=None):
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = atlas.device
    inputs_dicts_list: List[dict] = []
    encoded_keys_list: List[torch.Tensor] = []
    transformers_list: List[Transformer] = []

    if train_with_reflections:
        for batch_idx, inputs in enumerate(dataloader):
            if num_of_batches is not None and batch_idx >= num_of_batches:
                break

            inputs = process_inputs_dict(inputs, reflections=True, device=device)
            indices, imgs, keys, masks, frames = inputs['current_im_idx'], inputs['images'], inputs['keys'], inputs['masks'], inputs['frames']
            batch_size, ref_count = keys.shape[:2]

            falt_encoded_keys = autoencoder.encoder(keys.view(-1, *keys.shape[2:]))
            encoded_keys = falt_encoded_keys.view(batch_size, ref_count, *falt_encoded_keys.shape[1:])

            transformer: Transformer = stn(falt_encoded_keys)

            # for test dataset, we select the best reflection configuration based on the lowest loss
            if test_dataset:
                config = loss.loss_first_atlas_reflection(transformer, autoencoder.decoder, atlas.get_atlas(), keys, encoded_keys, masks, frames).argmin(dim=1) # [B]
                                                          
            # for train dataset, we select the current reflection configuration (the one used for the atlas)
            else:
                assert atlas.inputs_reflector is not None, "atlas.inputs_reflector is None but train_with_reflections is True"
                config = atlas.inputs_reflector.get_reflection_configurations(indices)  # [B]

            transformer.select_single_reflection_transformer(config, batch_size, ref_count)
            transformers_list.append(SequenceTransformer([ReflectionTransformer(transformer.img_size, config), transformer], 
                                                         combine_transformations=True))

            encoded_keys_list.append(encoded_keys[:, 0])
            inputs_dicts_list.append(select_first_inputs_config(inputs))

    else:
        for batch_idx, inputs in enumerate(dataloader):
            if num_of_batches is not None and batch_idx >= num_of_batches:
                break

            inputs = process_inputs_dict(inputs, reflections=False, device=atlas.device)
            encoded_keys = autoencoder.encoder(inputs['keys'])
            transformers_list.append(stn(encoded_keys))
            encoded_keys_list.append(encoded_keys)
            inputs_dicts_list.append(inputs)

    return transformers_list, encoded_keys_list, inputs_dicts_list




@torch.no_grad()
def visualize_training(inputs: dict, batch_transformer, encoded_keys: torch.Tensor, atlas: torch.Tensor,
                       autoencoder, epoch, save_path=None):

    indices, keys, masks, images = inputs["current_im_idx"], inputs["keys"], inputs["masks"], inputs["images"]
    warped_encoded_keys = batch_transformer(encoded_keys)
    warped_images = batch_transformer(images)
    atlas = atlas.repeat(keys.shape[0], 1, 1, 1)
    batch_transformer.set_image_size(atlas.shape[-2:])
    atlas_unwarped = batch_transformer(atlas, inverse=True)

    # save figures
    if save_path:
        dir_path = Path(os.path.dirname(save_path)) / f"epoch_{epoch}"
        os.makedirs(dir_path, exist_ok=True)
        os.makedirs(dir_path / "keys", exist_ok=True)
        os.makedirs(dir_path / "images", exist_ok=True)
        for idx, encoded_key, warp_encoded_key, img, warp_img in zip(indices, encoded_keys, warped_encoded_keys, images, warped_images):
            encoded_key = (encoded_key - encoded_key.min()) / (encoded_key.max() - encoded_key.min())
            warp_encoded_key = (warp_encoded_key - warp_encoded_key.min()) / (warp_encoded_key.max() - warp_encoded_key.min())
            plt.imsave(dir_path / "keys" / f"{idx:02d}.png", (encoded_key.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            plt.imsave(dir_path / "keys" / f"{idx:02d}_warped.png", (warp_encoded_key.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            plt.imsave(dir_path / "images" / f"{idx:02d}.png", (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            plt.imsave(dir_path / "images" / f"{idx:02d}_warped.png", (warp_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    row_titles = ['Masks', 'Encoded Keys', 'Warped Encoded Keys', 'Atlas (repeated)', 'Decoded Atlas Unwarped']
    column_titles = [f'Img {i+1}' for i in range(len(keys))]
    all_imgs = [masks, encoded_keys, warped_encoded_keys, atlas]

    imgs_ranges = [(x.min(), x.max()) for x in all_imgs]
    
    fig, axes = plt.subplots(nrows=len(all_imgs), ncols=len(keys), figsize=(int(1.5 * len(keys)) + 1, int(1.5 * len(all_imgs)) + 1))
    
    if len(all_imgs) == 1:
        axes = [axes]
    if len(keys) == 1:
        axes = [[ax] for ax in axes]
        
    for row_idx, row_data in enumerate(all_imgs):
        imgs_min, imgs_max = imgs_ranges[row_idx]
        for col_idx, img in enumerate(row_data):
            img_normalized = (img - imgs_min) / (imgs_max - imgs_min)

            ax = axes[row_idx][col_idx]
            ax.imshow(img_normalized.permute(1, 2, 0).cpu().numpy(), aspect='auto', extent=ax.get_xlim() + ax.get_ylim())
            ax.axis('off')

            if col_idx == 0:
                ax.text(-1.5, 0.5, f'{row_titles[row_idx]}\n({imgs_min:.2f}, {imgs_max:.2f})\n{list(img.shape)}', rotation=0, size='large', verticalalignment='center', transform=ax.transAxes)
            if row_idx == 0:
                ax.set_title(column_titles[col_idx])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle(f"Training Visualization Epoch {epoch}", fontsize=20, y=1.02)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.3, facecolor='white')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()

