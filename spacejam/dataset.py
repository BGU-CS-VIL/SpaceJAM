from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset):
    def __init__(self, imgs, keys, masks):
        self.image_resolution = imgs.shape[2:]
        
        self.all_images = imgs          # [N, 3,   h, w]
        self.imgs_dino_keys = keys      # [N, emb, h, w]
        self.imgs_masks = masks         # [N, 1,   h, w]
        self.keys_high_dim_size = self.imgs_dino_keys.shape[1]
        
                    
        self.number_of_images = self.all_images.shape[0]
        assert self.image_resolution[0] == self.image_resolution[1], "image_resolution[0] != image_resolution[1]"
        assert self.image_resolution[0] == self.all_images.shape[2], "image_resolution[0] != all_images.shape[2]"


    def get_img_data(self, im_idx):
        curr_im = self.all_images[im_idx]
        dino_keys = self.imgs_dino_keys[im_idx]
        mask = self.imgs_masks[im_idx]
        return curr_im, dino_keys, mask


    def __getitem__(self, index):
        input_image, dino_keys, mask = self.get_img_data(index)
        sample = {"images": input_image, 
                  "keys": dino_keys, 
                  "masks": mask,
                  "current_im_idx": index}
        return sample


    def __len__(self):
        return self.number_of_images

