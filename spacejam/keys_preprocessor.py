from spacejam.extractors.feature_map_extractor import FeatureMapExtractor

import torch
from PIL import Image
from utilities.pca import PCA
from torchvision import transforms
from tqdm import tqdm

from pathlib import Path
from typing import Tuple

from pathlib import Path
from dino_vit_features.cosegmentation import find_cosegmentation
from utilities.preprocessing_utils import stretch_images, unstretch_keys


class KeysPreprocessor:
    def __init__(self, pca_dim, dino_model_type, dino_model_num_patches,
                 device, img_size, extract_masks, weight_keys_with_masks=False, masks_method='coseg'):

        assert len(
            img_size) == 2, f"img_size {img_size} should be a list of 2 elements"
        assert img_size[0] == img_size[
            1], f"Currrently only square images are supported, but got {img_size}"
        self.img_size: Tuple[int, int] = tuple(img_size) # type: ignore
        self.pca_dim = pca_dim
        self.pca_model = PCA(n_components=self.pca_dim)
        self.pca1_mask_model = PCA(n_components=1)

        self.extract_masks = extract_masks
        self.weight_keys_with_masks = weight_keys_with_masks
        self.device = device
        self.extractor = FeatureMapExtractor(dino_model_type, device=device, num_patches=dino_model_num_patches, stride=(
            14 if 'dinov2' in dino_model_type else 4))

        self.args_str = f"{masks_method}_{dino_model_type}_patches-{dino_model_num_patches}_size-{img_size[0]}-{img_size[1]}_weightkeys-{weight_keys_with_masks}_pca-{pca_dim}"
        self.checkpoint_filename = f'preprocessor_checkpoint_{self.args_str}.pkl'
        self.trained = False
        self.masks_method = masks_method


    def _extract_images(self, data_folder: Path):
        images_folder_path = data_folder / 'images' if (data_folder / 'images').is_dir() else data_folder

        # Collect all the image files
        input_files = sorted(list(images_folder_path.glob("*.jpeg")) +
                             list(images_folder_path.glob("*.jpg")) +
                             list(images_folder_path.glob("*.png")))

        number_of_images = len(input_files)

        images = torch.zeros((number_of_images, 3, *self.img_size), device=self.device)
        for i in tqdm(range(number_of_images)):
            im_pil = Image.open(str(input_files[i])).convert("RGB").resize(self.img_size)
            im = transforms.ToTensor()(im_pil).to(self.device)
            images[i] = im

        return images, input_files


    def _extract_dino_coseg(self, img_paths):
        assert self.img_size[0] == self.img_size[1], "Supports only square images"
        masks, _ = find_cosegmentation(img_paths, load_size=self.img_size[0])
        masks = torch.tensor(masks).unsqueeze(1).float()
        return masks

    def _extract_dino_keys(self, images):
        extractor_emb_size: int = self.extractor.embedding_dim  # type: ignore
        number_of_images = images.shape[0]
        keys_before_pca = torch.zeros((number_of_images, extractor_emb_size, *self.img_size), device=self.device)

        for i, img in enumerate(tqdm(images)):
            im_keys = self.extractor.get_feature_map(img.unsqueeze(0), 'same').squeeze(0)  # [emb, h, w]
            keys_before_pca[i] = im_keys

        return keys_before_pca


    def train_and_process(self, data_folder: Path):
        ckpt_path = data_folder.joinpath(self.checkpoint_filename)

        if ckpt_path.exists():
            print(f"Loading preprocessor checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, weights_only=False)
            self.pca_model = checkpoint['pca_model']
            self.pca1_mask_model = checkpoint['pca1_mask_model']
            trained_data = checkpoint['train_data']
            self.trained = True
            return trained_data['images'], trained_data['keys'], trained_data['masks']

        else:
            print(f"Training preprocessor and saving checkpoint to {ckpt_path}")
            images, img_paths = self._extract_images(data_folder)          # [N, 3, H, W]

            # We find that the DINO and cosegmentation not responds well to non-square images / images padded with 0 padding 
            # Our solution is to stretch and unstretch the images
            images_stretched = images.clone()
            initial_masks = torch.ones((images.shape[0], 1, images.shape[2], images.shape[3]), device=self.device)
            images_stretched, initial_masks, contours_lst = stretch_images(images_stretched, initial_masks)
            keys_before_pca = self._extract_dino_keys(images_stretched)   
            keys_before_pca = unstretch_keys(keys_before_pca, contours_lst)
            number_of_images, extractor_emb_size, H, W = keys_before_pca.shape

            reshaped_keys_before_pca = keys_before_pca.permute(0, 2, 3, 1).reshape(-1, extractor_emb_size)  # [N*H*W, emb]
            if self.extract_masks:
                if self.masks_method == 'coseg':
                    masks = self._extract_dino_coseg(img_paths).to(images.device)
                else:
                    raise ValueError(f"Invalid masks_method: {self.masks_method}")
            else:
                masks = initial_masks

            transformed_keys = self.pca_model.fit_transform(reshaped_keys_before_pca).to(self.device)  # [N*H*W, pca_dim]
            transformed_keys = transformed_keys.reshape(number_of_images, H, W, self.pca_dim).permute(0, 3, 1, 2)

            if self.weight_keys_with_masks:
                transformed_keys = transformed_keys * masks

            assert transformed_keys.grad_fn is None, "keys should not have a grad_fn"

            self.trained = True

            torch.save({
                'train_data': {
                    'images': images,
                    'keys': transformed_keys,
                    'masks': masks,
                },
                'pca_model': self.pca_model,
                'pca1_mask_model': self.pca1_mask_model
            }, ckpt_path)

            return images, transformed_keys, masks

    def process(self, data_folder_path: Path):
        assert self.trained, "You must train the PCA model first"
        images, img_paths = self._extract_images(data_folder_path)     # [N, 3, H, W]
        keys_before_pca = self._extract_dino_keys(images)              # [N, emb, H, W]
        number_of_images, extractor_emb_size, H, W = keys_before_pca.shape

        reshaped_keys_before_pca = keys_before_pca.permute(0, 2, 3, 1).reshape(-1, extractor_emb_size)  # [N*H*W, emb]
        if self.extract_masks:
            if self.masks_method == 'coseg':
                masks = self._extract_dino_coseg(img_paths).to(images.device)
            else:
                raise ValueError(f"Invalid masks_method: {self.masks_method}")
        else:
            masks = torch.ones((number_of_images, 1, H, W),
                               requires_grad=False, device=self.device)

        transformed_keys = self.pca_model.transform(reshaped_keys_before_pca)  # [N*H*W, pca_dim]
        transformed_keys = transformed_keys.reshape(number_of_images, H, W, self.pca_dim).permute(0, 3, 1, 2)

        if self.weight_keys_with_masks:
            transformed_keys = transformed_keys * masks

        return images, transformed_keys, masks
