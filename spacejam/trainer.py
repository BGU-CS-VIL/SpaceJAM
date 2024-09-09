
from pathlib import Path
from time import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from eval.pck import get_pck
from spacejam.keys_preprocessor import KeysPreprocessor

from spacejam.dataset import Dataset
from spacejam.loss import Loss
from spacejam.models.autoencoder import Autoencoder
from spacejam.models.inputs_reflector import InputsReflector
from spacejam.models.transformers.transformer import Transformer
from spacejam.atlas import Atlas, SingleWarpedImageAtlas
from utilities.run_utils import train_pca_on_features, forward_inference_all, visualize_training, seed_everything
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import logging
import os
import numpy as np

from utilities.training_utils import init_results_dir, init_stns, process_inputs_dict


def trainer_from_args(args: Dict[str, Any]):
    return Trainer(args)


def trainer_from_checkpoint(checkpoint_path: str):
    return Trainer({}, checkpoint_path=checkpoint_path)


class Trainer:
    def __init__(self, args: Dict[str, Any], checkpoint_path: Optional[str] = None):
        super().__init__()

        # If checkpoint path is provided, load the args from the checkpoint and use the old experiment dir
        if checkpoint_path is not None:  # Loading the args only for now
            if not os.path.isfile(checkpoint_path) or not checkpoint_path.endswith(".pt"):
                raise ValueError(f"Checkpoint path {checkpoint_path} is not a .pt file")
            
            logging.info(f"Loading checkpoint from {checkpoint_path}")
            logging.warning("any arguments passed will be ignored. Using args from checkpoint.")

            self.experiment_dir = os.path.dirname(checkpoint_path)
            self.args = args = torch.load(checkpoint_path)['args']
        else:   
            self.args = args
            self.experiment_dir = os.path.join(args['results_folder'], f'{datetime.now().strftime("%Y-%m-%d_%H:%M")}_{args["data_folder"].replace("/", "_")}_{args["run_name"] if "run_name" in args else ""}')
            init_results_dir(args, self.experiment_dir)

        seed_everything(self.args['seed'])
        
        self.device = self.args['device']
        self.keys_preprocessor = KeysPreprocessor(pca_dim=self.args["input_keys_pca_dim"], 
                                                  dino_model_type=self.args["dino_model_type"], 
                                                  dino_model_num_patches=self.args["dino_model_num_patches"],
                                                  device=self.device,
                                                  img_size=self.args["image_resolution"],
                                                  extract_masks=self.args["extract_masks"], 
                                                  weight_keys_with_masks=self.args["weight_keys_with_masks"],
                                                  masks_method=self.args["masks_method"])
        imgs, keys, masks = self.keys_preprocessor.train_and_process(Path(self.args["data_folder"]))
        self.keys_pca3_model, _, _ = train_pca_on_features(keys)  # model for visualizing the high dim keys with RGB

        self.dataset = Dataset(imgs, keys, masks)   
        self.data_loader = DataLoader(self.dataset, self.args["batch_size"], shuffle=True)
        
        self.high_dim_channels = self.dataset.keys_high_dim_size
        self.low_dim_channels = self.args['low_dim_keys']
        self.train_with_reflections = self.args["add_reflections"]
    
        self.loss_handler = Loss(self.args["error_loss"], self.args["weight_loss_with_masks"], self.args["weight_keys_with_masks"])
    
        self.autoencoder = Autoencoder(high_dim_emb=self.high_dim_channels, low_dim_emb=self.low_dim_channels).to(self.device)
        self.stn, self.action_observer = init_stns(self.args)

        print(f'Number of parameters in autoencoder: {sum(p.numel() for p in self.autoencoder.parameters())}, trainable: {sum(p.numel() for p in self.autoencoder.parameters() if p.requires_grad)}')
        print(f'Number of parameters in STN: {sum(p.numel() for p in self.stn.parameters())}, trainable: {sum(p.numel() for p in self.stn.parameters() if p.requires_grad)}')

        if self.train_with_reflections:
            self.inputs_reflector = InputsReflector(self.args, len(self.dataset), self.device)
        else:
            self.inputs_reflector = None
            
        with torch.no_grad():
            self.atlas_handler: Atlas = SingleWarpedImageAtlas(self.device, self.inputs_reflector)  # The congealing method considers different warped image as "atlas"

        if self.args["optimizer"] == "adam":
            self.optimizer_class = torch.optim.Adam
        elif self.args["optimizer"] == "sgd":
            self.optimizer_class = torch.optim.SGD
        else:
            raise ValueError(f"Unknown optimizer {self.args['optimizer']}")
        
        self.training_done = False

        if checkpoint_path is not None:
            self._load_model(checkpoint_path)



    def save_model(self):
        if not self.training_done:
            raise ValueError("Cannot save model before training is done")
        
        os.makedirs(self.experiment_dir, exist_ok=True)
        torch.save({ 
            'stn_state_dict': self.stn.state_dict(),
            'autoencoder_state_dict': self.autoencoder.state_dict(),
            'atlas': self.atlas_handler,
            'inputs_reflector': self.inputs_reflector,
            'args': self.args,
            }, 
            os.path.join(self.experiment_dir, "model_result.pt")
        )
        

    def _load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.stn.load_state_dict(checkpoint['stn_state_dict'])
        self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        self.atlas_handler = checkpoint['atlas']
        self.inputs_reflector = checkpoint['inputs_reflector']
        self.training_done = True
        
    def _get_augmentation_strength(self, epoch, num_epochs):
        return self.args["data_augmentation_std"] * (epoch / num_epochs) if self.args["train_data_augmentation"] else 0.0

        
    def pretrain_ae(self):
        optimizer = self.optimizer_class(self.autoencoder.parameters(), lr=self.args["ae_lr"], weight_decay=self.args["ae_weight_decay"])
        scheduler = StepLR(optimizer, step_size=self.args["lr_scheduler_step_size"], gamma=self.args["lr_scheduler_gamma"])
          
        num_pretrain_epochs = self.args["pretrain_ae_epochs"]
        losses = []
        
        progress_bar = tqdm(range(num_pretrain_epochs), desc="Pretraining AE", position=0, leave=True)
        for epoch in progress_bar:
            self.autoencoder.train()
            epoch_loss = 0.0
            augmentation_strength = self._get_augmentation_strength(epoch, num_pretrain_epochs)
            
            for inputs_dict in self.data_loader:
                inputs_dict = process_inputs_dict(inputs_dict, reflections=self.train_with_reflections, augmentation_strength=augmentation_strength, device=self.device)
                loss = self.loss_handler.reconstruction_loss(self.autoencoder, inputs_dict["keys"], inputs_dict["masks"], inputs_dict["frames"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # End epoch
            scheduler.step()
            epoch_loss /= len(self.data_loader)
            progress_bar.set_description(f"Pretrain AE Epoch {epoch+1}/{num_pretrain_epochs}", refresh=False)
            progress_bar.set_postfix({"Loss": f"{epoch_loss:.4f}"}, refresh=False)
            losses.append(epoch_loss)

            # Log and Visualize            
            self.autoencoder.eval()
        


    def train_batch_with_reflections_fast(self, inputs_dict: Dict[str, torch.Tensor], optimizer_stn, optimizer_ae):
        _, keys, masks, frames = inputs_dict["images"], inputs_dict["keys"], inputs_dict["masks"], inputs_dict["frames"]  # [B, Ref, C, H, W]
        indices = inputs_dict["current_im_idx"] # [B]
        
        assert keys.dim() == 5, f"Expected keys to have 5 dimensions, got shape {keys.shape}"
        assert self.inputs_reflector is not None, "inputs_reflector is None"

        batch_size, n_reflections = keys.shape[:2]                
    
        encoded_keys = self.autoencoder.encoder(keys.view(-1, *keys.shape[-3:])).view(batch_size, n_reflections, self.low_dim_channels, *keys.shape[-2:])  # [B, Ref, low_dim_channels, H, W]
        batch_transformer: Transformer = self.stn(encoded_keys.view(-1, *encoded_keys.shape[-3:]))  # Transformer for [B*Ref, low_dim_channels, H, W] tensors

        loss = torch.tensor(0.0, device=self.device)
        optimizer_stn.zero_grad()
        optimizer_ae.zero_grad()
        for i in range(batch_size):
            
            self.atlas_handler.update_atlas(batch_transformer, indices, keys, encoded_keys, masks, frames, index=i)
            atlas = self.atlas_handler.get_atlas()
            loss += self.loss_handler.loss_with_best_atlas_reflections(batch_transformer, self.autoencoder.decoder, atlas, keys, encoded_keys, masks, frames).mean() 
            
        loss.backward()
        optimizer_stn.step()
        optimizer_ae.step()
        
        return loss.item() / batch_size


    def train_batch_with_reflections(self, inputs_dict: Dict[str, torch.Tensor], optimizer_stn, optimizer_ae):
        _, keys, masks, frames = inputs_dict["images"], inputs_dict["keys"], inputs_dict["masks"], inputs_dict["frames"]  # [B, Ref, C, H, W]
        indices = inputs_dict["current_im_idx"] # [B]
        
        assert keys.dim() == 5, f"Expected keys to have 5 dimensions, got shape {keys.shape}"
        assert self.inputs_reflector is not None, "inputs_reflector is None"

        batch_size, n_reflections = keys.shape[:2]                
    
        encoded_keys = self.autoencoder.encoder(keys.view(-1, *keys.shape[-3:])).view(batch_size, n_reflections, self.low_dim_channels, *keys.shape[-2:])  # [B, Ref, low_dim_channels, H, W]
        batch_transformer: Transformer = self.stn(encoded_keys.view(-1, *encoded_keys.shape[-3:]))  # Transformer for [B*Ref, low_dim_channels, H, W] tensors

        total_loss = 0.0
        for i in range(batch_size):
            
            self.atlas_handler.update_atlas(batch_transformer, indices, keys, encoded_keys, masks, frames, index=i)
            atlas = self.atlas_handler.get_atlas()

            optimizer_stn.zero_grad()
            optimizer_ae.zero_grad()
            loss = self.loss_handler.loss_with_best_atlas_reflections(batch_transformer, self.autoencoder.decoder, atlas, keys, encoded_keys, masks, frames).mean() 
            loss.backward(retain_graph=True)
            optimizer_stn.step()
            optimizer_ae.step()
            
            total_loss += loss.item()
        return total_loss / batch_size



    def train_batch_without_reflections(self, inputs_dict: Dict[str, torch.Tensor], optimizer_stn, optimizer_ae):
        _, keys, masks, frames = inputs_dict["images"], inputs_dict["keys"], inputs_dict["masks"], inputs_dict["frames"]    # [B, C, H, W]
        indices = inputs_dict["current_im_idx"] # [B]

        assert keys.dim() == 4, f"Expected keys to have 4 dimensions, got shape {keys.shape}"

        encoded_keys = self.autoencoder.encoder(keys)            # [B, low_dim_channels, H, W]
        batch_transformer: Transformer = self.stn(encoded_keys)  # Transformer for [B, low_dim_channels, H, W] tensors
    
        total_loss = 0.0
        for i in range(len(indices)):
            self.atlas_handler.update_atlas(batch_transformer, indices, keys, encoded_keys, masks, frames, index=i)
            atlas = self.atlas_handler.get_atlas()
            
            optimizer_stn.zero_grad()
            optimizer_ae.zero_grad()
            loss = self.loss_handler.loss_without_reflections(batch_transformer, self.autoencoder.decoder, atlas, keys, encoded_keys, masks, frames).mean()
            loss.backward(retain_graph=True)
            optimizer_stn.step()
            optimizer_ae.step()
            
            total_loss += loss.item()
            
        return total_loss / len(indices)


    @torch.no_grad()
    def set_reflection_losses(self, epoch):
        # Not updating throughout the training because forward here should be without augmentations

        assert self.train_with_reflections and self.inputs_reflector is not None, "train_with_reflections is False or inputs_reflector is None - cannot set reflection losses"

        for inputs_dict in self.data_loader:
            inputs_dict = process_inputs_dict(inputs_dict, reflections=self.train_with_reflections, device=self.device)   
            _, keys, masks, frames = inputs_dict["images"], inputs_dict["keys"], inputs_dict["masks"], inputs_dict["frames"]  # [B, Ref, C, H, W]
            indices = inputs_dict["current_im_idx"] # [B]
            
            flat_encoded_keys = self.autoencoder.encoder(keys.view(-1, *keys.shape[-3:]))    # [B*Ref, low_dim_channels, H, W]
            batch_transformer: Transformer = self.stn(flat_encoded_keys)                     # Transformer for [B*Ref, low_dim_channels, H, W] tensors
            encoded_keys = flat_encoded_keys.view(*keys.shape[:2], self.low_dim_channels, *keys.shape[-2:])  # [B, Ref, low_dim_channels, H, W]

            loss_per_reflection = self.loss_handler.loss_first_atlas_reflection(batch_transformer, self.autoencoder.decoder, self.atlas_handler.get_atlas(), 
                                                                                keys, encoded_keys, masks, frames)  # [B, Ref]
            self.inputs_reflector.set_loss_values(epoch, indices, loss_per_reflection)


    @torch.no_grad()
    def end_epoch(self, epoch):
        self.action_observer.trigger_actions(epoch)

        # updates loss values history
        if self.train_with_reflections: 
            assert self.inputs_reflector is not None, "inputs_reflector is None when train_with_reflections=True"

            self.set_reflection_losses(epoch)    # tracks the loss values throughout the epochs (for determining the reflection configurations)
            
            if epoch % self.args["update_reflections_freq"] == 0 and 0.1 < epoch / self.args["training_epochs"] < 0.9:
                self.inputs_reflector.update_reflection_configurations(epoch, self.args["update_reflections_freq"])


    def train_stn(self):
        optimizer_stn = self.optimizer_class(self.stn.parameters(), lr=self.args["stn_lr"], weight_decay=self.args["stn_weight_decay"])
        optimizer_ae = self.optimizer_class(self.autoencoder.parameters(), lr=self.args["ae_lr"], weight_decay=self.args["ae_weight_decay"])
        scheduler_stn = StepLR(optimizer_stn, step_size=self.args["lr_scheduler_step_size"], gamma=self.args["lr_scheduler_gamma"])
        scheduler_ae = StepLR(optimizer_ae, step_size=self.args["lr_scheduler_step_size"], gamma=self.args["lr_scheduler_gamma"])

        num_training_epochs = self.args["training_epochs"]
        losses = []

        progress_bar = tqdm(range(num_training_epochs), desc="Training STN and AE", position=0, leave=True)
        for epoch in progress_bar:
            self.stn.train()
            self.autoencoder.train()
            epoch_loss = 0.0

            for inputs_dict in self.data_loader:
                inputs_dict = process_inputs_dict(inputs_dict, reflections=self.train_with_reflections, 
                                                  augmentation_strength=self._get_augmentation_strength(epoch, num_training_epochs), device=self.device)

                if self.train_with_reflections:
                    epoch_loss += self.train_batch_with_reflections_fast(inputs_dict, optimizer_stn, optimizer_ae)
                else:
                    epoch_loss += self.train_batch_without_reflections(inputs_dict, optimizer_stn, optimizer_ae)

            # End epoch
            epoch_loss /= len(self.data_loader)
            progress_bar.set_description(f"Train STN+AE Epoch {epoch+1}/{num_training_epochs}", refresh=False)
            progress_bar.set_postfix({"Loss": f"{epoch_loss:.4f}"}, refresh=False)
            losses.append(epoch_loss)        
            scheduler_stn.step()
            scheduler_ae.step()
            
            self.stn.eval()
            self.autoencoder.eval()
            self.end_epoch(epoch)
                        

    def train(self):
        self.pretrain_ae()
        self.train_stn()
        self.training_done = True
        self.save_model()
        

    @torch.no_grad()
    def get_pck_data(self, transformers, data_folder, alpha_list, images, encoded_keys, masks, max_pairs_to_return=1):
        result_dict = {}

        pck_folder = Path(data_folder, "pck")
        if pck_folder.exists():
            pck_scores, _ = get_pck(alpha_list, images, data_folder, transformers, max_pairs_to_return=max_pairs_to_return, mirror=True)

            for alpha in alpha_list:
                result_dict[f"pck/pck@{alpha}"] = pck_scores[alpha]['score']

        return result_dict


    def eval(self, log_vis=False):
        # returns the pck values for the train and test datasets
        self.autoencoder.eval()
        self.stn.eval()
        
        results = {}
        
        # Train dataset
        transformers_list, encoded_keys_list, inputs_dicts_list = forward_inference_all(self.dataset, self.autoencoder, self.stn, 
                                                                                        self.atlas_handler, self.loss_handler, train_with_reflections=self.train_with_reflections)

        pck_results_train = self.get_pck_data(transformers_list, data_folder=self.args["data_folder"], alpha_list=[0.05, 0.1], 
                                              images=torch.cat([inputs_dict["images"] for inputs_dict in inputs_dicts_list], dim=0),
                                              encoded_keys=torch.cat(encoded_keys_list, dim=0), 
                                              masks=torch.cat([inputs_dict["masks"] for inputs_dict in inputs_dicts_list], dim=0))
        results.update({f"train/{key}": value for key, value in pck_results_train.items()})
        
        if log_vis:
            transformers_list, encoded_keys_list, inputs_dicts_list = forward_inference_all(self.dataset, self.autoencoder, self.stn, 
                                                                                self.atlas_handler, self.loss_handler, train_with_reflections=self.train_with_reflections,
                                                                                batch_size=8)


            for i, (transformer, encoded_key, input_dict) in enumerate(zip(transformers_list, encoded_keys_list, inputs_dicts_list)):
                visualize_training(input_dict, transformer, encoded_key, self.atlas_handler.get_atlas(), self.autoencoder, epoch=self.args["training_epochs"], save_path=self.experiment_dir + f'/res_{i}.png')

        return results
            