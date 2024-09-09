from collections import defaultdict
import os
from datetime import datetime
from typing import Dict, List, Tuple
from collections.abc import Callable
import torch
import torch.nn as nn
import yaml
from spacejam.models.inputs_reflector import InputsReflector

# from src.models.stns.cpab_stn import CpabSTN
from spacejam.models.stns.basic_stns import LieHomographySTN
from spacejam.models.stns.recurrent_stn import RecurrentSTN
from spacejam.models.stns.sequence_stn import SequenceSTN
from spacejam.models.stns.wrapper_stn import WrapperSTN
from spacejam.models.transformers.homography_transformer import HomographyTransformer


class ActionObserver:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.actions_per_epoch: Dict[int, List[Tuple[str, Callable]]] = defaultdict(list)  # epoch -> [(name, action)]

    def trigger_actions(self, current_epoch: int):
        for name, action in self.actions_per_epoch[current_epoch]:
            print(f"Triggering action {name} at epoch {current_epoch}")
            action()

    def register_action(self, name: str, time: float, action: Callable):
        epoch = int(time * self.total_epochs)
        self.actions_per_epoch[epoch].append((name, action))


def init_results_dir(args, experiment_dir):
    os.makedirs(experiment_dir, exist_ok=True)

    with open(os.path.join(experiment_dir, "args.yaml"), "w") as f:
        yaml.dump(args, f)


    with open(os.path.join(experiment_dir, "readme.txt"), "w") as f:
        f.writelines([
            f"This folder contains the results of the experiment: {args['run_name']}\n",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"Initial config is saved in args.yaml\n",
        ])


def init_stns(config) -> Tuple[nn.Module, ActionObserver]:
    '''
    Note - The implementation might seem overkill...
    The motivation is to support easily sequence of STNs (in addition to the "recurrent feeding" and "curriculum learning" that supported) with the ability of:
    1. "Unlocking" STNs during the training and concatenate it dynamically during the training
    2. "Freeze" some of STN's weights
    e.g. you can train LieHomography for coarse alignment, then freeze it and add another (probably more expressive) STN for fine alignment. 
    For this you will have to implement new STN of your choice (transformations must be invertible)
    '''
    low_dim_channels = config['low_dim_keys']
    action_observer = ActionObserver(config["training_epochs"])
    device = config["device"]

    # ------ SETUP METHODS FOR BASIC STNS ------ #
    def setup_lie_homography_stn():
        # The homography STN will be used for all the transformations less general than it, for supporting switching between them during training.
        # action_observer should be triggered on "update_model" once every epoch.
        stn = LieHomographySTN(
            config, low_dim_channels, initial_transformation=config["initial_transformation"])
        for time, transformation_type in config["actions_transformation"]:
            action_observer.register_action(f"set_transformation_type_to_{transformation_type}",
                                            time,
                                            lambda transformation_type=transformation_type: stn.set_transformation_type(transformation_type))  # Note the default value is MUST (for capturing the current value of transformation_type)

        if config["recurrent_n_warps"] > 1:
            return RecurrentSTN(stn, config["recurrent_n_warps"], avoid_interpolations=True)
        else:
            return stn
    # ------ SETUP METHODS FOR BASIC STNS ------ #


    # ------ SETUP METHODS FOR COMPLEX STN SEQUENCE ------ #
    def setup_stns_only_lie_homography() -> nn.ModuleList: 
        lie_hom_stn = WrapperSTN(setup_lie_homography_stn(),
                                 active=True, trainable=True)
        stns = nn.ModuleList([lie_hom_stn])
        return stns
    # ------ SETUP METHODS FOR COMPLEX STN SEQUENCE ------ #

    
    stn = SequenceSTN(setup_stns_only_lie_homography()).to(device) 
    return stn, action_observer


def process_inputs_dict(inputs_dict: Dict[str, torch.Tensor], reflections=True, augmentation_strength=0.0, device=None) -> Dict[str, torch.Tensor]:
    # Return: dict of "images" (B, C, H, W), "keys" (B, C, H, W), "masks" (B, 1, H, W), "frames" (B, 1, H, W), "current_im_idx" (B) - if reflections=False
    # If reflections=True then the tensors are of shape (B, reflections, C, H, W)
    # All tensors moved to device if device is not None.
    inputs_dict["frames"] = torch.ones_like(inputs_dict["masks"])

    if device is not None:
        inputs_dict = {key: value.to(device) for key, value in inputs_dict.items()}
    else:
        device = inputs_dict["images"].device

    if augmentation_strength > 0: # affine augmentation in lie algebra    
        theta = torch.normal(mean=0, std=augmentation_strength,
                             size=(inputs_dict["images"].shape[0], 3, 3), device=device)
        theta[:, 2, :] = 0

    for key in ["images", "keys", "masks", "frames"]:
        if augmentation_strength > 0:
            # type: ignore (theta is bounded var here)
            transformer = HomographyTransformer(inputs_dict[key].shape[-2:], theta, lie_algebra=True)
            inputs_dict[key] = transformer(inputs_dict[key])

        if reflections:
            inputs_dict[key] = InputsReflector.apply_reflection(inputs_dict[key], dim=1)  # [batch_size, n_reflections, C, H, W]

    return inputs_dict


# Used for training with reflections
def select_first_inputs_config(inputs_dict: Dict[str, torch.Tensor]):
    return {key: value[:, 0] if value.dim() == 5 else value for key, value in inputs_dict.items()}

# Used for training with reflections
def select_inputs_config(inputs_dict: Dict[str, torch.Tensor], config):
    indices = inputs_dict["current_im_idx"]
    return {key: value[torch.arange(len(indices), device=indices.device), config] if value.dim() == 5 else value for key, value in inputs_dict.items()}
