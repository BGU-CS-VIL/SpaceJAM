import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputsReflector:
    """The class is responsible for handling the state of the reflections and the loss values, and reflecting the inputs of the model.
    """

    def __init__(self, config, n_images, device):
        super().__init__()
        
        self.device = device
        self.n_images = n_images
        self.n_reflections = 2 
        self.reflection_configurations = torch.zeros(n_images, dtype=torch.long, device=self.device)
        self.loss_values_history = torch.zeros(config["training_epochs"], self.n_images, self.n_reflections, dtype=torch.float32, device=self.device) - 1  # -1 means that the value is not initialized yet
        

    def get_reflection_configurations(self, indices):
        return self.reflection_configurations[indices]


    def update_reflection_configurations(self, epoch, window_size):
        """selecting the reflection configuration with the lowest loss within the window until the given epoch"""
        assert torch.all(self.loss_values_history[:epoch+1] != -1), f"Not all loss values until epoch {epoch} are initialized yet. losses: {self.loss_values_history[:epoch+1]}"
        self.reflection_configurations = torch.argmin(self.loss_values_history[epoch-window_size+1:epoch+1].mean(dim=0), dim=1)
        print(f"Updated reflection configurations: {self.reflection_configurations}")
        


    def set_loss_values(self, epoch: int, indices: torch.Tensor, loss_values: torch.Tensor):
        # indices: [batch_size]
        # loss_values: [batch_size, n_reflections]
        self.loss_values_history[epoch, indices] = loss_values
            

    @staticmethod
    def apply_reflection(x, dim=0):
        """input: tensor of some shape [..., h, w]
        output: (2, ..., h, w) tensor (when using dim=0), where the first dimension is the reflection index
        """
        return torch.stack(
            [
                x,                                # no reflection
                x.flip(-1),                       # reflect along x axis

                # More can be used for finding smaller transformation:
                # x.flip(-2),                       # reflect along y axis
                # x.flip(-2, -1),                   # reflect along both axes
                # x.transpose(-2, -1),              # reflect diagonal
                # x.transpose(-2, -1).flip(-2, -1), # reflect anti-diagonal
            ], dim=dim
        )