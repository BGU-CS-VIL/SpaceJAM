

import torch
import torch.nn as nn

from spacejam.models.transformers.homography_transformer import HomographyTransformer
from spacejam.models.transformers.transformer import Transformer
from abc import ABC, abstractmethod

class STN(ABC, nn.Module):
    @abstractmethod
    def forward(self, input: torch.Tensor) -> Transformer:
        pass


class LieHomographySTN(STN):
    """
    Supported actions:
    ------------------

    translation -   [[0, 0, a13], 
                    [0, 0, a23], 
                    [0, 0, 1  ]]  
                    OR
                e([[0, 0, A13], 
                    [0, 0, A23], 
                    [0, 0, 0  ]])
    Degrees of freedom: 2

    rigid -         [[cos(t), -sin(t), a13], 
                    [sin(t),  cos(t), a23], 
                    [0,       0,      1  ]] 
                    OR 
                e([[0,    A12, A13], 
                    [-A12, 0,   A23], 
                    [0,    0,   0  ]])
    Degrees of freedom: 3

    similarity -    [[s*cos(t), -s*sin(t), a13], 
                    [s*sin(t), s*cos(t),  a23], 
                    [0,        0,         1  ]] 
                    OR 
                e([[A11,  A12, A13], 
                    [-A12, A11, A23], 
                    [0,    0,   0  ]])
    Degrees of freedom: 4

    affine -        [[a11, a12, a13], 
                    [a21, a22, a23], 
                    [0,   0,   1  ]] 
                    OR
                e([[A11, A12, A13], 
                    [A21, A22, A23], 
                    [0,   0,   0  ]])
    Degrees of freedom: 6

    homography -    [[a11, a12, a13], 
                    [a21, a22, a23], 
                    [a31, a32, a33]] with det = 1 
                    OR
                e([[A11, A12, A13       ], 
                    [A21, A22, A23       ], 
                    [A31, A32, -(A11+A22)]])
    Degrees of freedom: 8

    Note: If you are not familiar with sl(3) and using the exponential matrix for transformations, see the paper 
    """
    def __init__(self, config, embedding_size, initial_transformation: str):
        super().__init__()

        in_channels = embedding_size
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=7),
            nn.AdaptiveMaxPool2d((32, 32)),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 5, kernel_size=5),
            nn.AdaptiveMaxPool2d((8, 8)),
            nn.ReLU(inplace=True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(5 * 8 * 8, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 9)
        )

        self.fc_loc[-1].weight.data.zero_()  # type: ignore
        self.fc_loc[-1].bias.data.zero_()   # type: ignore

        self._transformation_type = initial_transformation

    def get_extra_state(self):
        return {"transformation_type": self._transformation_type}

    def set_extra_state(self, state: dict):
        self._transformation_type = state["transformation_type"]

    def set_transformation_type(self, transformation_type):
        self._transformation_type = transformation_type

    def set_transformation_constrains_(self, theta: torch.Tensor) -> torch.Tensor:
        """Changes the theta [B, 3, 3] matrix to fit the constrains of the current transformation type.
        The changes are done inplace.

        Args:
            theta (torch.Tensor): [B, 3, 3] matrix of the transformation BEFORE matrix exponent

        Returns:
            torch.Tensor: the parameter theta (although done inplace)
        """

        if self._transformation_type == "translation":
            theta[:, :, :2] = 0
            theta[:, 2, 2] = 0
        elif self._transformation_type == "rigid":
            theta[:, 0, 0] = theta[:, 1, 1] = 0
            theta[:, 2, :] = 0
            # making sure a12 = -a21 and still use both thetas gradients
            avg_a12_a21 = (theta[:, 0, 1] + theta[:, 1, 0]) / 2
            theta[:, 0, 1] -= avg_a12_a21
            theta[:, 1, 0] -= avg_a12_a21
        elif self._transformation_type == "similarity":
            theta[:, 2, :] = 0
            # making sure a11 = a22 and still use both thetas gradients
            avg_a11_a22 = (theta[:, 0, 0] + theta[:, 1, 1]) / 2
            theta[:, 0, 0] = theta[:, 1, 1] = avg_a11_a22
            # making sure a12 = -a21 and still use both thetas gradients
            avg_a12_a21 = (theta[:, 0, 1] + theta[:, 1, 0]) / 2
            theta[:, 0, 1] -= avg_a12_a21
            theta[:, 1, 0] -= avg_a12_a21
        elif self._transformation_type == "affine":
            theta[:, 2, :] = 0
        elif self._transformation_type == "homography":
            # making sure det = 1 for all transformations
            theta[:, 2, 2] = -(theta[:, 0, 0] + theta[:, 1, 1])

        

        return theta

    def forward(self, input: torch.Tensor) -> Transformer:
        """
        :param input: torch.Tensor of shape [B, C, H, W]
        :return: transformer object
        """

        # input size [B, C, H, W]
        assert len(input.size()) == 4, "img must be of shape (N, C, H, W)"
        theta = self.fc_loc(self.localization(
            input).reshape(input.shape[0], -1))
        assert len(theta.size()) == 2 and theta.size(
            1) == 9, "theta must be of shape (N, 9)"

        # Theta is from the Lie Algebra sl(3), so matrices has trace = 0, and the matrix exponential (from the special linear group 3) has det = 1
        theta = theta.reshape(-1, 3, 3)

        self.set_transformation_constrains_(theta)

        return HomographyTransformer(input.shape[-2:], theta, lie_algebra=True)
