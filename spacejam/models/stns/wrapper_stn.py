import torch.nn as nn

class WrapperSTN(nn.Module):
    def __init__(self, stn, active=True, trainable=True):
        super(WrapperSTN, self).__init__()
        self.stn = stn
        self.active = active
        self.trainable = trainable
        self._set_trainability()

    def _set_trainability(self):
        for param in self.stn.parameters():
            param.requires_grad = self.trainable

    def set_state(self, active: bool, trainable: bool):
        self.active = active
        if self.trainable != trainable:
            self.trainable = trainable
            self._set_trainability()

    def forward(self, x):
        if not self.active:
            raise RuntimeError("STN is not active")
        return self.stn(x)

    def __getattr__(self, name):
        try:
            # Get attribute from the parent class (nn.Module)
            return super(WrapperSTN, self).__getattr__(name)
        except AttributeError:
            # If not found, try to fetch from the wrapped module
            return getattr(self.stn, name)