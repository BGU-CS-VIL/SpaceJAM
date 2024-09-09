import torch
from torch import nn

class GlobalFeatureMapNormalizer(nn.Module):
    def __init__(self, eps=1e-8):
        super(GlobalFeatureMapNormalizer, self).__init__()
        self.eps = eps  
    
    def forward(self, x):
        mean = x.mean()
        std = x.std() + self.eps
        return (x - mean) / std


class Encoder(nn.Module):
    def __init__(self, high_dim_emb, low_dim_emb):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(high_dim_emb, 32, 1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, low_dim_emb, 1, padding=0),
            GlobalFeatureMapNormalizer())
        
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, high_dim_emb, low_dim_emb):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(low_dim_emb, 16, 1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, high_dim_emb, 1, padding=0)
        )
        
    def forward(self, x):
        return self.net(x)


class Autoencoder(nn.Module):
    def __init__(self, high_dim_emb, low_dim_emb):
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder(high_dim_emb, low_dim_emb)
        self.decoder = Decoder(high_dim_emb, low_dim_emb)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
