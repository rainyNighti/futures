import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SupervisedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, code_dim, target_dim, noise_std=0.1):
        super().__init__()
        self.noise = GaussianNoise(noise_std)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, code_dim),
            Swish()
        )
        # Decoder reconstructs input
        self.decoder = nn.Sequential(
            nn.Linear(code_dim + target_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x, y, is_inference=False):
        if not is_inference:
            x = self.noise(x)
        code = self.encoder(x)
        # supervised: concat code and target
        if not is_inference:
            code_y = torch.cat([code, y], dim=1)
            recon = self.decoder(code_y)
        else:
            recon = None
        return code, recon

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

class AE_MLP_Model(nn.Module):
    def __init__(self, feat_dim, target_dim, ae_hidden, ae_code, mlp_hidden, dropout, noise_std):
        super().__init__()
        self.ae = SupervisedAutoencoder(feat_dim, ae_hidden, ae_code, target_dim, noise_std)
        self.mlp = MLP(feat_dim + ae_code, mlp_hidden, dropout)

    def forward(self, x, y=None, is_inference=False):
        code, recon = self.ae(x, y, is_inference)
        mlp_in = torch.cat([x, code], dim=1)
        out = self.mlp(mlp_in)
        return out, recon

# 损失函数
def loss_fn(pred, target, recon, x, alpha=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    recon_loss = F.mse_loss(recon, x)
    return (1-alpha) * bce + alpha * recon_loss, bce
