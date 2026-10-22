import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop
import numpy as np


class RiskHead(nn.Module): 
    """
    Predicts default risk from latent borrower representation.
    Input:
        mu: posterior mean with shape [batch_size, latent_dim]
    Output:
        logits: [batch_size]
    """

    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, mu):
        logits = self.net(mu)
        return logits.squeeze(1)

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        conv_out_dim = 64 * input_dim  

        self.fc_mu = nn.Sequential(
            nn.Linear(conv_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(conv_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1) 
        h = self.conv(x)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class Generator(nn.Module):
    def __init__(self, input_dim, gside=19, num_channels=32):
        super().__init__()
        self.init_dim = (num_channels * 4, 5, 5)
        self.fc = nn.Linear(input_dim, int(np.prod(self.init_dim)))

        self.deconv = nn.Sequential(
            nn.BatchNorm2d(num_channels * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels * 4, num_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(num_channels * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels * 2, 1, 4, 2, 1),
        )
        self.target_size = gside

    def forward(self, z):
        x = self.fc(z).view(-1, *self.init_dim)
        x = self.deconv(x)
        x = center_crop(x, output_size=(self.target_size, self.target_size))
        return x


class Discriminator(nn.Module):
    def __init__(self, dside, num_channels=32):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, num_channels, 4, 2, 1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_channels, num_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(num_channels * 2),
            nn.LeakyReLU(0.2),
        )
        self.out = nn.Conv2d(num_channels * 2, 1, dside // 4, 1, 0)  

    def forward(self, x):
        h = self.feature(x)
        out = self.out(h)
        return out.view(-1, 1), h.view(h.size(0), -1)  


class Risk_VAE_CTABGAN(nn.Module):
    def __init__(self, embedding_dim, z_dim, device, batch_size, lr, sample_dir, checkpoint_dir):
        super(Risk_VAE_CTABGAN, self).__init__()
        self.embedding_dim = embedding_dim
        self.z_dim = z_dim
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.sample_dir = sample_dir
        self.checkpoint_dir = checkpoint_dir

    def save_checkpoint(self, encoder, generator, discriminator, epoch, risk_head=None):
        save_path = f"{self.checkpoint_dir}/risk_vae_ctabgan_epoch{epoch}.pth"
        checkpoint = {
            'encoder': encoder.state_dict(),
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict()
        }
        if risk_head is not None:
            checkpoint['risk_head'] = risk_head.state_dict()
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
