import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time as t
import os

from latentvae.config import *

def compute_kld_loss(mu, log_var):
    """
    mu: (N, z_dim)
    log_var: (N, z_dim)
    """
    assert len(mu.shape) == 2
    assert mu.shape == log_var.shape
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

class Encoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dims, z_dim):
        super().__init__()
        dims = [in_dim] + hidden_dims
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.ReLU(True))
        self.main_module = nn.Sequential(*modules)

        self.lin_mu = nn.Linear(dims[-1], z_dim)
        self.lin_var = nn.Linear(dims[-1], z_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.training: # randomize during training, use mu during evaluation
            eps = torch.randn_like(std)
        else:
            eps = torch.zeros_like(std)
        return eps * std + mu

    def forward(self, x):
        x = self.main_module(x)
        mu = self.lin_mu(x)
        log_var = self.lin_var(x)
        x = self.reparameterize(mu, log_var)
        return x, mu, log_var

    def save_model(self, save_dir, epoch):
        torch.save(self.state_dict(), os.path.join(save_dir, str(epoch) + "_e.pth"))

    def load_model(self, load_dir, epoch):
        filepath = os.path.join(load_dir, str(epoch) + "_e.pth")
        self.load_state_dict(torch.load(filepath))
        print('encoder model loaded from {}.'.format(filepath))


class Decoder(torch.nn.Module):
    def __init__(self, z_dim, hidden_dims, out_dim):
        super().__init__()
        self.z_dim = z_dim
        dims = [z_dim] + hidden_dims
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dims[-1], out_dim))
        self.main_module = nn.Sequential(*modules)

    def forward(self, x):
        return self.main_module(x)

    def generate(self, num_codes=1, z=None):
        if z == None:
            z = torch.randn(num_codes, self.z_dim)
        samples = self.forward(z.cuda()).detach()
        return samples
    
    def save_model(self, save_dir, epoch):
        torch.save(self.state_dict(), os.path.join(save_dir, str(epoch) + "_d.pth"))

    def load_model(self, load_dir, epoch):
        filepath = os.path.join(load_dir, str(epoch) + "_d.pth")
        self.load_state_dict(torch.load(filepath))
        print('decoder model loaded from {}.'.format(filepath))


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(latent_size, hidden_dims_e, z_dim)
        self.decoder = Decoder(z_dim, hidden_dims_d, latent_size)

        self.save_dir = os.path.join("experiments", model_name, model_params_subdir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.learning_rate = learning_rate
        self.kld_loss_weight = kld_loss_weight
        if batch_size == None:
            self.batch_size = 64
        else:
            self.batch_size = batch_size

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=step_gamma)

    def forward(self, x):
        x, mu, log_var = self.encoder(x)
        x = self.decoder(x)
        return x, mu, log_var
    
    # def reconstruct(self, x):
    #     _, mu, _ = self.encoder(x)
    #     x = self.decoder(mu)
    #     return x

    def train_(self, epochs, train_loader):
        t_begin = t.time()

        losses = []
        kld_losses = []
        for epoch in range(1, epochs + 1):
            print("Epoch", epoch)
            loss_epoch = 0
            kld_loss_epoch = 0
            for codes in train_loader:
                codes = codes.cuda()
                self.optimizer.zero_grad()

                codes_reconstructed, mu, log_var = self.forward(codes)

                loss = F.mse_loss(codes_reconstructed, codes)
                loss_epoch += loss.item()

                kld_loss = self.kld_loss_weight * compute_kld_loss(mu, log_var)
                kld_loss_epoch += kld_loss.item()

                total_loss = loss + kld_loss

                total_loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            losses.append(loss_epoch)
            kld_losses.append(kld_loss_epoch)

            if epoch % save_per_epochs == 0:
                self.save_model(epoch)
                torch.save(
                    {
                        "loss": losses,
                        "kld_loss": kld_losses
                    },
                    os.path.join("experiments", model_name, "Logs.pth")
                )

        t_end = t.time()
        print('Time of training: {}'.format((t_end - t_begin)))

        # Save the trained parameters
        self.save_model("latest")

        torch.save(
            {
                "loss": losses,
                "kld_loss": kld_losses
            },
            os.path.join("experiments", model_name, "Logs.pth")
        )

    def generate(self, num_codes=1, z=None):
        return self.decoder.generate(num_codes=num_codes, z=z)

    def save_model(self, epoch):
        self.encoder.save_model(self.save_dir, epoch)
        self.decoder.save_model(self.save_dir, epoch)

    def load_model(self, load_dir, epoch):
        self.encoder.load_model(load_dir, epoch)
        self.decoder.load_model(load_dir, epoch)
    
    # def cuda(self):
    #     self.encoder = self.encoder.cuda()
    #     self.decoder = self.decoder.cuda()

    # def eval(self):
    #     self.encoder = self.encoder.eval()
    #     self.decoder = self.decoder.eval()