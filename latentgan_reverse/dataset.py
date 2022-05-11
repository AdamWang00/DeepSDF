import torch

from latentgan_reverse.config import *


class ModelLatentSamplesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        generator,
        num_codes=num_training_codes,
        detach=True,
        generate_once=True,
    ):
        self.num_codes = num_codes
        self.detach = detach
        
        if generate_once:
            self.data_z = torch.randn(num_codes, z_dim).cuda()
            self.data_latent = generator(self.data_z)
            if detach:
                self.data_z = self.data_z.detach()
                self.data_latent = self.data_latent.detach()
        else:
            self.generator = generator
            self.data_z = None
            self.data_latent = None

    def __len__(self):
        return self.num_codes

    def __getitem__(self, idx):
        if self.data_z == None:
            z = torch.randn(z_dim).cuda()
            latent = self.generator(z.unsqueeze(0)).squeeze()
            if self.detach:
                z = z.detach()
                latent = latent.detach()
            return z, latent
        else:
            return self.data_z[idx], self.data_latent[idx]