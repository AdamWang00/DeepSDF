import torch

from latentgan_reverse.config import *


class ModelLatentSamplesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        generator,
        num_codes=num_training_codes
    ):
        self.data_z = torch.randn(num_codes, z_dim)
        self.data_latent = generator.generate(z=self.data_z)
        self.data_z = self.data_z.detach()
        self.data_latent = self.data_latent.detach()

    def __len__(self):
        return len(self.data_z)

    def __getitem__(self, idx):
        return self.data_z[idx], self.data_latent[idx]