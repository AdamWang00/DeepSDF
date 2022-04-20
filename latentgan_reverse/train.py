import torch
import torch.utils.data as data_utils
from latentgan_reverse.model import Generator, GeneratorReverse
from latentgan_reverse.dataset import ModelLatentSamplesDataset
from latentgan_reverse.config import *

model_generator = Generator(z_dim, hidden_dims_g, latent_size)
model_generator.load_state_dict(torch.load(generator_params_path))
model_generator = model_generator.eval().cuda()

model_latent_samples_dataset = ModelLatentSamplesDataset(model_generator)

train_loader = data_utils.DataLoader(
    model_latent_samples_dataset,
    batch_size=batch_size,
    shuffle=True,
    # num_workers=1,
    drop_last=True,
)

model_generator_reverse = GeneratorReverse(latent_size, hidden_dims, z_dim).cuda()
model_generator_reverse.train_(epochs, train_loader)