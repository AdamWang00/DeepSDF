import torch.utils.data as data_utils
from latentgan.model import WGAN_GP
from latentgan.dataset import ModelLatentDataset
from latentgan.config import *
from latentgan_reverse.model import GeneratorReverse
from latentgan_reverse.dataset import ModelLatentSamplesDataset
from latentgan_reverse.config import *

model_latent_dataset = ModelLatentDataset(deepsdf_model_codes_path)

train_loader_gan = data_utils.DataLoader(
    model_latent_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

model_gan = WGAN_GP()

model_generator = model_gan.G

model_latent_samples_dataset = ModelLatentSamplesDataset(model_generator, detach=False, generate_once=False)

train_loader_reverse = data_utils.DataLoader(
    model_latent_samples_dataset,
    batch_size=batch_size,
    shuffle=True,
    # num_workers=1,
    drop_last=True,
)

model_generator_reverse = GeneratorReverse(latent_size, hidden_dims, z_dim).cuda()

model_gan_trainer = model_gan.train_yield(generator_iters, train_loader_gan)
model_generator_reverse_trainer = model_generator_reverse.train_yield(epochs, train_loader_reverse)

generator_iters_per_epoch = generator_iters // epochs
for i in range(epochs):
	for i in range(generator_iters_per_epoch):
		next(model_gan_trainer)
	next(model_generator_reverse_trainer)