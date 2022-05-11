import torch.utils.data as data_utils
from latentgan.model import WGAN_GP
from latentgan.dataset import ModelLatentDataset
from latentgan.config import *

model_latent_dataset = ModelLatentDataset(deepsdf_model_codes_path)

BATCH_SIZE = batch_size
GENERATOR_ITERS = generator_iters # int(NUM_EPOCHS * model_latent_dataset.__len__() / BATCH_SIZE / 5)

train_loader = data_utils.DataLoader(
    model_latent_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

model_gan = WGAN_GP()
model_gan.train_(GENERATOR_ITERS, train_loader)