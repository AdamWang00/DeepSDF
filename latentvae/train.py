import torch.utils.data as data_utils
from latentvae.model import VAE
from latentvae.dataset import ModelLatentDataset
from latentvae.config import *

model_latent_dataset = ModelLatentDataset(deepsdf_model_codes_path)

train_loader = data_utils.DataLoader(
    model_latent_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

model_vae = VAE()
model_vae = model_vae.train().cuda()
model_vae.train_(epochs, train_loader)