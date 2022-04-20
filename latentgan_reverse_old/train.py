import torch.utils.data as data_utils
from latentgan_reverse.model import GeneratorAE
from latentgan_reverse.dataset import ModelLatentDataset
from latentgan_reverse.config import *

model_latent_dataset = ModelLatentDataset(deepsdf_model_codes_path)

train_loader = data_utils.DataLoader(
    model_latent_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

model_ae = GeneratorAE()
model_ae.load_decoder_from_generator(generator_params_path)
model_ae.train(epochs, train_loader)