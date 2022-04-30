import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from latentgan_reverse.model import Generator, GeneratorReverse
from latentgan.dataset import ModelLatentDataset
from latentgan_reverse.config import *

model_latent_data = ModelLatentDataset(deepsdf_model_codes_path).data

num_codes = model_latent_data.shape[0]

model_generator = Generator(z_dim, hidden_dims_g, latent_size)
model_generator.load_state_dict(torch.load(generator_params_path))
model_generator = model_generator.eval().cuda()

model_generator_reverse_path = os.path.join("experiments", model_name, model_params_subdir)
model_generator_reverse = GeneratorReverse(latent_size, hidden_dims, z_dim)
model_generator_reverse.load_model(model_generator_reverse_path, epoch_load)
model_generator_reverse = model_generator_reverse.eval().cuda()

z_list = np.empty((2 * num_codes, latent_size))
z_list[:num_codes] = model_latent_data
z_list[num_codes:] = model_generator(model_generator_reverse(model_latent_data.cuda())).detach().cpu()

tsne = TSNE(n_components=2)
transformed = tsne.fit_transform(z_list)
plt.scatter(transformed[:num_codes, 0], transformed[:num_codes, 1], alpha=0.5, color="g")
plt.scatter(transformed[num_codes:, 0], transformed[num_codes:, 1], alpha=0.5, color="r")
plt.savefig(f"tsneAE_{model_name}.png")