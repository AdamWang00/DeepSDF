import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from latentvae.model import VAE
from latentvae.dataset import ModelLatentDataset
from latentvae.config import *

model_latent_data = ModelLatentDataset(deepsdf_model_codes_path).data
num_codes = model_latent_data.shape[0]

model_vae_path = os.path.join("experiments", model_name, model_params_subdir)
model_vae = VAE()
model_vae.load_model(model_vae_path, epoch_load)
model_vae = model_vae.eval().cuda()

latent_list = np.empty((2 * num_codes, latent_size))
latent_list[:num_codes] = model_latent_data
latent_list[num_codes:] = model_vae(model_latent_data.cuda())[0].detach().cpu()

tsne = TSNE(n_components=2)
transformed = tsne.fit_transform(latent_list)
plt.scatter(transformed[:num_codes, 0], transformed[:num_codes, 1], alpha=0.5, color="g")
plt.scatter(transformed[num_codes:, 0], transformed[num_codes:, 1], alpha=0.5, color="r")
plt.savefig(f"tsneAE_{model_name}.png")