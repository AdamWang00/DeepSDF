import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from latentgan.model import WGAN_GP
from latentgan.dataset import ModelLatentDataset
from latentgan.config import *

num_examples = 10000

model_latent_data = ModelLatentDataset(deepsdf_model_codes_path).data
num_codes = model_latent_data.shape[0]

gan_load_path = os.path.join("experiments", model_name, model_params_subdir)
model_gan = WGAN_GP()
model_gan.load_model(gan_load_path, iter_load)
model_gan.eval()

z_list = np.empty((num_codes + num_examples, latent_size))
z_list[:num_codes] = model_latent_data.numpy()
z_list[num_codes:] = model_gan.generate(num_codes=num_examples).cpu().numpy()

tsne = TSNE(n_components=2)
transformed = tsne.fit_transform(z_list)
plt.scatter(transformed[:num_codes, 0], transformed[:num_codes, 1], alpha=0.5, color="g")
plt.scatter(transformed[num_codes:, 0], transformed[num_codes:, 1], alpha=0.5, color="r")
plt.savefig(f"tsne_{model_name}.png")