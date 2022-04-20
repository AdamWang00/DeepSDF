import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from latentgan_reverse.model import GeneratorAE
from latentgan_reverse.dataset import ModelLatentDataset
from latentgan_reverse.config import *

model_latent_data = ModelLatentDataset(deepsdf_model_codes_path).data
num_latent = model_latent_data.shape[0]

gan_load_path = os.path.join("experiments", model_name, model_params_subdir)
model_gan = GeneratorAE()
model_gan.load_model(gan_load_path, epoch_load)
model_gan.eval()

z_list = np.empty((num_latent, 256))
z_list[:num_latent] = model_gan.encoder.forward(model_latent_data.cuda()).detach().cpu().numpy()
# z_list[num_latent:] = model_latent_data.numpy()

tsne = TSNE(n_components=2)
transformed = tsne.fit_transform(z_list)
plt.scatter(transformed[:, 0], transformed[:num_latent, 1], alpha=0.5, color="r")
# plt.scatter(transformed[num_latent:, 0], transformed[num_latent:, 1], alpha=0.5, color="y")
plt.savefig(f"tsne1_{model_name}.png")