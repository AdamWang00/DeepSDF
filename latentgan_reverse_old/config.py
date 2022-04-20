import json
import os

model_name = "nightstand2"
epoch_load = "10000"
print("latentgan_reverse", model_name, epoch_load)

save_per_epochs = 10000

model_params_subdir = "ModelParameters"
model_generations_subdir = "Generations"

params_history = {
    "nightstand1": {
        "gan_model_name": "nightstand1",
        "gan_iter_load": 200000,
        "epochs": 10001,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "hidden_dims_e": [512, 512, 512, 512],
    },
    "nightstand2": {
        "gan_model_name": "nightstand1",
        "gan_iter_load": 200000,
        "epochs": 10001,
        "batch_size": 64,
        "learning_rate": 0.00003,
        "hidden_dims_e": [512, 512, 512, 512],
    },
}

params = params_history[model_name]
gan_model_name = params["gan_model_name"]
gan_iter_load = params["gan_iter_load"]
epochs = params["epochs"]
batch_size = params["batch_size"]
learning_rate = params["learning_rate"]
hidden_dims_e = params["hidden_dims_e"]

from latentgan.config import params_history as gan_params_history

gan_params = gan_params_history[gan_model_name]

deepsdf_model_name = gan_params["deepsdf_model_name"]
deepsdf_epoch_load = gan_params["deepsdf_epoch_load"]

# for loading LGAN model
z_dim = gan_params["z_dim"]
hidden_dims_g = gan_params["hidden_dims_g"]
generator_params_path = os.path.join(
    "../latentgan/experiments",
    gan_model_name,
    model_params_subdir,
    str(gan_iter_load) + "_g.pth"
)

deepsdf_experiments_dir = "/home/awang156/DeepSDF/experiments"
deepsdf_model_specs_path = os.path.join(deepsdf_experiments_dir, deepsdf_model_name, "specs.json")
deepsdf_model_codes_path = os.path.join(deepsdf_experiments_dir, deepsdf_model_name, "LatentCodes", str(deepsdf_epoch_load) + ".pth")
deepsdf_specs = json.load(open(deepsdf_model_specs_path))
latent_size = deepsdf_specs["CodeLength"]