import json
import os

model_name = "nightstand2"
epoch_load = "10000"
print("latentvae", model_name, epoch_load)

save_per_epochs = 10000

model_params_subdir = "ModelParameters"

params_history = {
    "nightstand1": {
        "deepsdf_model_name": "nightstand4",
        "deepsdf_epoch_load": 1000,
        "epochs": 10001,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "kld_loss_weight": 0.0000001,
        "step_size": 2500,
        "step_gamma": 0.5,
        "z_dim": 64,
        "hidden_dims_e": [512, 512, 256, 128],
        "hidden_dims_d": [128, 256, 512, 512],
    },
    "nightstand2": {
        "deepsdf_model_name": "nightstand4",
        "deepsdf_epoch_load": 1000,
        "epochs": 10001,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "kld_loss_weight": 0.000001,
        "step_size": 2500,
        "step_gamma": 0.5,
        "z_dim": 64,
        "hidden_dims_e": [512, 512, 256, 128],
        "hidden_dims_d": [128, 256, 512, 512],
    },
    "nightstand3": {
        "deepsdf_model_name": "nightstand4",
        "deepsdf_epoch_load": 1000,
        "epochs": 10001,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "kld_loss_weight": 0.00001,
        "step_size": 2500,
        "step_gamma": 0.5,
        "z_dim": 64,
        "hidden_dims_e": [512, 512, 256, 128],
        "hidden_dims_d": [128, 256, 512, 512],
    },
    "nightstand4": {
        "deepsdf_model_name": "nightstand4",
        "deepsdf_epoch_load": 1000,
        "epochs": 10001,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "kld_loss_weight": 0.0001,
        "step_size": 2500,
        "step_gamma": 0.5,
        "z_dim": 64,
        "hidden_dims_e": [512, 512, 256, 128],
        "hidden_dims_d": [128, 256, 512, 512],
    },
}

params = params_history[model_name]
deepsdf_model_name = params["deepsdf_model_name"]
deepsdf_epoch_load = params["deepsdf_epoch_load"]
epochs = params["epochs"]
batch_size = params["batch_size"]
learning_rate = params["learning_rate"]
kld_loss_weight = params["kld_loss_weight"]
step_size = params["step_size"]
step_gamma = params["step_gamma"]
z_dim = params["z_dim"]
hidden_dims_e = params["hidden_dims_e"]
hidden_dims_d = params["hidden_dims_d"]

deepsdf_experiments_dir = "/home/awang156/DeepSDF/experiments"
deepsdf_model_specs_path = os.path.join(deepsdf_experiments_dir, deepsdf_model_name, "specs.json")
deepsdf_model_codes_path = os.path.join(deepsdf_experiments_dir, deepsdf_model_name, "LatentCodes", str(deepsdf_epoch_load) + ".pth")
deepsdf_specs = json.load(open(deepsdf_model_specs_path))
latent_size = deepsdf_specs["CodeLength"]