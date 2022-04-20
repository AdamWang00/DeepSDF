import json
import os

model_name = "nightstand1"
iter_load = "200000"
print("latentgan", model_name, iter_load)

save_per_iters = 50000

model_params_subdir = "ModelParameters"
model_generations_subdir = "Generations"

params_history = {
    "nightstand1": {
        "deepsdf_model_name": "nightstand4",
        "deepsdf_epoch_load": 1000,
        "generator_iters": 200000,
        "batch_size": 64,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "gp_lambda": 10,
        "z_dim": 256,
        "hidden_dims_g": [512, 512, 512, 512],
        "hidden_dims_d": [512, 512, 512, 512]
    },
}

params = params_history[model_name]

deepsdf_model_name = params["deepsdf_model_name"]
deepsdf_epoch_load = params["deepsdf_epoch_load"]

# for loading LGAN model
generator_iters = params["generator_iters"]
batch_size = params["batch_size"]
learning_rate_g = params["learning_rate_g"]
learning_rate_d = params["learning_rate_d"]
z_dim = params["z_dim"]
hidden_dims_g = params["hidden_dims_g"]
hidden_dims_d = params["hidden_dims_d"]
gp_lambda = params["gp_lambda"]

deepsdf_experiments_dir = "/home/awang156/DeepSDF/experiments"
deepsdf_model_specs_path = os.path.join(deepsdf_experiments_dir, deepsdf_model_name, "specs.json")
deepsdf_model_codes_path = os.path.join(deepsdf_experiments_dir, deepsdf_model_name, "LatentCodes", str(deepsdf_epoch_load) + ".pth")
deepsdf_specs = json.load(open(deepsdf_model_specs_path))
latent_size = deepsdf_specs["CodeLength"]