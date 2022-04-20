import json
import os
import torch

import deep_sdf
import deep_sdf.workspace as ws

from latentgan_reverse.config import *
from latentgan_reverse.model import GeneratorAE


experiment_directory = f"experiments/{deepsdf_model_name}"
epoch = str(deepsdf_epoch_load)
mesh_id1 = "c2f29b88-e421-4b9c-ab06-50e9e83cf8a1"
mesh_id2 = "30d524b6-ad13-440f-a1ac-6d564213c2f7"
# mesh_id2 = "b6ec33de-c372-407c-b5b7-141436b02e7b"

NUM_INTERPOLATIONS = 32
BBOX_FACTOR = 1.02  # samples from BBOX_FACTOR times the bounding box size
LEVEL_SET = 0.0 # SDF value used to determine surface level set
SAMPLING_DIM = 256 # samples NxNxN points inside the bounding box
COLOR_ANNEALING_TEMPERATURE = 0.38 # 1.0 = mean, 0.01 = mode

if NUM_INTERPOLATIONS < 2:
    raise Exception("number of interpolations must be at least 2")

specs_filename = os.path.join(experiment_directory, "specs.json")

if not os.path.isfile(specs_filename):
    raise Exception(
        'The experiment directory does not include specifications file "specs.json"'
    )

specs = json.load(open(specs_filename))

arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

latent_size = specs["CodeLength"]

decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

# decoder = torch.nn.DataParallel(decoder)

saved_model_state = torch.load(
    os.path.join(experiment_directory, ws.model_params_subdir, epoch + ".pth")
)
saved_model_epoch = saved_model_state["epoch"]

decoder.load_state_dict(saved_model_state["model_state_dict"])

# decoder = decoder.module.cuda()
decoder = decoder.cuda()

decoder.eval()

latent_vectors = ws.load_latent_vectors(experiment_directory, epoch)

train_split_file = specs["TrainSplit"]

with open(train_split_file, "r") as f:
    train_split = json.load(f)

data_source = specs["DataSource"]

instance_filenames = deep_sdf.data.get_instance_filenames(data_source, train_split)

instance_names = []
for instance_filename in instance_filenames:
    dataset_name, class_name, instance_name = instance_filename.split("/")
    instance_name = instance_name.split(".")[0]
    instance_names.append(instance_name)

try:
    mesh_index1 = instance_names.index(mesh_id1)
except ValueError:
    raise Exception("mesh_id1 " + mesh_id1 + " not found")

try:
    mesh_index2 = instance_names.index(mesh_id2)
except ValueError:
    raise Exception("mesh_id2 " + mesh_id2 + " not found")

latent_vector1 = latent_vectors[mesh_index1]
latent_vector2 = latent_vectors[mesh_index2]

generator_ae = GeneratorAE()
generator_ae_params_dir = os.path.join("latentgan_reverse/experiments", model_name, model_params_subdir)
generator_ae.load_model(generator_ae_params_dir, epoch_load)
latent_vector1 = generator_ae.encoder(latent_vector1.unsqueeze(0).cuda()).squeeze()
latent_vector2 = generator_ae.encoder(latent_vector2.unsqueeze(0).cuda()).squeeze()

mesh_dir = os.path.join(
    experiment_directory,
    'Interpolations2',
    epoch,
    mesh_id1 + '_' + mesh_id2
)

if not os.path.isdir(mesh_dir):
    os.makedirs(mesh_dir)

for i in range(NUM_INTERPOLATIONS):
    w = 1 - (i / (NUM_INTERPOLATIONS - 1)) # weight on latent_vector1
    latent_vector = latent_vector1 * w + latent_vector2 * (1-w)
    latent_vector = generator_ae.decoder(latent_vector.unsqueeze(0)).squeeze()

    print(f'interpolation {i+1} of {NUM_INTERPOLATIONS} (w={w})')
    mesh_filename = os.path.join(mesh_dir, str(i+1))

    with torch.no_grad():
        deep_sdf.mesh_colorcat.create_mesh(
            decoder,
            latent_vector,
            mesh_filename,
            specs["ColorBinsPath"],
            specs["ColorBinsKey"],
            max_batch=int(2 ** 17),
            offset=None,
            scale=None,
            N=SAMPLING_DIM,
            bbox_factor=BBOX_FACTOR,
            level_set=LEVEL_SET,
            annealing_temperature=COLOR_ANNEALING_TEMPERATURE,
        )