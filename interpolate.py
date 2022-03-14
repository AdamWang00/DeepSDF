import json
import os
import torch

import deep_sdf
import deep_sdf.workspace as ws


experiment_directory = "experiments/nightstand2"
epoch = "latest"
mesh_id1 = "c2f29b88-e421-4b9c-ab06-50e9e83cf8a1"
mesh_id2 = "b6ec33de-c372-407c-b5b7-141436b02e7b"

num_interpolations = 32
bbox_factor = 1.01  # samples from BBOX_FACTOR times the bounding box size
level_set = 0.0 # SDF value used to determine surface level set
sampling_dim = 256
is_colorcat = False

if num_interpolations < 2:
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

mesh_dir = os.path.join(
    experiment_directory,
    'Interpolations',
    epoch,
    mesh_id1 + '_' + mesh_id2
)

if not os.path.isdir(mesh_dir):
    os.makedirs(mesh_dir)

for i in range(num_interpolations):
    w = 1 - (i / (num_interpolations - 1)) # weight on latent_vector1
    latent_vector = latent_vector1 * w + latent_vector2 * (1-w)

    print(f'interpolation {i+1} of {num_interpolations} (w={w})')
    mesh_filename = os.path.join(mesh_dir, str(i+1))

    with torch.no_grad():
        deep_sdf.mesh_color.create_mesh(
            decoder,
            latent_vector,
            mesh_filename,
            max_batch=int(2 ** 17),
            offset=None,
            scale=None,
            N=sampling_dim,
            bbox_factor=bbox_factor,
            level_set=level_set,
            is_colorcat=is_colorcat,
        )