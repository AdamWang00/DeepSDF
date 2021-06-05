import json
import os
import torch

import deep_sdf
import deep_sdf.workspace as ws

def interpolate(experiment_directory, checkpoint, mesh_id1, mesh_id2, num_interpolations=2):
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

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, checkpoint + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    decoder.eval()

    latent_vectors = ws.load_latent_vectors(experiment_directory, checkpoint)

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
        'interpolations',
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
                N=256,
                max_batch=int(2 ** 17),
                offset=None,
                scale=None,
            )


if __name__ == "__main__":
    mesh_id1 = '18992c9e-25a6-301d-8be8-95b7494e010f'
    mesh_id2 = '5157b498-3cdf-4513-bd97-237358b1979e'
    interpolate(
        "experiments/chair2",
        "latest",
        mesh_id1,
        mesh_id2,
        num_interpolations=8
    )