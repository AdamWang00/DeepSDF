import json
import os
import torch

import deep_sdf
import deep_sdf.workspace as ws

def reconstruct_training_mesh(experiment_directory, checkpoint, mesh_id):

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
        mesh_index = instance_names.index(mesh_id)
    except ValueError:
        raise Exception("mesh_id " + mesh_id + " not found")

    latent_vector = latent_vectors[mesh_index]

    mesh_dir = os.path.join(
        'foo',
        experiment_directory
    )

    if not os.path.isdir(mesh_dir):
        os.makedirs(mesh_dir)

    mesh_filename = os.path.join(mesh_dir, mesh_id)

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
    mesh_id = '5de45849-7d1b-4378-82da-ed183b7ecc37'
    reconstruct_training_mesh(
        "experiments/chair5",
        "latest",
        mesh_id
    )