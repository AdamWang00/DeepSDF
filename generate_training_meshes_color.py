#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import numpy as np
import os
import torch

import deep_sdf
import deep_sdf.workspace as ws


BBOX_FACTOR = 1.02  # samples from BBOX_FACTOR times the bounding box size
LEVEL_SET = 0.0 # SDF value used to determine surface level set
SAMPLING_DIM = 256


def code_to_mesh(experiment_directory, checkpoint, max_meshes, keep_normalized=False, is_colorcat=False):

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
        os.path.join(experiment_directory,
                     ws.model_params_subdir, checkpoint + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    # decoder = decoder.module.cuda()
    decoder = decoder.cuda()

    decoder.eval()

    latent_vectors = ws.load_latent_vectors(experiment_directory, checkpoint)

    train_split_file = specs["TrainSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    data_source = specs["DataSource"]

    instance_filenames = deep_sdf.data.get_instance_filenames(
        data_source, train_split)

    print(len(instance_filenames), " vs ", len(latent_vectors))

    for i, latent_vector in enumerate(latent_vectors):
        if i == max_meshes:
            break

        dataset_name, class_name, instance_name = instance_filenames[i].split(
            "/")
        instance_name = instance_name.split(".")[0]

        print("{}/{}: {} {} {}".format(i + 1, max_meshes,
              dataset_name, class_name, instance_name))

        mesh_dir = os.path.join(
            experiment_directory,
            ws.training_meshes_subdir,
            str(saved_model_epoch),
            dataset_name,
            class_name,
        )

        if not os.path.isdir(mesh_dir):
            os.makedirs(mesh_dir)

        mesh_filename = os.path.join(mesh_dir, instance_name)

        if keep_normalized:
            offset = None
            scale = None
        else:
            normalization_params = np.load(
                ws.get_normalization_params_filename(
                    data_source, dataset_name, class_name, instance_name
                )
            )
            offset = normalization_params["offset"]
            scale = normalization_params["scale"]

        with torch.no_grad():
            deep_sdf.mesh_color.create_mesh(
                decoder,
                latent_vector,
                mesh_filename,
                N=SAMPLING_DIM,
                max_batch=int(2 ** 17),
                offset=offset,
                scale=scale,
                bbox_factor=BBOX_FACTOR,
                is_colorcat=is_colorcat,
                level_set=LEVEL_SET
            )


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to generate a mesh given a latent code."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--keep_normalization",
        "-n",
        dest="keep_normalized",
        default=False,
        action="store_true",
        help="If set, keep the meshes in the normalized scale.",
    )
    arg_parser.add_argument(
        "--max_meshes",
        "-m",
        dest="max_meshes",
        default=-1,
        help="The maximum number of meshes to generate, or -1 for no limit.",
    )
    arg_parser.add_argument(
        "--cat",
        dest="is_colorcat",
        default=False,
        action="store_true",
        help="Set to true for categorical color.",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    code_to_mesh(args.experiment_directory, args.checkpoint, int(args.max_meshes),
        keep_normalized=args.keep_normalized, is_colorcat=args.is_colorcat)
