#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import logging
import json
import numpy as np
import os
import trimesh

import deep_sdf
import deep_sdf.workspace as ws


def evaluate(experiment_directory, checkpoint, max_meshes, data_dir=None, split_filename=None, keep_normalized=False, source_name_surface=None):

    with open(os.path.join(experiment_directory, "specs.json"), "r") as f:
        specs = json.load(f)

    if split_filename == None:
        split_filename = specs["TrainSplit"]

    if data_dir == None:
        data_dir = specs["DataSource"]

    with open(split_filename, "r") as f:
        split = json.load(f)

    chamfer_results = []
    chamfer_distances = []
    color_distances = []

    for dataset in split:
        if len(chamfer_results) == max_meshes:
            break

        for class_name in split[dataset]:
            if len(chamfer_results) == max_meshes:
                break

            for instance_name in split[dataset][class_name]:
                if len(chamfer_results) == max_meshes:
                    break

                logging.debug(
                    "evaluating " + os.path.join(dataset, class_name, instance_name)
                )

                reconstructed_mesh_filename = ws.get_reconstructed_training_mesh_filename(
                    experiment_directory, checkpoint, dataset, class_name, instance_name
                )

                logging.debug(
                    'reconstructed mesh is "' + reconstructed_mesh_filename + '"'
                )

                ground_truth_samples_filename = os.path.join(
                    data_dir,
                    "SurfaceSamples",
                    source_name_surface if source_name_surface != None else dataset,
                    class_name,
                    instance_name + ".ply",
                )

                logging.debug(
                    "ground truth samples are " + ground_truth_samples_filename
                )

                normalization_params_filename = os.path.join(
                    data_dir,
                    "NormalizationParameters",
                    dataset,
                    class_name,
                    instance_name + ".npz",
                )

                logging.debug(
                    "normalization params are " + ground_truth_samples_filename
                )

                # Surface Samples (with color)
                ground_truth_points = trimesh.load(ground_truth_samples_filename, process=False)

                # Reconstructed Training Mesh (with color)
                reconstruction = trimesh.load(reconstructed_mesh_filename, process=False)

                if keep_normalized:
                    offset = 0
                    scale = 1
                else:
                    normalization_params = np.load(normalization_params_filename)
                    offset = normalization_params["offset"]
                    scale = normalization_params["scale"]
                
                chamfer_dist, color_dist = deep_sdf.metrics.chamfer.compute_trimesh_chamfer_color(
                    ground_truth_points,
                    reconstruction,
                    offset,
                    scale,
                )

                logging.debug("chamfer distance: " + str(chamfer_dist))

                chamfer_results.append(
                    (os.path.join(dataset, class_name, instance_name), chamfer_dist, color_dist)
                )
                chamfer_distances.append(chamfer_dist)
                color_distances.append(color_dist)

    with open(
        os.path.join(
            ws.get_evaluation_dir(experiment_directory, checkpoint, True), "chamfer.csv"
        ),
        "w",
    ) as f:
        f.write("shape, chamfer_dist, color_dist\n")
        f.write("MEAN, {}, {}\n".format(np.mean(chamfer_distances), np.mean(color_distances)))
        f.write("MEDIAN, {}, {}\n".format(np.median(chamfer_distances), np.median(color_distances)))
        for result in chamfer_results:
            f.write("{}, {}, {}\n".format(result[0], result[1], result[2]))


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Evaluate a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment specifications in "
        + '"specs.json", and logging will be done in this directory as well.',
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint to test.",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        default=None,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        default=None,
        help="The split to evaluate.",
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
        help="The maximum number of meshes to evaluate, or -1 for no limit.",
    )
    arg_parser.add_argument(
        "--name_surface",
        dest="source_name_surface",
        default=None,
        help="The name to use for the data source. If unspecified, it defaults to the "
        + "directory name.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    evaluate(
        args.experiment_directory,
        args.checkpoint,
        int(args.max_meshes),
        data_dir=args.data_source,
        split_filename=args.split_filename,
        keep_normalized=args.keep_normalized,
        source_name_surface=args.source_name_surface
    )
