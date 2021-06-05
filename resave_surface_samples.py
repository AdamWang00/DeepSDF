#!/usr/bin/env python3

import argparse
import json
import logging
import os

import deep_sdf
import deep_sdf.workspace as ws

import numpy as np
import scipy as sp
import trimesh
from trimesh.visual import uv_to_color
from PIL import Image
from scipy.spatial import Delaunay, cKDTree

def get_trimesh_and_uv(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(g for g in scene_or_mesh.geometry.values())
        )
        uv = np.concatenate(
            tuple(g.visual.uv for g in scene_or_mesh.geometry.values()),
            axis=0
        )
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
        uv = mesh.visual.uv
    return mesh, uv


def reset_color(model_path, texture_path, surface_samples_path, surface_sample_faces_path, sdf_path, ignore_sdf=False, ignore_surface=False):
    surface_samples = trimesh.load(surface_samples_path, process=False)
    assert surface_samples.visual.kind == 'vertex'
    surface_samples.export(file_obj=surface_samples_path, file_type='ply', encoding='ascii')


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Adds color to preprocessed information (after preprocess_data).",
    )
    arg_parser.add_argument(
        "--data_dir",
        "-d",
        dest="data_dir",
        required=True,
        help="The directory which holds all preprocessed data.",
    )
    arg_parser.add_argument(
        "--source",
        "-s",
        dest="source_dir",
        required=True,
        help="The directory which holds the data to preprocess and append.",
    )
    arg_parser.add_argument(
        "--name",
        "-n",
        dest="source_name",
        default=None,
        help="The name to use for the data source. If unspecified, it defaults to the "
        + "directory name.",
    )
    arg_parser.add_argument(
        "--split",
        dest="split_filename",
        required=True,
        help="A split filename defining the shapes to be processed.",
    )
    arg_parser.add_argument(
        "--ignore_sdf",
        dest="ignore_sdf",
        default=False,
        action="store_true",
        help="If set, sdf samples are unchanged. Set this if sdf samples are already colored.",
    )
    arg_parser.add_argument(
        "--ignore_surface",
        dest="ignore_surface",
        default=False,
        action="store_true",
        help="If set, surface samples are unchanged. Set this if surface samples are already colored.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    ignore_sdf = args.ignore_sdf
    ignore_surface = args.ignore_surface

    deep_sdf.configure_logging(args)

    deepsdf_dir = os.path.dirname(os.path.abspath(__file__))

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    if args.source_name is None:
        args.source_name = os.path.basename(os.path.normpath(args.source_dir))

    mesh_dir = args.source_dir
    surface_samples_dir = os.path.join(args.data_dir, ws.surface_samples_subdir, args.source_name)
    surface_samples_faces_dir = os.path.join(args.data_dir, ws.surface_sample_faces_subdir, args.source_name)
    sdf_samples_dir = os.path.join(args.data_dir, ws.sdf_samples_subdir, args.source_name)

    preprocess_color_inputs = []
    class_directories = split[args.source_name]
    for class_dir in class_directories:
        instance_dirs = class_directories[class_dir]

        mesh_class_dir = os.path.join(mesh_dir, class_dir)
        surface_samples_class_dir = os.path.join(surface_samples_dir, class_dir)
        surface_samples_faces_class_dir = os.path.join(surface_samples_faces_dir, class_dir)
        sdf_samples_class_dir = os.path.join(sdf_samples_dir, class_dir)

        for instance_dir in instance_dirs:
            mesh_class_instance_dir = os.path.join(mesh_class_dir, instance_dir)
            texture_class_instance_path = os.path.join(mesh_class_instance_dir, "texture.png")
            surface_samples_class_instance_path = os.path.join(surface_samples_class_dir, instance_dir + ".ply")
            surface_sample_faces_class_instance_path = os.path.join(surface_samples_faces_class_dir, instance_dir + ".txt")
            sdf_samples_class_instance_path = os.path.join(sdf_samples_class_dir, instance_dir + ".npz")

            try:
                mesh_class_instance_path = deep_sdf.data.find_mesh_in_directory(mesh_class_instance_dir)
                preprocess_color_inputs.append(
                    (
                        mesh_class_instance_path,
                        texture_class_instance_path,
                        surface_samples_class_instance_path,
                        surface_sample_faces_class_instance_path,
                        sdf_samples_class_instance_path
                    )
                )
            except deep_sdf.data.NoMeshFileError:
                logging.warning("No mesh found for instance " + instance_dir)
            except deep_sdf.data.MultipleMeshFileError:
                logging.warning("Multiple meshes found for instance " + instance_dir)

    c = 0
    total = len(preprocess_color_inputs)
    for args in preprocess_color_inputs:
        c += 1
        print(c, "of", total, args[0])
        reset_color(*args, ignore_sdf=ignore_sdf, ignore_surface=ignore_surface)