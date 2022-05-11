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
from skimage import color
from trimesh.visual import uv_to_color
from PIL import Image
from scipy.spatial import Delaunay, cKDTree

COLOR_BINS_KEY = "average"

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


def preprocess_color(model_path, texture_path, surface_samples_path, surface_sample_faces_path, sdf_path, color_bins_path, ignore_sdf=False, ignore_surface=False, overwrite_sdf=False):
    # Load SDF samples
    sdf = np.load(sdf_path)
    sdf_pos = sdf["pos"]
    sdf_neg = sdf["neg"]

    output_shape = 4 + 1 # input_shape (4) + bin index (1)
    if ignore_sdf: # if we only want to add color to the surface samples (.ply)
        print("ignoring sdf")
    elif sdf_pos.shape[1] == output_shape and sdf_neg.shape[1] == output_shape:
        if overwrite_sdf:
            print("overwriting sdf")
        else:
            print("sdf already contains color, skipping...")
            return
    elif sdf_pos.shape[1] != 4 or sdf_neg.shape[1] != 4:
        print("unexpected sdf, overwriting")
        # return

    if ignore_surface:
        surface_samples_header_size=14
    else:
        surface_samples_header_size=9

    # Load surface samples and the vertices of the face from which each is sampled
    surface_sample_faces = np.genfromtxt(surface_sample_faces_path)
    num_samples = surface_sample_faces.shape[0] // 3

    surface_sample_coords = np.genfromtxt(surface_samples_path, skip_header=surface_samples_header_size, max_rows=num_samples)[:, 0:3]

    if ignore_surface:
        samples_color = np.genfromtxt(surface_samples_path, skip_header=surface_samples_header_size, max_rows=num_samples)[:, 3:6]
    else:
        # Load model without processing and textures
        mesh = trimesh.load(model_path, process=False)
        try:
            mesh, vertex_uv = get_trimesh_and_uv(mesh)
        except:
            print("failed to get mesh and uv", model_path)
            return
        textures = Image.open(texture_path)

        # Build k-d tree of the model's face centroids
        model_face_centroids = np.mean(mesh.vertices[mesh.faces], axis=1)
        model_face_centroids_kdtree = cKDTree(model_face_centroids)

        # Match each surface sample to the face of the model from which it is sampled
        sample_face_centroids = np.mean(np.reshape(surface_sample_faces, (-1, 3, 3)), axis=1)
        _, sample_face_indices = model_face_centroids_kdtree.query(sample_face_centroids)

        # Get the UV positions of the vertices of the model faces
        sample_triangles = mesh.vertices[mesh.faces[sample_face_indices]]
        sample_triangles_uv = vertex_uv[mesh.faces[sample_face_indices]]

        # Gets the barycentric coordinates of each sample in the simplex defined by points.
        # Also returns the order of the simplex points used to form the simplex
        def barycentric_coords(points, sample):
            dim = len(points[0])
            tri = Delaunay(points)
            s = 0 # tri.find_simplex(sample)
            b = tri.transform[s, :dim].dot(np.transpose(sample - tri.transform[s, dim]))
            return np.append(b, 1 - b.sum()), tri.simplices[s]

        # Use barycentric coordinates to calculate the UV position of each sample
        samples_uv = np.empty((num_samples, 2))
        for i in range(num_samples):
            b_coords_best = None
            points_indices_best = None
            loss_best = 1
            planes = [[0, 1], [0, 2], [1, 2]]

            for plane in planes:
                points = np.transpose(sample_triangles[i, :, plane])
                sample = surface_sample_coords[i, plane]
                try:
                    # Barycentric transform using 2D projections of triangle vertices
                    b_coords, points_indices = barycentric_coords(points, sample)
                except sp.spatial.qhull.QhullError:
                    continue
                reconstructed = np.dot(b_coords, sample_triangles[i][points_indices])
                loss = np.linalg.norm(reconstructed - surface_sample_coords[i])
                if (loss < loss_best):
                    b_coords_best = b_coords
                    points_indices_best = points_indices
                    loss_best = loss
            
            if b_coords_best is None:
                samples_uv[i] = [0, 0]
            else :
                samples_uv[i] = np.dot(b_coords_best, sample_triangles_uv[i][points_indices_best])

        samples_color = uv_to_color(samples_uv, textures)[:, 0:3] # discard alpha

        surface_samples = trimesh.load(surface_samples_path, process=False)
        surface_samples.visual.vertex_colors = samples_color
        assert surface_samples.visual.kind == 'vertex'
        surface_samples.export(file_obj=surface_samples_path, file_type='ply', encoding='ascii')

    if not ignore_sdf:
        # Build k-d tree of surface samples and query sdf samples
        surface_sample_kdtree = cKDTree(surface_sample_coords)
        _, sdf_pos_indices = surface_sample_kdtree.query(sdf_pos[:, 0:3])
        _, sdf_neg_indices = surface_sample_kdtree.query(sdf_neg[:, 0:3])
        surface_sample_kdtree = None # gc

        # RGB of closest surface sample
        sdf_pos_color = samples_color[sdf_pos_indices]
        sdf_neg_color = samples_color[sdf_neg_indices]

        sdf_pos_lab = color.rgb2lab(sdf_pos_color / 255)
        sdf_neg_lab = color.rgb2lab(sdf_neg_color / 255)

        color_bins_lab = np.load(color_bins_path)[COLOR_BINS_KEY]
        color_bins_kdtree = cKDTree(color_bins_lab)
        _, sdf_pos_indices = color_bins_kdtree.query(sdf_pos_lab)
        _, sdf_neg_indices = color_bins_kdtree.query(sdf_neg_lab)

        sdf_pos = np.concatenate((
            sdf_pos[:, 0:4],
            np.expand_dims(sdf_pos_indices, 1),
        ), axis=1)
        sdf_neg = np.concatenate((
            sdf_neg[:, 0:4],
            np.expand_dims(sdf_neg_indices, 1),
        ), axis=1)

        np.savez(sdf_path, pos=sdf_pos.astype('float32'), neg=sdf_neg.astype('float32'))


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Adds color to preprocessed data (after preprocess_data).",
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
        "--name_surface",
        dest="source_name_surface",
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
    arg_parser.add_argument(
        "--overwrite_sdf",
        dest="overwrite_sdf",
        default=False,
        action="store_true",
        help="If set, sdf samples overwritten.",
    )
    arg_parser.add_argument(
        "--color_bins",
        dest="color_bins",
        required=True,
        help="Filepath to color bins.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    ignore_sdf = args.ignore_sdf
    ignore_surface = args.ignore_surface
    overwrite_sdf = args.overwrite_sdf
    color_bins_path = args.color_bins

    deep_sdf.configure_logging(args)

    deepsdf_dir = os.path.dirname(os.path.abspath(__file__))

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    if args.source_name is None:
        args.source_name = os.path.basename(os.path.normpath(args.source_dir))

    if args.source_name_surface is None:
        args.source_name_surface = args.source_name

    mesh_dir = args.source_dir
    surface_samples_dir = os.path.join(args.data_dir, ws.surface_samples_subdir, args.source_name_surface)
    surface_samples_faces_dir = os.path.join(args.data_dir, ws.surface_sample_faces_subdir, args.source_name_surface)
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
                        sdf_samples_class_instance_path,
                        color_bins_path
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
        print(c, "of", total, args[4])
        preprocess_color(*args, ignore_sdf=ignore_sdf, ignore_surface=ignore_surface, overwrite_sdf=overwrite_sdf)