#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch

import deep_sdf.utils


def create_mesh(
    decoder,
    latent_vec,
    filename,
    color_bins_filepath,
    color_bins_key,
    N=256,
    max_batch=32 ** 3,
    offset=None,
    scale=None,
    bbox_factor=1.0,
    level_set=0.0,
    annealing_temperature = 0.38,
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-bbox_factor, -bbox_factor, -bbox_factor]
    voxel_size = 2.0 * bbox_factor / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head: min(
            head + max_batch, num_samples), 0:3].cuda()

        samples[head: min(head + max_batch, num_samples), 3] = (
            deep_sdf.utils.decode_sdf(decoder, latent_vec, sample_subset)[0]  # sdf
            .detach()
            .cpu()
        )

        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        decoder,
        latent_vec,
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        color_bins_filepath,
        color_bins_key,
        offset=offset,
        scale=scale,
        level_set=level_set,
        annealing_temperature = 0.38,
    )


def convert_sdf_samples_to_ply(
    decoder,
    latent_vec,
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    color_bins_filepath,
    color_bins_key,
    offset=None,
    scale=None,
    max_batch=32 ** 3,
    level_set=0.0,
    annealing_temperature=0.38,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, _, _ = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level_set, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # get vertex colors and clamp to valid RGB range
    mesh_point_colors = np.zeros_like(verts)
    mesh_points_torch = torch.from_numpy(mesh_points).float()

    bin_to_rgb = np.load(color_bins_filepath)[color_bins_key]

    head = 0
    num_points = mesh_point_colors.shape[0]
    while head < num_points:
        color_bins = (
            deep_sdf.utils.decode_sdf(
                decoder,
                latent_vec,
                mesh_points_torch[head: min(
                    head + max_batch, num_points), :].cuda()
            )[1] # 512 bins
            .detach()
            .cpu()
            .numpy()
        )
        mesh_point_colors[head: min(head + max_batch, num_points), :] = deep_sdf.utils.lab_bins_to_rgb(
            color_bins,
            bin_to_rgb,
            annealing_temperature=annealing_temperature
        )

        head += max_batch

    mesh_point_colors = np.clip(mesh_point_colors, 0, 255)

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[(
        "x", "f4"), ("y", "f4"), ("z", "f4"), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    for i in range(0, num_verts):
        verts_tuple[i] = (
            mesh_points[i, 0],
            mesh_points[i, 1],
            mesh_points[i, 2],
            mesh_point_colors[i, 0],
            mesh_point_colors[i, 1],
            mesh_point_colors[i, 2]
        )

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[
                           ("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
