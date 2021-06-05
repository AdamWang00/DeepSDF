#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh


def compute_trimesh_chamfer(gt_points, gen_mesh, offset, scale, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just points, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    gt_points = gt_points.vertices

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


def compute_trimesh_chamfer_color(gt_mesh, gen_mesh, offset, scale, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    It also returns the symmetric "color distance" between pairs in chamfer distance.

    gt_mesh and gen_mesh should both contain colored vertices.
    """

    gt_points = gt_mesh.vertices
    gt_colors = gt_mesh.visual.vertex_colors[:, 0:3].astype(np.int64) # discard alpha

    gen_points = gen_mesh.vertices
    gen_colors = gen_mesh.visual.vertex_colors[:, 0:3].astype(np.int64)

    # sample from generated mesh vertices
    sample_indices = np.random.choice(
        gen_points.shape[0],
        min(num_mesh_samples, gen_points.shape[0]),
        replace=False
    )
    gen_points_sampled = gen_points[sample_indices]
    gen_colors_sampled = gen_colors[sample_indices]

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))
    one_colors = gen_colors_sampled[one_vertex_ids]
    gt_to_gen_color = np.mean(np.linalg.norm(gt_colors - one_colors, ord=1, axis=1))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))
    two_colors = gt_colors[two_vertex_ids]
    gen_to_gt_color = np.mean(np.linalg.norm(gen_colors_sampled - two_colors, ord=1, axis=1))

    return gt_to_gen_chamfer + gen_to_gt_chamfer, gt_to_gen_color + gen_to_gt_color