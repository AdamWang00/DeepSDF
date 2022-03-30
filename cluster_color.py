from skimage import color
import sklearn.cluster as cluster
import numpy as np
import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import trimesh
import trimesh.creation as creation

category = "nightstand"
split_name = "3D-FUTURE-model"

linkages = ['average']

skip_factor = 2
# sdf_abs_threshold = 0.1 # only consider points with |sdf| < threshold

remove_dup = True
normalize_before_clustering = False

colors_filename = "colors"
cluster_filename = "clusters"
save_dir = os.path.join("color_bins", category)

with open(f"./experiments/splits/{category}.json", "r") as f:
    split = json.load(f)[split_name]

if remove_dup:
    allRGB = set()
else:
    allRGB = []

for cat in split:
    i = 0
    total = len(split[cat])
    for model_id in split[cat]:
        i += 1
        print(i, '/', total, cat, model_id)

        # sdf_path = os.path.join(
        #     'data/SdfSamples', split_name, cat, model_id + '.npz')
        # sdf = np.load(sdf_path)
        # sdf_pos = sdf["pos"]
        # sdf_neg = sdf["neg"]
        # sdf_all = np.concatenate((sdf_pos, sdf_neg), axis=0)
        # sdf_all = sdf_all[np.abs(sdf_all[:, 3]) < sdf_abs_threshold, :]
        # rgb = sdf_all[:, 4:7].astype(int) # [0, 255]

        surface_samples_path = os.path.join(
            'data/SurfaceSamples', split_name, cat, model_id + '.ply')

        rgb = trimesh.load(surface_samples_path).visual.vertex_colors[:, 0:3]

        if remove_dup:
            for t in rgb:
                allRGB.add(tuple(t))
        else:
            allRGB.extend(rgb)

if remove_dup:
    allRGB = np.array(list(allRGB))[::skip_factor].astype(np.float32)
else:
    allRGB = np.array(allRGB)[::skip_factor].astype(np.float32)

allLAB = color.rgb2lab(allRGB / 255)  # input is [0.0, 1.0]
allRGB = None # gc

if normalize_before_clustering:
    allLAB[:, 0] = allLAB[:, 0] / 100  # [0.0, 1.0]
    allLAB[:, 1] = (allLAB[:, 1] + 100) / 200  # about (0.0, 1.0)
    allLAB[:, 2] = (allLAB[:, 2] + 100) / 200  # about (0.0, 1.0)

num_samples = allLAB.shape[0]
print("num_samples", num_samples)
# print(allLAB[0:10, :])
# print(np.min(allLAB, axis=0))
# print(np.max(allLAB, axis=0))

# =================================================================

to_save = {}

for linkage in linkages:

    n_clusters = 512
    clustering = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(allLAB)

    n_clusters_actual = np.max(clustering.labels_)+1
    print("Number of clusters:", n_clusters_actual)

    centroids = np.zeros((n_clusters_actual, 3))
    proportions = []
    for i in range(n_clusters_actual):
        proportion = np.sum(np.where(clustering.labels_ == i, 1, 0)) / num_samples
        centroid = np.mean(allLAB[clustering.labels_ == i, :], axis=0)
        centroids[i, :] = centroid
        proportions.append(proportion)
        # print("Cluster", i, proportion, centroid)
    print("Min/max proportions:", min(proportions), max(proportions))

    if normalize_before_clustering:
        centroids[:, 0] = centroids[:, 0] * 100
        centroids[:, 1] = centroids[:, 1] * 200 - 100
        centroids[:, 2] = centroids[:, 2] * 200 - 100

    vertex_colors = centroids
    vertex_colors = color.lab2rgb(vertex_colors)
    vertex_colors = (np.clip(vertex_colors * 256, 0, 255)).astype(np.int)

    scene = trimesh.Scene()
    proportion_scale = 10
    for i in range(n_clusters_actual):
        sphere = creation.icosphere(subdivisions=3, radius=proportion_scale*(proportions[i]**(1./3.)), color=vertex_colors[i, :])
        sphere.apply_translation(centroids[i, :])
        scene.add_geometry(sphere)

    mesh = trimesh.util.concatenate(
        tuple(g for g in scene.geometry.values())
    )
    mesh.export(file_obj=f"{cluster_filename}_{linkage}.ply", file_type='ply')

    to_save[linkage] = centroids

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
np.savez(os.path.join(save_dir, f"{cluster_filename}.npz"), **to_save)

if normalize_before_clustering:
    allLAB[:, 0] = allLAB[:, 0] * 100
    allLAB[:, 1] = allLAB[:, 1] * 200 - 100
    allLAB[:, 2] = allLAB[:, 2] * 200 - 100

vertex_colors = allLAB
vertex_colors = color.lab2rgb(vertex_colors)
vertex_colors = (np.clip(vertex_colors * 256, 0, 255)).astype(np.int)

mesh = trimesh.Trimesh(vertices=allLAB, vertex_colors=vertex_colors)
mesh.export(file_obj=f"{colors_filename}.ply", file_type='ply')