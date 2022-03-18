from skimage import color
import sklearn.cluster as cluster
import numpy as np
import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import trimesh
import trimesh.creation as creation

split_path = "./experiments/splits/bed.json"
split_name = "3D-FUTURE-model"
sdf_abs_threshold = 0.0001 # only consider points with |sdf| < threshold

with open(split_path, "r") as f:
    split = json.load(f)[split_name]

allRGB = []

for category in split:
    i = 0
    total = len(split[category])
    for model_id in split[category]:
        i += 1
        print(i, '/', total, category, model_id)

        sdf_path = os.path.join(
            'data/SdfSamples', split_name, category, model_id + '.npz')

        sdf = np.load(sdf_path)
        sdf_pos = sdf["pos"]
        sdf_neg = sdf["neg"]
        sdf_all = np.concatenate((sdf_pos, sdf_neg), axis=0)
        sdf_all = sdf_all[np.abs(sdf_all[:, 3]) < sdf_abs_threshold, :]

        rgb = sdf_all[:, 4:7]  # [0, 255]
        allRGB.extend(rgb)

allRGB = np.array(allRGB)
allLAB = color.rgb2lab(allRGB / 255)  # input is [0.0, 1.0]
allRGB = None # gc

allLAB[:, 0] = allLAB[:, 0] / 100  # [0.0, 1.0]
allLAB[:, 1] = (allLAB[:, 1] + 100) / 200  # about (0.0, 1.0)
allLAB[:, 2] = (allLAB[:, 2] + 100) / 200  # about (0.0, 1.0)


allLAB = allLAB[::5].astype(np.float32)
num_samples = allLAB.shape[0]
print("num_samples", num_samples)

# print(allLAB[0:10, :])
# print(np.min(allLAB, axis=0))
# print(np.max(allLAB, axis=0))


clustering = cluster.DBSCAN(eps=0.01, min_samples=num_samples / 5000, n_jobs=12).fit(allLAB)

# n_clusters = 512
# clustering = cluster.AgglomerativeClustering(n_clusters=n_clusters).fit(allLAB)

# =================================================================

# print(np.min(clustering.labels_))
# print(np.max(clustering.labels_))
# print(np.sum(np.where(clustering.labels_ == -1, 1, 0)) / num_samples)

n_clusters_actual = np.max(clustering.labels_)+1
print("Number of clusters:", n_clusters_actual)

centroids = np.zeros((n_clusters_actual, 3))
proportions = []
for i in range(n_clusters_actual):
    proportion = np.sum(np.where(clustering.labels_ == i, 1, 0)) / num_samples
    centroid = np.mean(allLAB[clustering.labels_ == i, :], axis=0)
    centroids[i, :] = centroid
    proportions.append(proportion)
    print("Cluster", i, proportion, centroid)

print(min(proportions), max(proportions))

vertex_colors = centroids # normalized Lab
vertex_colors[:, 0] = vertex_colors[:, 0] * 100
vertex_colors[:, 1] = vertex_colors[:, 1] * 200 - 100
vertex_colors[:, 2] = vertex_colors[:, 2] * 200 - 100
vertex_colors = color.lab2rgb(vertex_colors) # normalized rgb
vertex_colors = (np.clip(vertex_colors * 256, 0, 255)).astype(np.int)
print(vertex_colors[0:10, :])


# mesh = trimesh.Trimesh(vertices=centroids, vertex_colors=vertex_colors)
# mesh.export(file_obj="clustersDBSCAN.ply", file_type='ply')

scene = trimesh.Scene()
proportion_scale = 10
for i in range(n_clusters_actual):
    sphere = creation.icosphere(subdivisions=4, radius=proportion_scale*(proportions[i]**(1./3.)), color=vertex_colors[i, :])
    sphere.apply_translation(centroids[i, :])
    scene.add_geometry(sphere)

mesh = trimesh.util.concatenate(
    tuple(g for g in scene.geometry.values())
)
mesh.export(file_obj="clustersDBSCAN.ply", file_type='ply')


vertex_colors = allLAB # normalized Lab
vertex_colors[:, 0] = vertex_colors[:, 0] * 100
vertex_colors[:, 1] = vertex_colors[:, 1] * 200 - 100
vertex_colors[:, 2] = vertex_colors[:, 2] * 200 - 100
vertex_colors = color.lab2rgb(vertex_colors) # normalized rgb
vertex_colors = (np.clip(vertex_colors * 256, 0, 255)).astype(np.int)
print(vertex_colors[0:10, :])

mesh = trimesh.Trimesh(vertices=allLAB, vertex_colors=vertex_colors)
mesh.export(file_obj="colorsDBSCAN.ply", file_type='ply')