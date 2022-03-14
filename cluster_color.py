from skimage import color
from sklearn.cluster import DBSCAN
import numpy as np
import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


split_path = "./experiments/splits/bed.json"
split_name = "3D-FUTURE-model"
sdf_abs_threshold = 0.01 # only consider points with |sdf| < threshold

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


allLAB = allLAB[::200].astype(np.float32)
num_samples = allLAB.shape[0]
print("num_samples", num_samples)

# print(allLAB[0:10, :])
# print(np.min(allLAB, axis=0))
# print(np.max(allLAB, axis=0))

clustering = DBSCAN(eps=0.01, min_samples=num_samples / 10000, n_jobs=12).fit(allLAB)
print(clustering.labels_[0:10])
print(np.min(clustering.labels_))
print(np.max(clustering.labels_))
print(np.sum(np.where(clustering.labels_ == -1, 1, 0)) / num_samples)

for i in range(min(np.max(clustering.labels_), 20)):
    proportion = np.sum(np.where(clustering.labels_ == i, 1, 0)) / num_samples
    centroid = np.mean(allLAB[clustering.labels_ == i, :], axis=0)
    print("Cluster", i, proportion, centroid)
