from skimage import color
import numpy as np
import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

split_path = "./experiments/splits/nightstand.json"
split_name = "3D-FUTURE-model"
sdf_abs_threshold = 0.01

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
allLAB[:, 0] = allLAB[:, 0] / 100  # [0.0, 1.0)
allLAB[:, 1] = (allLAB[:, 1] + 100) / 200  # about (0.0, 1.0)
allLAB[:, 2] = (allLAB[:, 2] + 100) / 200  # about (0.0, 1.0)
allLAB = np.clip((allLAB * 8).astype(np.int), 0, 7)  # [0.0, 8.0) -> [0, 7]

allRGB = allRGB / 256  # [0.0, 1.0)
allRGB = np.clip((allRGB * 8).astype(np.int), 0, 7)  # [0.0, 8.0) -> [0, 7]

for i in range(8):
    slice_idx = allRGB[:, 0] == i
    plt.clf()
    plt.hist2d(allRGB[slice_idx, 1], allRGB[slice_idx, 2], bins=(
        8, 8), range=((0, 8), (0, 8)), norm=mpl.colors.LogNorm())
    plt.savefig(f"color_frequency_rgb{i}.png")

plt.clf()
plt.hist2d(allRGB[:, 1], allRGB[:, 2], bins=(8, 8), range=(
    (0, 8), (0, 8)), norm=mpl.colors.LogNorm())
plt.savefig("color_frequency_rgb.png")

for i in range(8):
    slice_idx = allLAB[:, 0] == i
    plt.clf()
    plt.hist2d(allLAB[slice_idx, 1], allLAB[slice_idx, 2], bins=(
        8, 8), range=((0, 8), (0, 8)), norm=mpl.colors.LogNorm())
    plt.savefig(f"color_frequency_lab{i}.png")

plt.clf()
plt.hist2d(allLAB[:, 1], allLAB[:, 2], bins=(8, 8), range=(
    (0, 8), (0, 8)), norm=mpl.colors.LogNorm())
plt.savefig("color_frequency_lab.png")