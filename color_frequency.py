from skimage import color
import numpy as np
import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def normalize(x, center, norm):
    return (x - center) / norm

split_path = "./experiments/splits/nightstand.json"
split_name = "3D-FUTURE-model"
sdf_abs_threshold = 0.01
measure_min_max = False


with open(split_path, "r") as f:
    split = json.load(f)[split_name]

if measure_min_max:
    minL = 10000
    maxL = -10000
    minA = 10000
    maxA = -10000
    minB = 10000
    maxB = -10000

allA = []
allB = []

for category in split:
    i = 0
    total = len(split[category])
    for model_id in split[category]:
        i += 1
        print(i, '/', total)

        sdf_path = os.path.join('data/SdfSamples', split_name, category, model_id + '.npz')

        sdf = np.load(sdf_path)
        sdf_pos = sdf["pos"]
        sdf_neg = sdf["neg"]
        sdf_all = np.concatenate((sdf_pos, sdf_neg), axis=0)
        sdf_all = sdf_all[np.abs(sdf_all[:, 3]) < sdf_abs_threshold, :]

        rgb = sdf_all[:, 4:7]
        lab = color.rgb2lab(rgb)
        
        lab[:, 0] = normalize(lab[:, 0], 50, 100)
        lab[:, 1] = normalize(lab[:, 1], 0, 110)
        lab[:, 2] = normalize(lab[:, 2], 0, 110)

        if measure_min_max:
            minL = min(minL, np.min(lab[:, 0]))
            maxL = max(maxL, np.max(lab[:, 0]))
            minA = min(minA, np.min(lab[:, 1]))
            maxA = max(maxA, np.max(lab[:, 1]))
            minB = min(minB, np.min(lab[:, 2]))
            maxB = max(maxB, np.max(lab[:, 2]))
        
        allA.extend(lab[:, 1])
        allB.extend(lab[:, 2])

if measure_min_max:
    print(minL, maxL, minA, maxA, minB, maxB)

plt.hist2d(allA, allB, bins=(40, 40), range=((-2, 2), (-2, 2)), norm=mpl.colors.LogNorm())
plt.show()