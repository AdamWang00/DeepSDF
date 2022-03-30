import trimesh
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from deep_sdf.utils import compute_soft_encoding_lookup_table, lab_bins_to_rgb

save_to = 'cloud.ply'
save_to_hist = 'cloud_hist.png'
color_bins_path = 'color_bins/nightstand/clusters.npz'
color_bins_key = 'average'
annealing_temperature = 0.38 # 1.0 = mean, 0.01 = mode
# aa4714b2-e1f4-40b9-b680-5ee2d82c920b a2a8b471-48b3-4dd9-9ffd-dcf66d72dd58 192ac441-48b7-4559-bc02-7c532171b531 0ecff51f-5f8e-4ac7-b28a-d98d19446ff2
sdf_path = 'data/SdfSamples/3D-FUTURE-model_manifold/category_2/f3a8453b-a69d-4227-a501-97dd702bbbe7.npz'
# sdf_path = 'data/SdfSamples/3D-FUTURE-model_manifold/category_13/81aec6cd-34d7-4619-81e6-56bd1cdc1265.npz'
sdf_abs = 0.01
sdf_lower = -0.01
sdf_upper = 0.01

sdf = np.load(sdf_path)
sdf_pos = sdf["pos"]
sdf_neg = sdf["neg"]
sdf_all = np.concatenate((sdf_pos, sdf_neg), axis=0)
print("total number of sdf samples:", sdf_all.shape[0])
if sdf_abs is None:
    sdf_all = sdf_all[np.logical_and(sdf_lower < sdf_all[:, 3], sdf_all[:, 3] < sdf_upper), :]
else:
    sdf_all = sdf_all[np.abs(sdf_all[:, 3]) < sdf_abs, :]
print("number of samples within threshold:", sdf_all.shape[0])
print(np.min(sdf_all[:, 0]), np.max(sdf_all[:, 0]), np.min(sdf_all[:, 1]), np.max(
    sdf_all[:, 1]), np.min(sdf_all[:, 2]), np.max(sdf_all[:, 2]))

bin_width = 1
plt.hist(sdf_all[:, 4], bins=np.arange(min(sdf_all[:, 4]),
         max(sdf_all[:, 4]) + bin_width, bin_width))
plt.savefig(save_to_hist)

vertices = sdf_all[::, 0:3]

color_bin_idx = sdf_all[:, 4].astype(np.int)
bin_to_lab = np.load(color_bins_path)[color_bins_key]

color_bin_soft_encodings = compute_soft_encoding_lookup_table(bin_to_lab)
vertex_colors_512 = color_bin_soft_encodings[color_bin_idx]

vertex_colors = lab_bins_to_rgb(vertex_colors_512, bin_to_lab, annealing_temperature=annealing_temperature)

mesh = trimesh.Trimesh(vertices=vertices, vertex_colors=vertex_colors)

mesh.export(file_obj=save_to, file_type='ply')