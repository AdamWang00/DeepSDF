import trimesh
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

save_to = 'cloud.ply'
save_to_hist = 'cloud_hist.png'
sdf_path = 'data/SdfSamples/3D-FUTURE-model/category_13/7d0a3b31-7d66-4d09-9761-647938bf43a6.npz'
sdf_abs_threshold = 0.01

sdf = np.load(sdf_path)
sdf_pos = sdf["pos"]
sdf_neg = sdf["neg"]
sdf_all = np.concatenate((sdf_pos, sdf_neg), axis=0)
print("total number of sdf samples:", sdf_all.shape[0])
sdf_all = sdf_all[np.abs(sdf_all[:, 3]) < sdf_abs_threshold, :]
print("number of samples within threshold:", sdf_all.shape[0])
print(np.min(sdf_all[:, 0]), np.max(sdf_all[:, 0]), np.min(sdf_all[:, 1]), np.max(
    sdf_all[:, 1]), np.min(sdf_all[:, 2]), np.max(sdf_all[:, 2]))

# bin_width = 0.01
# plt.hist(sdf_all[:, 3], bins=np.arange(min(sdf_all[:, 3]),
#          max(sdf_all[:, 3]) + bin_width, bin_width))
# plt.savefig(save_to_hist)

vertices = sdf_all[:, 0:3]
vertex_colors = sdf_all[:, 4:7]

mesh = trimesh.Trimesh(vertices=vertices, vertex_colors=vertex_colors)

mesh.export(file_obj=save_to, file_type='ply')

grid_frequency = np.zeros((32, 32, 32))
grid_indices = np.clip(((vertices[:, 0:3] + 1) * 16).astype(np.int), 0, 31)
for grid_index in grid_indices:
    grid_frequency[tuple(grid_index)] += 1

print(min(grid_indices[:, 0]), max(grid_indices[:, 0]), min(grid_indices[:, 1]), max(
    grid_indices[:, 1]), min(grid_indices[:, 2]), max(grid_indices[:, 2]))
print(grid_frequency[0, 0, 0])
print(np.min(grid_frequency), np.max(grid_frequency))

plt.clf()

slice_idx = grid_indices[:, 2] == 15
plt.hist2d(grid_indices[slice_idx, 0], grid_indices[slice_idx, 1], bins=(32, 32), range=(
    (0, 32), (0, 32)), norm=mpl.colors.LogNorm())
plt.savefig("grid_frequency.png")
