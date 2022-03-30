import trimesh
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

save_to = 'cloud.ply'
save_to_hist = 'cloud_hist.png'
# aa4714b2-e1f4-40b9-b680-5ee2d82c920b a2a8b471-48b3-4dd9-9ffd-dcf66d72dd58 192ac441-48b7-4559-bc02-7c532171b531 0ecff51f-5f8e-4ac7-b28a-d98d19446ff2
sdf_path = 'data/SdfSamples/3D-FUTURE-model/category_2/aa4714b2-e1f4-40b9-b680-5ee2d82c920b.npz'
# sdf_path = 'data/SdfSamples/3D-FUTURE-model/category_13/81aec6cd-34d7-4619-81e6-56bd1cdc1265.npz'
sdf_abs = 0.05
sdf_lower = -0.01
sdf_upper = 0.01
is_colorcat = True

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

# bin_width = 1
# plt.hist(sdf_all[:, 3], bins=np.arange(min(sdf_all[:, 3]),
#          max(sdf_all[:, 3]) + bin_width, bin_width))
# plt.savefig(save_to_hist)
# plt.show()

vertices = sdf_all[::, 0:3]
vertex_colors = sdf_all[:, 4:7]
# vertex_colors = np.clip(-sdf_all[::, [3, 3, 3]] * 100000 + 128, 0, 255)

if is_colorcat:
    def rgb_to_bin(r, g, b, dim=8):
        return r * dim * dim + g * dim + b

    annealing_temperature = 0.38 # 1.0 = mean, 0.01 = mode

    bin_to_rgb = np.zeros((512, 3))
    range_512 = np.arange(512, dtype=int)

    # e.g. [16, 48, ..., 240]
    offset = 16
    scale = 32

    bin_to_rgb[:, 0] = offset + scale * np.mod(np.floor_divide(range_512, 8 * 8), 8) # r
    bin_to_rgb[:, 1] = offset + scale * np.mod(np.floor_divide(range_512, 8), 8) # g
    bin_to_rgb[:, 2] = offset + scale * np.mod(range_512, 8) # b

    vertex_colors = vertex_colors.astype(np.int)

    bin_idx = rgb_to_bin(vertex_colors[:, 0], vertex_colors[:, 1], vertex_colors[:, 2]) # index of 512

    num_verts = vertex_colors.shape[0]
    vertex_colors_512 = np.zeros((num_verts, 512))
    idx_range = np.arange(num_verts)

    dirs3d = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ]
    vertex_colors_512[idx_range, bin_idx] += 1.0
    for direction in dirs3d:
        neighbor_bin_idx = rgb_to_bin(
            np.clip(vertex_colors[:, 0] + direction[0], 0, 7),
            np.clip(vertex_colors[:, 1] + direction[1], 0, 7),
            np.clip(vertex_colors[:, 2] + direction[2], 0, 7)
        )
        vertex_colors_512[idx_range, neighbor_bin_idx] += 0.00

    vertex_colors = np.exp(np.log(vertex_colors_512+1e-10) / annealing_temperature)
    vertex_colors = np.divide(vertex_colors.T, np.sum(vertex_colors, axis=1)).T
    vertex_colors = vertex_colors @ bin_to_rgb

mesh = trimesh.Trimesh(vertices=vertices, vertex_colors=vertex_colors)

mesh.export(file_obj=save_to, file_type='ply')

# grid_frequency = np.zeros((32, 32, 32))
# grid_indices = np.clip(((vertices[:, 0:3] + 1) * 16).astype(np.int), 0, 31)
# for grid_index in grid_indices:
#     grid_frequency[tuple(grid_index)] += 1

# print(min(grid_indices[:, 0]), max(grid_indices[:, 0]), min(grid_indices[:, 1]), max(
#     grid_indices[:, 1]), min(grid_indices[:, 2]), max(grid_indices[:, 2]))
# print(grid_frequency[0, 0, 0])
# print(np.min(grid_frequency), np.max(grid_frequency))

# plt.clf()

# slice_idx = grid_indices[:, 2] == 15
# plt.hist2d(grid_indices[slice_idx, 0], grid_indices[slice_idx, 1], bins=(32, 32), range=(
#     (0, 32), (0, 32)), norm=mpl.colors.LogNorm())
# plt.savefig("grid_frequency.png")
