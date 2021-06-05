import trimesh
from trimesh.viewer import SceneViewer
import numpy as np
import matplotlib.pyplot as plt

sdf_path = 'data/SdfSamplesOld/3D-FUTURE-model/chair/0da2a2ff-fd2d-3cd3-8dc9-9303c7a80814.npz'

sdf = np.load(sdf_path)
sdf_pos = sdf["pos"]
sdf_neg = sdf["neg"]
sdf_all = np.concatenate((sdf_pos, sdf_neg), axis=0)
print("total number of sdf samples:", sdf_all.shape[0])
sdf_all = sdf_all[np.abs(sdf_all[:, 3]) < 0.1, :]
print("number of samples within threshold:", sdf_all.shape[0])

# bin_width = 0.01
# plt.hist(sdf_all[:, 3], bins=np.arange(min(sdf_all[:, 3]), max(sdf_all[:, 3]) + bin_width, bin_width))
# plt.show()

vertices = sdf_all[:, 0:3]
vertex_colors = sdf_all[:, 4:7]

mesh = trimesh.Trimesh(vertices=vertices, vertex_colors=vertex_colors)

mesh.export(file_obj='cloudOld.ply', file_type='ply')