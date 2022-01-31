import trimesh
from trimesh.viewer import SceneViewer
import numpy as np
import matplotlib.pyplot as plt

sdf_path = 'data/SdfSamples/3D-FUTURE-model_2021/category_18/0a4431a4-7214-32eb-88ff-6954da29a2da.npz'
sdf_abs_threshold = 10

sdf = np.load(sdf_path)
sdf_pos = sdf["pos"]
sdf_neg = sdf["neg"]
sdf_all = np.concatenate((sdf_pos, sdf_neg), axis=0)
print("total number of sdf samples:", sdf_all.shape[0])
sdf_all = sdf_all[np.abs(sdf_all[:, 3]) < sdf_abs_threshold, :]
print("number of samples within threshold:", sdf_all.shape[0])
print(np.min(sdf_all[:, 0]), np.max(sdf_all[:, 0]), np.min(sdf_all[:, 1]), np.max(sdf_all[:, 1]), np.min(sdf_all[:, 2]), np.max(sdf_all[:, 2]))

bin_width = 0.01
plt.hist(sdf_all[:, 3], bins=np.arange(min(sdf_all[:, 3]), max(sdf_all[:, 3]) + bin_width, bin_width))
plt.show()

vertices = sdf_all[:, 0:3]
vertex_colors = sdf_all[:, 4:7]

mesh = trimesh.Trimesh(vertices=vertices, vertex_colors=vertex_colors)

mesh.export(file_obj='cloud.ply', file_type='ply')