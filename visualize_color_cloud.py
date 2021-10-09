import trimesh
from trimesh.viewer import SceneViewer
import numpy as np
import matplotlib.pyplot as plt

sdf_path = 'data/SdfSamples/3D-FUTURE-model/category_32/02bfd47f-1ca7-4ae4-b430-58caf73cc4a3.npz'

sdf = np.load(sdf_path)
sdf_pos = sdf["pos"]
sdf_neg = sdf["neg"]
sdf_all = np.concatenate((sdf_pos, sdf_neg), axis=0)
print("total number of sdf samples:", sdf_all.shape[0])
sdf_all = sdf_all[np.abs(sdf_all[:, 3]) < 10, :]
print("number of samples within threshold:", sdf_all.shape[0])

bin_width = 0.01
plt.hist(sdf_all[:, 3], bins=np.arange(min(sdf_all[:, 3]), max(sdf_all[:, 3]) + bin_width, bin_width))
plt.show()

vertices = sdf_all[:, 0:3]
vertex_colors = sdf_all[:, 4:7]

mesh = trimesh.Trimesh(vertices=vertices, vertex_colors=vertex_colors)

mesh.export(file_obj='cloud.ply', file_type='ply')