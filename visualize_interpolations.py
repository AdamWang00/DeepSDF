import trimesh
from pyrender import Mesh, Node, Scene, Viewer, PerspectiveCamera
import numpy as np
import os
import time


interpolations_version = "2" # "" is default
experiment_directory = "experiments/nightstand4"
epoch = "1000"
mesh_id1 = "c2f29b88-e421-4b9c-ab06-50e9e83cf8a1"
mesh_id2 = "30d524b6-ad13-440f-a1ac-6d564213c2f7"
# mesh_id2 = "b6ec33de-c372-407c-b5b7-141436b02e7b"

viewport_w = 900
viewport_h = 900
fps = 6

interpolations_dir = os.path.join(
    experiment_directory,
    'Interpolations' + interpolations_version,
    epoch,
    mesh_id1 + '_' + mesh_id2
)
num_interpolations = len(os.listdir(interpolations_dir))
print(num_interpolations, "interpolations found")

scene = Scene()

mesh_node_list = []
for i in range(num_interpolations):
    # if i % 2 == 0: # use if memory cannot store all interpolations
    #     continue
    print(i+1, "of", num_interpolations)
    mesh_path = os.path.join(interpolations_dir, str(i+1) + '.ply')
    mesh = trimesh.load(mesh_path, process=False)
    print(mesh.vertices.shape)
    mesh_node = Node(mesh=Mesh.from_trimesh(mesh, smooth=False))
    mesh_node.mesh.is_visible = False
    mesh_node_list.append(mesh_node)
    scene.add_node(mesh_node)

camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=viewport_w/viewport_h)
camera_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)

viewer = Viewer(scene, use_raymond_lighting=True, viewport_size=(viewport_w,viewport_h), run_in_thread=True)

while True:
    viewer.render_lock.acquire()
    mesh_node_list[-1].mesh.is_visible = False
    mesh_node_list[0].mesh.is_visible = True
    viewer.render_lock.release()

    time.sleep(1)
    for i in range(1, len(mesh_node_list)):
        viewer.render_lock.acquire()
        mesh_node_list[i - 1].mesh.is_visible = False
        mesh_node_list[i].mesh.is_visible = True
        viewer.render_lock.release()
        time.sleep(1/fps)
    time.sleep(1)