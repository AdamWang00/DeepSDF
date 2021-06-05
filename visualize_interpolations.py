import trimesh
from pyrender import Mesh, Node, Scene, Viewer, PerspectiveCamera
import numpy as np

if __name__ == "__main__":
    mesh_id1 = '5b767119-7fa8-48b0-a646-28af049330fc'
    mesh_id2 = '65b5062b-666a-3c3c-a465-bb43291d0e66'
    # mesh_id1 = '18992c9e-25a6-301d-8be8-95b7494e010f'
    # mesh_id2 = '5157b498-3cdf-4513-bd97-237358b1979e'
    interpolations_dir = './interpolations/' + mesh_id1 + '_' + mesh_id2 + '/'
    num_interpolations = 8

    scene = Scene()
    c = 0

    for i in range(num_interpolations):
        mesh_path = interpolations_dir + str(i+1) + '.ply'
        mesh = trimesh.load(mesh_path, process=False)
        scene.add_node(Node(mesh=Mesh.from_trimesh(mesh), translation=[i*2, 0, 0]))

    camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)

    Viewer(scene, use_raymond_lighting=True)