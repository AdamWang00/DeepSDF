import trimesh
from pyrender import Mesh, Node, Scene, Viewer, PerspectiveCamera
import numpy as np
from PIL import Image
import json
import os

def get_trimesh_and_uv(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(g for g in scene_or_mesh.geometry.values())
        )
        uv = np.concatenate(
            tuple(g.visual.uv for g in scene_or_mesh.geometry.values()),
            axis=0
        )
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
        uv = mesh.visual.uv
    return mesh, uv

if __name__ == "__main__":
    split_path = "./experiments/splits/nightstand.json"
    split_name = "3D-FUTURE-model"
    split_category = "category_2"
    experiment_names = ['nightstand1', 'nightstand_test5', 'nightstand_test6', 'nightstand_test7', 'nightstand_test8']
    epochs = ['1000', '1000', '1000', '1000', '1000']
    num_models = 16
    num_models_offset = 0

    scene = Scene()
    c = 0

    with open(split_path, "r") as f:
        model_ids = json.load(f)[split_name][split_category]

    for model_id in model_ids:
        if num_models_offset > 0:
            num_models_offset -= 1
            continue
        if num_models == 0:
            break
        num_models -= 1

        gt_path = os.path.join('../data', split_name+"_2021", split_category, model_id, 'normalized_model.obj')
        gt_texture_path = os.path.join('../data', split_name+"_2021", split_category, model_id, 'texture.png')
        
        try:
            gt_mesh, gt_uv = get_trimesh_and_uv(trimesh.load(gt_path, process=False))
            texture_im = Image.open(gt_texture_path)
            gt_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=gt_uv, image=texture_im).to_color()
            scene.add_node(Node(mesh=Mesh.from_trimesh(gt_mesh), translation=[0, c*3, 0]))
            for index, (experiment_name, epoch) in enumerate(zip(experiment_names, epochs)):
                gen_path = os.path.join('./experiments', experiment_name, 'TrainingMeshes', epoch, split_name, split_category, model_id + '.ply')
                gen_mesh = trimesh.load(gen_path, process=False)
                assert gt_mesh.visual.kind == 'vertex'
                scene.add_node(Node(mesh=Mesh.from_trimesh(gen_mesh), translation=[2*(index+1), c*3, 0]))
        except ValueError as e:
            print("[error]", str(e))
            continue

        print(c, model_id)
        c += 1

    camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)

    Viewer(scene, use_raymond_lighting=True)