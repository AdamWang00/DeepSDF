import trimesh
from pyrender import Mesh, Node, Scene, Viewer, PerspectiveCamera
import numpy as np
from PIL import Image
import json
import os


if __name__ == "__main__":
    split_path = "./experiments/splits/nightstand.json"
    split_name_gt = "3D-FUTURE-model"
    # split_category_gt = "category_X"
    # split_categories = ['category_X', 'category_X']
    experiment_names = ['nightstand4Ma', 'nightstand4Mc', 'nightstand4Md']
    epochs = ['1000'] * len(experiment_names)
    split_names = ['3D-FUTURE-model_manifold'] * len(experiment_names)
    num_models = 16
    num_models_offset = 0
    color = True

    scene = Scene()
    viewport_w = 1600
    viewport_h = 900
    c = 0

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

    with open(split_path, "r") as f:
        split = model_ids = json.load(f)[split_name_gt]
        split_category_gt = next(iter(split))
        split_categories = [split_category_gt] * len(split_names)
        model_ids = split[split_category_gt]

    for model_id in model_ids:
        if num_models_offset > 0:
            num_models_offset -= 1
            continue
        if num_models == 0:
            break
        num_models -= 1

        gt_path = os.path.join(
            '../data', split_name_gt, split_category_gt, model_id, 'normalized_model.obj')
        gt_texture_path = os.path.join(
            '../data', split_name_gt, split_category_gt, model_id, 'texture.png')

        try:
            gt_mesh, gt_uv = get_trimesh_and_uv(
                trimesh.load(gt_path, process=False))
            texture_im = Image.open(gt_texture_path)
            gt_mesh.visual = trimesh.visual.texture.TextureVisuals(
                uv=gt_uv, image=texture_im).to_color()
            scene.add_node(Node(mesh=Mesh.from_trimesh(
                gt_mesh), translation=[0, c*3, 0]))
            for index, (experiment_name, epoch, split_name, split_category) in enumerate(zip(experiment_names, epochs, split_names, split_categories)):
                gen_path = os.path.join('./experiments', experiment_name, 'TrainingMeshes',
                                        epoch, split_name, split_category, model_id + '.ply')
                gen_mesh = trimesh.load(gen_path, process=False)
                assert gt_mesh.visual.kind == 'vertex'
                if not color:
                    gen_mesh.visual.vertex_colors = (128, 128, 128, 255)
                scene.add_node(Node(mesh=Mesh.from_trimesh(
                    gen_mesh), translation=[2.5*(index+1), c*3, 0]))
        except ValueError as e:
            print("[error]", str(e))
            continue

        print(c+1, model_id)
        c += 1

    camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=viewport_w/viewport_h)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)

    Viewer(scene, use_raymond_lighting=True, viewport_size=(viewport_w,viewport_h))
