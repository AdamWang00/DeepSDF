import trimesh
from pyrender import Mesh, Node, Scene, Viewer, PerspectiveCamera
import numpy as np
from PIL import Image

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

    experiment_names = ['chair5']
    epochs = ['2000']

    model_ids = ["00a91a81-fc73-4625-8298-06ecd55b6aaa", "01839053-f5e1-4165-b492-85836be84194", "0bca869d-b715-4795-b911-9de38f642f22", "0cd722cb-4cf5-4026-9906-187e51d78c67", "147fb31a-fcf2-4b58-8c0a-6ff888abff04", "18992c9e-25a6-301d-8be8-95b7494e010f", "18b5b81f-7860-449f-87fb-5552e3a96fb9", "19f47b83-3a3e-4f63-b9c3-a421468dbd7f", "202be378-8e97-471d-9041-8f24d8c3df00", "26d454a6-308d-3c4f-a871-df9e29e70d63", "2a17cf5f-ec24-4ee6-87fa-a504712a45f6", "44b8dd59-8e5c-4fb3-a3ca-dba9b7419cd2", "4b600aed-51ed-445c-8c08-283ea0d5b53a", "5157b498-3cdf-4513-bd97-237358b1979e", "57580a13-bef0-44fd-b611-863ea4a3aad8", "5949665c-499d-413e-ba9c-b8595ee700e5", "5b767119-7fa8-48b0-a646-28af049330fc", "5c743f3d-6647-4f2b-9ecf-561b119cc104", "5ec8be8f-f953-40e1-9a1e-9710b83c176c", "65b5062b-666a-3c3c-a465-bb43291d0e66", "77dcc4f8-98ae-43aa-994b-c1874f9a00b7", "932caad9-022d-4a6d-a5e9-32145b16833d", "99e35a9e-b125-4334-949e-5845f987a53f", "b3b7e415-b77b-4dc5-8bdf-7673a00f2b0d", "b98f48d1-9b8b-430a-b803-c24d87a77386", "bd736985-025a-4d60-865f-d8cab731280b", "c744f59f-6133-4131-a131-88209820de61", "d72b7ee3-e39b-42e7-b796-e7ec9d23fbd7", "da47801b-a9c7-4370-92b5-2c13bdfd6c42", "ecf6d756-e5f7-4e54-86ff-5595f77ccb01", "f7b9d8f5-1458-4a0d-a753-0a2a4f377e31", "fad585b9-df56-4b87-8a82-5130d5273c9b"]

    # ["19f47b83-3a3e-4f63-b9c3-a421468dbd7f", "2a17cf5f-ec24-4ee6-87fa-a504712a45f6", "bd736985-025a-4d60-865f-d8cab731280b", "00a91a81-fc73-4625-8298-06ecd55b6aaa", "d72b7ee3-e39b-42e7-b796-e7ec9d23fbd7", "5949665c-499d-413e-ba9c-b8595ee700e5", "0bca869d-b715-4795-b911-9de38f642f22", "01839053-f5e1-4165-b492-85836be84194", "4b600aed-51ed-445c-8c08-283ea0d5b53a", "65b5062b-666a-3c3c-a465-bb43291d0e66", "5b767119-7fa8-48b0-a646-28af049330fc", "44b8dd59-8e5c-4fb3-a3ca-dba9b7419cd2", "18992c9e-25a6-301d-8be8-95b7494e010f", "26d454a6-308d-3c4f-a871-df9e29e70d63", "932caad9-022d-4a6d-a5e9-32145b16833d", "202be378-8e97-471d-9041-8f24d8c3df00", "147fb31a-fcf2-4b58-8c0a-6ff888abff04", "5c743f3d-6647-4f2b-9ecf-561b119cc104", "77dcc4f8-98ae-43aa-994b-c1874f9a00b7", "3e36f8bc-83aa-497b-8179-92c76f2d7d56", "f7b9d8f5-1458-4a0d-a753-0a2a4f377e31", "18b5b81f-7860-449f-87fb-5552e3a96fb9", "b98f48d1-9b8b-430a-b803-c24d87a77386", "5ec8be8f-f953-40e1-9a1e-9710b83c176c", "5157b498-3cdf-4513-bd97-237358b1979e", "b3b7e415-b77b-4dc5-8bdf-7673a00f2b0d", "da47801b-a9c7-4370-92b5-2c13bdfd6c42", "57580a13-bef0-44fd-b611-863ea4a3aad8", "fad585b9-df56-4b87-8a82-5130d5273c9b"]

    scene = Scene()
    c = 0

    for model_id in model_ids:
        gt_path = '../data/3D-FUTURE-model/chair/' + model_id + '/normalized_model.obj'
        gt_texture_path = '../data/3D-FUTURE-model/chair/' + model_id + '/texture.png'
        
        try:
            gt_mesh, gt_uv = get_trimesh_and_uv(trimesh.load(gt_path, process=False))
            texture_im = Image.open(gt_texture_path)
            gt_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=gt_uv, image=texture_im).to_color()
            scene.add_node(Node(mesh=Mesh.from_trimesh(gt_mesh), translation=[0, c*3, 0]))
            for index, (experiment_name, epoch) in enumerate(zip(experiment_names, epochs)):
                gen_path = './experiments/' + experiment_name + '/TrainingMeshes/' + epoch + '/3D-FUTURE-model/chair/' + model_id + '.ply'
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