import json
import subprocess
import trimesh
import numpy as np

def get_trimesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(g for g in scene_or_mesh.geometry.values())
        )
        uv = np.concatenate(
            tuple(g.visual.uv for g in scene_or_mesh.geometry.values()),
            axis=0
        )
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv)
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

# def trimesh_remove_floor(scene_or_mesh):
#     if isinstance(scene_or_mesh, trimesh.Scene):
#         mesh_list = []
#         uv_list = []

#         for g in scene_or_mesh.geometry.values():
#             if g.vertices.shape[0] in [4, 6]: # the rectangular floor
#                 print("FLOOR FOUND", g)
#             else:
#                 print("non-floor", g)
#                 mesh_list.append(g)
#                 uv_list.append(g.visual.uv)

#         mesh = trimesh.util.concatenate(mesh_list)
#         uv = np.concatenate(uv_list, axis=0)
#         mesh.visual.uv = uv
#     else:
#         print("not a scene")
#         assert(isinstance(scene_or_mesh, trimesh.Trimesh))
#         mesh = scene_or_mesh
#         cc = trimesh.graph.connected_components(mesh.face_adjacency)
#         print("connected components:", len(cc))
#         for component in cc: # a component is a list of indices of faces that are connected
#             component_is_floor = True
#             component_faces = mesh.faces[component]
#             for component_face in component_faces:
#                 vertices = mesh.vertices[component_face]
#                 if not np.all(vertices[:, 1] == vertices[0, 1]): # different height
#                     component_is_floor = False
#                     break
#             if component_is_floor:
#                 print("FLOOR COMPONENT FOUND")
#                 mask = np.ones(len(mesh.faces), dtype=np.bool)
#                 mask[component] = False
#                 mesh.update_faces(mask)
#                 break
#     return mesh

# with open('fix_floor.json') as f:
#     fix_floor = json.load(f)["data"]

# with open('fixed.json') as f:
#     fixed = json.load(f)["data"]

# num_chairs = len(fix_floor)

# i = 0
# for chair_id in fix_floor:
#     print(i, "of", num_chairs, chair_id)
#     i += 1

#     if chair_id in fixed:
#         print("Already fixed", chair_id)
#         continue

#     chair_path = '../data/3D-FUTURE-model/chair/' + chair_id + '/normalized_model.obj'
#     process = subprocess.Popen(['meshlab', chair_path], stdout=subprocess.DEVNULL)
#     output, error = process.communicate()

#     m = trimesh.load(chair_path, process=False)
#     m = trimesh_remove_floor(m)
#     m.export(file_obj='temp.obj', file_type='obj')
#     process = subprocess.Popen(['meshlab', 'temp.obj'], stdout=subprocess.DEVNULL)
#     output, error = process.communicate()

#     a = input("bad?")
#     if len(a) == 0:
#         fixed.append(chair_id)
#         m.export(file_obj=chair_path, file_type='obj')

# print("fixed", fixed)

# with open('fixed.json', 'w') as f:
#     json.dump({"data": fixed}, f)


with open('fix_rotate_2.json') as f:
    fix_rotate = json.load(f)["data"]

# with open('fixed.json') as f:
#     fixed = json.load(f)["data"]

num_chairs = len(fix_rotate)

i = 0
for chair_id in fix_rotate:
    print(i, "of", num_chairs, chair_id)
    i += 1

    # if chair_id in fixed:
    #     print("Already fixed", chair_id)
    #     continue

    chair_path = '../data/chair/' + chair_id + '/normalized_model.obj'
    new_path = '../data/chair/' + chair_id + '/normalized_model.obj'
    process = subprocess.Popen(['meshlab', chair_path], stdout=subprocess.DEVNULL)
    output, error = process.communicate()

    angle = input("degrees?")
    while len(angle) > 0:
        try:
            angle = float(angle) * np.pi / 180
            rot_mat = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
            m = trimesh.load(chair_path, process=False)
            m = get_trimesh(m)
            m.vertices = np.tensordot(m.vertices, rot_mat, axes=([1], [1]))
            m.export(file_obj='temp.obj', file_type='obj')
            process = subprocess.Popen(['meshlab', 'temp.obj'], stdout=subprocess.DEVNULL)
            output, error = process.communicate()
        except Exception as e:
            print(e)
        finally:
            angle = input("degrees?")

    a = input("bad?")
    if len(a) == 0:
        # fixed.append(chair_id)
        m.export(file_obj=new_path, file_type='obj')

# print("fixed", fixed)

# with open('fixed.json', 'w') as f:
#     json.dump({"data": fixed}, f)