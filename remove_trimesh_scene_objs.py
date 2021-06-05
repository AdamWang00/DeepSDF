import shutil
import trimesh
import os

directory = '../data/3D-FUTURE-model/chair/'
model_names = next(os.walk(directory))[1]

removals = 0
total = 0

for model_name in model_names:
    total += 1
    mesh = trimesh.load(directory + model_name + '/normalized_model.obj', process=False)
    if isinstance(mesh, trimesh.Scene):
        removals += 1
        print("Removing", model_name)
        shutil.rmtree(directory + model_name)

print("Removed", removals, "of", total, "models")