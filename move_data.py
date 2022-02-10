import os
import shutil
import json

split_path = os.path.join("experiments", "splits", "nightstand.json")
split_name = "3D-FUTURE-model"

ssd_data_path = "data"
hdd_data_path = "/mnt/hdd1/awang_scene_synth/deepsdf/data"
subdirs = ["SdfSamples", "SurfaceSampleFaces", "SurfaceSamples"]

ssd_to_hdd = False  # true = ssd->hdd, false = hdd->ssd

with open(split_path, 'rb') as f:
    split_json = json.load(f)
categories = list(split_json[split_name].keys())

confirm = input(f"move {categories} ? (y)")
if (confirm == "y"):
    for category in categories:
        for subdir in subdirs:
            if ssd_to_hdd:
                path_from = os.path.join(
                    ssd_data_path, subdir, split_name, category)
                path_to = os.path.join(hdd_data_path, subdir, split_name)
            else:
                path_from = os.path.join(
                    hdd_data_path, subdir, split_name, category)
                path_to = os.path.join(ssd_data_path, subdir, split_name)
            print(path_from, "==>", path_to)
            os.makedirs(path_to, exist_ok=True)
            if os.path.exists(path_from):
                shutil.move(path_from, path_to)
            else:
                print("[ERROR] nonexistent:", path_from)