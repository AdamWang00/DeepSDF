import json
import os

dataset_name = "3D-FUTURE-model"
super_category = "Chair"
split_category_name = "chair"
split_size = -1

directory = os.path.join('../data', dataset_name, split_category_name)
model_names = next(os.walk(directory))[1]

split = {}
split[dataset_name] = {}
split[dataset_name][split_category_name] = []

c = 0
for model_name in model_names:
    split[dataset_name][split_category_name].append(model_name)
    c += 1
    if c == split_size:
        break

with open('split.json', 'w') as outfile:
    json.dump(split, outfile)