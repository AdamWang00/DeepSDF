import json
import os

dataset_name = "3D-FUTURE-model_2021"
split_name = "chair2"
category_dir_ids = [18, 19, 21, 22, 27, 30, 37]

split = {}
split[dataset_name] = {}

for category_dir_id in category_dir_ids:
    split_category_name = "category_" + str(category_dir_id)
    category_dir = os.path.join('../data', dataset_name, split_category_name)
    model_names = next(os.walk(category_dir))[1]

    split[dataset_name][split_category_name] = []

    for model_name in model_names:
        split[dataset_name][split_category_name].append(model_name)

    with open(f'./experiments/splits/{split_name}.json', 'w') as outfile:
        json.dump(split, outfile)