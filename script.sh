# colorcat
# python preprocess_data.py --data_dir ./data --source ../data/3D-FUTURE-model_manifold/ --name 3D-FUTURE-model_manifold --threads 12 --split ./experiments/splits/nightstandM.json
# python preprocess_data.py --data_dir ./data --source ../data/3D-FUTURE-model/ --name 3D-FUTURE-model --threads 12 --surface --split ./experiments/splits/nightstand.json --skip
# python preprocess_color.py --data_dir ./data --source ../data/3D-FUTURE-model/ --name 3D-FUTURE-model --split ./experiments/splits/nightstand.json --ignore_sdf
# python preprocess_colorcat.py --data_dir ./data --source ../data/3D-FUTURE-model_manifold/ --name 3D-FUTURE-model_manifold --name_surface 3D-FUTURE-model --split ./experiments/splits/nightstandM.json --ignore_surface

# python train_deep_sdf_colorcat_coarse.py -e experiments/nightstand3Ma
# python generate_training_meshes_color.py --cat -n -m 32 -c 1000 -e experiments/nightstand3Ma
# python evaluate_color.py -d ./data -n -m 32 -c 1000 -e experiments/nightstand3Ma --name_surface 3D-FUTURE-model

python train_deep_sdf_colorcat_coarse.py -e experiments/nightstand4
python generate_training_meshes_colorcat.py -n -m 32 -c 1000 -e experiments/nightstand4
python evaluate_color.py -n -m 32 -c 1000 -e experiments/nightstand4 --name_surface 3D-FUTURE-model

# for dir in experiments/nightstand3*/; do
#     python evaluate_color.py -d ./data -n -m 32 -c 1000 -e "$dir" --name_surface 3D-FUTURE-model
# done