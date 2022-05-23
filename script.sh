DEEPSDF_CATEGORY_NAME="cabinet"
DEEPSDF_SPLIT_NAME="cabinetM"
DEEPSDF_EXPERIMENT_NAME="nightstand5b"

# python preprocess_data.py \
# 	--data_dir ./data \
# 	--source ../data/3D-FUTURE-model_manifold/ \
# 	--name 3D-FUTURE-model_manifold \
# 	--threads 12 \
# 	--split ./experiments/splits/${DEEPSDF_SPLIT_NAME}.json \
# 	--skip
# python preprocess_data.py \
# 	--data_dir ./data \
# 	--source ../data/3D-FUTURE-model/ \
# 	--name 3D-FUTURE-model \
# 	--threads 12 \
# 	--split ./experiments/splits/bed.json \
# 	--surface \
#	--skip
# python preprocess_colorcat.py \
# 	--data_dir ./data \
# 	--source ../data/3D-FUTURE-model_manifold/ \
# 	--name 3D-FUTURE-model_manifold \
# 	--name_surface 3D-FUTURE-model \
# 	--split ./experiments/splits/${DEEPSDF_SPLIT_NAME}.json \
# 	--ignore_surface \
# 	--color_bins color_bins/${DEEPSDF_CATEGORY_NAME}/clusters.npz

python train_deep_sdf_colorcat_coarse_acai.py -e experiments/${DEEPSDF_EXPERIMENT_NAME}
python generate_training_meshes_colorcat.py -n -m 32 -c 1000 -e experiments/${DEEPSDF_EXPERIMENT_NAME}
python evaluate_color.py -n -m 32 -c 1000 -e experiments/${DEEPSDF_EXPERIMENT_NAME} --name_surface 3D-FUTURE-model

# for dir in experiments/nightstand3*/; do
#     python evaluate_color.py -d ./data -n -m 32 -c 1000 -e "$dir" --name_surface 3D-FUTURE-model
# done