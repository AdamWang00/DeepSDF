rsync -r -e ssh awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/experiments/nightstand4M* ./experiments

# rsync -r -e ssh ./experiments/nightstand3Mg* awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/experiments

# rsync -r -e ssh awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/*.ply .
# rsync -r -e ssh awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/color_bins .

# rsync -r -e ssh ./data/SdfSamples/3D-FUTURE-model_manifold/category_2 awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/data/SdfSamples/3D-FUTURE-model_manifold

# rsync -r -e ssh \
# 	--exclude 'data' \
# 	--exclude 'experiments' \
# 	--exclude 'pngs' \
# 	--exclude 'scratch_work' \
# 	--exclude 'venv' \
# 	./* \
# 	awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF