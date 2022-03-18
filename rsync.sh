# rsync -r -e ssh awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/experiments/bed2MO* ./experiments

rsync -r -e ssh awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/*.ply .

# rsync -r -e ssh ./data/SdfSamples/3D-FUTURE-model_manifold awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/data/SdfSamples

# rsync -r -e ssh \
# 	--exclude 'data' \
# 	--exclude 'experiments' \
# 	--exclude 'pngs' \
# 	--exclude 'scratch_work' \
# 	--exclude 'venv' \
# 	./* \
# 	awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF