# rsync -r -e ssh awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/experiments/nightstand4M* ./experiments

# rsync -r -e ssh ./experiments/nightstand3Mg* awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/experiments

# rsync -r -e ssh ./experiments/splits awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/experiments

# rsync -r -e ssh awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/*.ply .
# rsync -r -e ssh awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/color_bins .

# rsync -r -e ssh ./data/SurfaceSamples/3D-FUTURE-model/* awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/data/SurfaceSamples/3D-FUTURE-model
# rsync -r -e ssh ./data/SurfaceSampleFaces/3D-FUTURE-model/* awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/data/SurfaceSampleFaces/3D-FUTURE-model

rsync -r -e ssh /mnt/hdd1/awang_scene_synth/deepsdf/data/SurfaceSamples/3D-FUTURE-model/* awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/data/SurfaceSamples/3D-FUTURE-model
rsync -r -e ssh /mnt/hdd1/awang_scene_synth/deepsdf/data/SurfaceSampleFaces/3D-FUTURE-model/* awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF/data/SurfaceSampleFaces/3D-FUTURE-model

# rsync -r -e ssh \
# 	--exclude 'data' \
# 	--exclude 'experiments' \
# 	--exclude 'pngs' \
# 	--exclude 'scratch_work' \
# 	--exclude 'venv' \
# 	./* \
# 	awang158@ssh.ccv.brown.edu:/users/awang158/scratch/DeepSDF