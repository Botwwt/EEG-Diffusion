# common environment for ad-hoc jobs: source envr.sh <gpu>   (sets $P to the project python)
R=${RAD_ROOT:?set RAD_ROOT}
export RAD_ROOT=$R HF_HOME=$R/cache/hf TORCH_HOME=$R/cache/torch XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1 PYTHONWARNINGS=ignore
P=$R/env/bin/python
