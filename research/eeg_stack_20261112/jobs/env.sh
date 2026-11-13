#!/bin/bash
R=${RAD_ROOT:?set RAD_ROOT}; cd $R; export UV_CACHE_DIR=$R/cache/uv TMPDIR=$R/tmp
[ -x env/bin/python ] || uv venv --python /usr/bin/python3.10 env 2>&1 | tee logs/env.log
uv pip install --python env/bin/python -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision transformers diffusers datasets open_clip_torch omegaconf pathlib2 accelerate tqdm timm scikit-image einops huggingface_hub safetensors scipy pandas pyarrow ftfy regex 2>&1 | tee -a logs/env.log
uv pip install --python env/bin/python git+https://github.com/openai/CLIP.git 2>&1 | tee -a logs/env.log
echo ENV_DONE | tee -a logs/env.log
