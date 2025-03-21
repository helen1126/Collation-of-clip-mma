# Collation-of-clip-mma
Historical code collation on Multi-Modal Adapter for Vision-Language Models, CLIP Multi-Modal Adapter
Code for a paper ["Multi-Modal Adapter for Vision-Language Models"](https://arxiv.org/abs/2409.02958)
---

# requirements
python==3.11
--aiohttp == 3.9.5 
aiosignal==1.3.1 
attrs==23.2.0 
certifi==2024.2.2 
charset-normalizer==3.3.2 
colorama==0.4.6
contourpy==1.2.1 
cycler==0.12.1 
datasets==2.19.0 
dill==0.3.8 
evaluate==0.4.1 
filelock==3.13.4 
fonttools==4.51.0 
frozenlist==1.4.1 
fsspec==2024.3.1 
fsspec[http]==2024.3.1 
ftfy==6.2.0 
huggingface-hub==0.22.2 
idna==3.7 
jinja2==3.1.3 
joblib==1.4.0 
kiwisolver==1.4.5 
markupsafe==2.1.5 
matplotlib==3.8.4 
mpmath==1.3.0 
multidict==6.0.5 
multiprocess==0.70.16 
networkx==3.3 
numpy==1.26.4 
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105 
nvidia-cuda-runtime-cu12==12.1.105 
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106 
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106 
nvidia-nccl-cu12==2.19.3
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.1.105 
openai-clip==1.0.1 
packaging==24.0  
pandas==2.2.2  
pillow==10.3.0  
pyarrow-hotfix==0.6 
pyarrow==15.0.2 
pyparsing==3.1.2 
python-dateutil==2.9.0.post0 
pytz==2024.1
pyyaml==6.0.1 
regex==2024.4.16 
requests==2.31.0 
responses==0.18.0 
safetensors==0.4.3 
scikit-learn==1.4.2
scipy==1.13.0
seqeval==1.2.2 
six==1.16.0 
sympy==1.12 
threadpoolctl==3.4.0 
tokenizers==0.15.2
torch==2.2.2
torchvision==0.17.2
tqdm==4.66.2 
transformers==4.38.2 
triton==2.2.0 
typing-extensions==4.11.0
tzdata==2024.1 
urllib3==2.2.1 
wcwidth==0.2.13 
xxhash==3.4.1 
yarl==1.9.4 

---
## Setup

1. Setup required python version using your preferred method (e.g. pyenv, virtualenv, etc.). For pyenv users:

```bash
pyenv install 3.11.6
pyenv local 3.11.6
```

2. Install poetry if needed following the instructions at https://python-poetry.org/docs/#installation
3. Install dependencies:

```bash
poetry install
```

4. Set up the pre-commit hooks:

```bash
poetry run pre-commit install
```

### Setting up the environment for Conda

First, create a new conda environment from the `environment.yml` file:

```bash
conda env create -f environment.yml
```

Then, activate the environment:

```bash
conda activate fomo
```

After install dependencies with pip using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Datasets

For some of the datasets, you have to download them from the original source. The datasets are not included in this repository.

### Stanford Cars

Download the dataset from https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder to `data/stanford-cars`.
