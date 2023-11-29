
# Intellectual Property Protection of Diffusion Models via the Watermark Diffusion Process

This repository is the official implementation of the [Intellectual Property Protection of Diffusion Models via the Watermark Diffusion Process](https://arxiv.org/abs/2306.03436). 
It is based on the parent repository [improved-diffusion](https://github.com/openai/improved-diffusion) and [guided-diffusion](https://github.com/openai/guided-diffusion).

## 1. Requirements
<!-- We use Anaconda3 to manage the environment:
```shell
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
```
We highly recommend you use the [Mamba](https://github.com/conda-forge/miniforge#mambaforge) to install the requirements. (choosing your coda version)

```shell
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
```
install the environment: -->
We use [Mamba](https://github.com/conda-forge/miniforge#mambaforge) to manage the environment requirements.

```shell
mamba create -n wdm python=3.10
mamba activate wdm
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install tensorflow-gpu matplotlib blobfile mpi4py tqdm PyYAML pandas
```

and intall the package:
```shell
pip install -e .
```


## 2. Prepare Datasets

To prepare for the task and watermark dataset, run the following:
### 2.1 CIFAR-10 as the Task Dataset
```shell
python cifar10.py
```
### 2.2 sWDM Setting
```shell
python single_wm.py
```
### 2.2 mWDM Setting (class five in MNIST)
```shell
python multiple_wm.py
```

## 3. Baseline Model Training
To train the baseline models in the paper, configure ``configs/train_mse.yaml`` and run this command:

```shell
python scripts/image_train.py --p configs/train_mse.yaml
```
or multi-GPU version (replace $NUM_GPUS)
```shell
mpiexec -n $NUM_GPUS python scripts/image_train.py --p configs/train_mse.yaml
```
## 4. Watermark Embedding
To embed the watermark data into the baseline models, configure ``configs/train_mse_wdp.yaml`` and run this command:

```shell
python scripts/image_train.py --p configs/train_mse_wdp.yaml
```
## 5. Task Sampling & Watermark Extraction

To sample from the task, configure the ``configs/sample_mse.yaml`` run this command:

```shell
python scripts/image_sample.py --p configs/sample_mse.yaml
```
To extract the watermark from the suspected model, configure the ``configs/sample_mse_wdp.yaml`` run this command:

```shell
python scripts/image_sample.py --p configs/sample_mse_wdp.yaml
```

## 6. Evaluation
The evaluation is based on the [guided diffusion](https://github.com/openai/guided-diffusion) repo.
```shell
python scripts/evaluator.py <reference_batch> <evaluated_batch>
```