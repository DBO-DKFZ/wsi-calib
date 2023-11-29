# Evaluating calibration for WSI classification

[![python](https://img.shields.io/badge/-Python_3.9-blue?logo=python&logoColor=white)](https://www.python.org/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![pytorch](https://img.shields.io/badge/PyTorch_1.13-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.8.6-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)

## Download data
- The TCGA colorectal cancer slides from TCGA-COAD and TCGA-READ studies can be downloaded from <https://portal.gdc.cancer.gov/>
- Access to the MCO slides can be requested at <https://www.sredhconsortium.org/sredh-datasets/mco-study-whole-slide-image-dataset>


## Setup directories and generate patch locations

1. Set environment variables. This can be done by adding the following lines to your `.bashrc` file:
```bash
export TCGA_ROOT_DIR=path/to/tcga/slides
export MCO_ROOT_DIR=path/to/mco/slides
export SLIDE_PROCESS_DIR=path/to/local/storage
```

2. Copy the content of the `data/` directory to `$SLIDE_PROCESS_DIR`:
```
|-- MCO-SCalib
|   |-- folds/
|   |-- slide_information.csv
|   |-- test_slides.csv
|
|-- TCGA-CRC-SCalib
|   |-- slide_information.csv
```

3. Compute patch locations by using our preprocessing repository at <https://github.com/DBO-DKFZ/wsi_preprocessing-crc> (tested with version 0.2 `git checkout v0.2`).
The file structure should then look as follows:
```
|-- MCO-SCalib
|   |-- folds/
|   |-- patches_256/
|   |-- patches_512/
|   |-- slide_information.csv
|   |-- test_slides.csv
|
|-- TCGA-CRC-SCalib
|   |-- patches_256/
|   |-- patches_512/
|   |-- slide_information.csv
```

## Setup conda environment for repository

We recommend to use *miniconda* to install the required dependencies for this project:

```bash
conda env create -f environment.yml
conda activate slidecalib
```

## Extract features for patch locations

The code for feature extraction is located in the `preprocess/` directory and the config files are located under `configs/`.

For feature extraction with the Ciga model, the weights need to be downloaded from <https://github.com/ozanciga/self-supervised-histopathology> and the corresponding `*.ckpt` file needs to be put in the `checkpoints/` directory.

An example command for extracting features for the MCO slides is 
```bash
python preprocess/extract_features.py --config configs/features/features_mco_512.yaml
```

As backend for loading the image patches from the slide we support both [Openslide](https://openslide.org/api/python/) and [Cucim](https://docs.rapids.ai/api/cucim/stable/).

## Train different models

Once the features are extracted, we can train the different model architectures on the extracted features. The corresponding config files are again provided in the `configs/` directory.

Before training the model, the `EXPERIMENT_LOCATION` needs to be set as environment variable (for example by adjusting `~/.bashrc`):
```bash
export EXPERIMENT_LOCATION=path/to/store/results
```

An example to train the [CLAM](https://github.com/mahmoodlab/CLAM) model with our pipeline is to run
```bash
python train.py --config configs/train_mco_clam.yaml
```

The example expects the following directory content:
```
|-- MCO-SCalib
|   |-- features_512_resnet18-ciga
|   |-- folds/
|   |-- patches_512/
|   |-- slide_information.csv
|   |-- test_slides.csv
|
|-- TCGA-CRC-SCalib
|   |-- features_512_resnet18-ciga
|   |-- patches_512/
|   |-- slide_information.csv
```

With the provided config file, the model predictions are automatically stored for the MCO test data, as well as for the TCGA slides.

To train the model on a different fold of MCO slides, the fold parameter can be set from the command line with:
```bash
python train.py --config configs/train_mco_clam.yaml --data.fold X
```
where X has to be in the range of [1, 5].

## Evaluate models
We provide a juypter notebook to perform evaluations on the stored predicitons in the `notebooks/` directory.

To better track notebooks with git, add the following lines to `.git/config`:
```git
[filter "strip-notebook-output"]
    clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
```