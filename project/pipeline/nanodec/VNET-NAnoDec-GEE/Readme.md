# GEE: A Gradient-based Explainable Variational Autoencoder for Network Anomaly Detection

## This is a private copy of https://github.com/munhouiani/GEE to enable working with data provided by VNET.

Details in blog post: https://blog.munhou.com/2020/07/12/Pytorch-Implementation-of-GEE-A-Gradient-based-Explainable-Variational-Autoencoder-for-Network-Anomaly-Detection/

## Installation

Requirements:
* [Anaconda for Linux](https://www.anaconda.com/products/distribution)
* A CUDA-enabled GPU

Create the conda environment:
```
conda env create --file environment.yml
```


## Usage

Activate the conda environment:

```
conda activate vnet-gee
```

To train a model, run

```
./run_pipeline_vae_vnet_train.sh [options] train_dirpath validation_dirpath model_dirpath
```

where `train_dirpath` and `validation_dirpath` refer to directory paths containing train and validation features, respectively.
`model_dirpath` is the path to a directory containing the trained model.

To predict new network flows using the trained model, run

```
./run_pipeline_vae_vnet_predict.sh [options] test_dirpath model_dirpath results_dirpath
```

where `test_dirpath` is a directory containing test data, `model_dirpath` is a directory path containing the trained model
and `results_dirpath` is a directory path containing the prediction results.

For the description of options for the scripts, run them with the `--help` option.
