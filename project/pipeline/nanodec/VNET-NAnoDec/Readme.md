# NAnoDec: Network Anomaly Detection

Network anomaly detection from flows using machine learning algorithms

## Installation

Requirements:
* [Anaconda for Linux](https://www.anaconda.com/products/distribution)
* A CUDA-enabled GPU
* [VNET-NAnoDec-GEE](https://github.com/kinit-sk/VNET-NAnoDec-GEE) repository on the same directory level as this repository.

Run `setup.sh` to install dependencies and conda environments.

## Usage

Activate the conda environment:

```
conda activate vnet
```

To train an anomaly detector, run

```
./scripts/run_anomaly_detection.sh [options] path 'train'
```

where `path` is a directory containing flow files exported from nProbe.

To predict anomalies with a trained anomaly detector, run

```
./scripts/run_anomaly_detection.sh [options] path 'predict'
```

where `path` is a directory containing new flow files subject to preprocessing and prediction.

To re-create clusters of IP addresses (network node profiles), run 

```
VNET-NAnoDec/scripts/run_clustering.sh <path>
```

where `path` is a directory containing flow files exported from nProbe.

For the description of options for the scripts, run them with the `--help` option.
