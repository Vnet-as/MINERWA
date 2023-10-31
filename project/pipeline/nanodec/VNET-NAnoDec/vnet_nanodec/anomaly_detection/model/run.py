"""Runs a model training and evaluation procedure.

Usage:

python run.py config

Parameters:

"""

# Pandas needs to be imported before torch, otherwise an error occurs
import pandas as pd
import torch
import sys
import yaml
from torch.utils.data import DataLoader

import defines
import dataset
import ml_stats
import model


# Set the seed for replicability
import random

random.seed(42)
torch.manual_seed(42)


def main(args: list) -> None:
    if len(args) != 2:
        raise Exception('Invalid number of arguments.')

    model_config_path = args[1]     # Configuration file path
    config            = None        # Loaded configuration as a dictionary
    device            = None        # Device to store models and tensors on
    autoenc           = None        # Autoencoder model to use for precitions

    # Load model configuration
    with open(model_config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Obtain data paths
    data_train_path = config[defines.CONFIG_KEY_MAIN]['path_train']
    data_test_path  = config[defines.CONFIG_KEY_MAIN]['path_test']
    data_valid_path = config[defines.CONFIG_KEY_MAIN]['path_valid']

    # Determine device to use and initialize the model upon it
    if config[defines.CONFIG_KEY_MAIN]['use_cuda']:
        device = torch.device(defines.DEVICE_CUDA_NAME if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    autoenc = model.Autoencoder(device, config)

    # Create dataset instances and their associated data loaders
    data_train = dataset.FlowsDataset(data_train_path, config[defines.CONFIG_KEY_DATASET], True)
    data_test  = dataset.FlowsDataset(data_test_path, config[defines.CONFIG_KEY_DATASET], False)
    data_valid = dataset.FlowsDataset(data_valid_path, config[defines.CONFIG_KEY_DATASET], False)

    loader_train = DataLoader(data_train, **defines.DATALOADER_SETTINGS)
    loader_test  = DataLoader(data_test, **defines.DATALOADER_SETTINGS)
    loader_valid = DataLoader(data_valid, **defines.DATALOADER_SETTINGS)

    autoenc.fit(loader_train, loader_valid)

    # Save the trained model if desired
    if config[defines.CONFIG_KEY_MAIN]['save_model']:
        torch.save(autoenc.state_dict(), config[defines.CONFIG_KEY_MAIN]['saved_modelname'])

    # Obtain model predictions
    y_true, y_preds, losses = autoenc.test(loader_test)

    # Generate model decisions for post-processing and analysis
    if config[defines.CONFIG_KEY_MAIN]['save_stats']:
        results = pd.DataFrame({'y_true': y_true, 'y_preds': y_preds, 'losses': losses})
        results.to_csv(config[defines.CONFIG_KEY_MAIN]['saved_resultsname'], index=False)

        training_stats = autoenc.get_train_stats()
        validation_stats = autoenc.get_valid_stats()
        export_dict = {
            'epoch'      : list(range(1, len(training_stats['loss']) + 1)),
            'train_loss' : training_stats['loss'],
            'valid_loss' : validation_stats['loss'],
            'valid_acc'  : validation_stats['accuracy']
        }

        stats = pd.DataFrame(export_dict)
        stats.to_csv(config[defines.CONFIG_KEY_MAIN]['saved_statsname'], index=False)

    # Print output model statistics
    ml_stats.print_model_statistics(y_true, y_preds, losses)


if __name__ == '__main__':
    main(sys.argv)
