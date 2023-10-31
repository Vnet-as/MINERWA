"""Common non-configuration settings for model-related options."""

# Configuration keys
CONFIG_KEY_MAIN    = 'main'
CONFIG_KEY_MODEL   = 'model'
CONFIG_KEY_DATASET = 'dataset'

# Data & Dataset-related configuration
DATASET_LABEL_COLNAME    = 'Label'
DATASET_LABEL_BACKGROUND = 'background'
DATASET_COLS_DROP        = ['Label', 'IP_SRC', 'IP_DST']
DATA_X_DIMENSIONALITY    = 170

# Name of the CUDA device - use cuda:{0,1,..., n} to specify concrete device
DEVICE_CUDA_NAME = 'cuda'

# DataLoader with disabled automatic batching
# batch_size=None and batch_sampler=None
DATALOADER_SETTINGS = {
    'shuffle' : True,
    'batch_size' : None,
    'batch_sampler' : None,
    'num_workers' : 4,
}
