"""Custom DataLoader class for flow-based data."""

import numpy as np
import pandas as pd
import random
import torch
import os
from torch.utils.data import Dataset

import defines


class FlowsDataset(Dataset):
    """Loads flows from files based on the input directory"""

    def __init__(self, dataset, config: dict, background_only=False):
        """Initializes the Flows loader dataloader instance.

        Parameters:
            dataset         -- Folder to the dataset containing flows in CSV files
            batch_size      -- Number of flows to return per batch
            shuffle         -- Shuffle files within the dataset
            background_only -- True if only background data should be considered
        """

        self.data_dirpath    = dataset
        self.data_files      = os.listdir(self.data_dirpath)
        self.background_only = background_only
        self.min_batch_size  = config['min_batch_size']


    def __len__(self) -> int:
        return len(self.data_files)


    def __getitem__(self, idx):
        data_all = pd.DataFrame()

        # Read the data until at least minimum batch size is reached
        while True:
            # Load the data and filter only background traffic if desired
            data = pd.read_csv(os.path.join(self.data_dirpath, self.data_files[idx]))

            if self.background_only:
                data = data[data[defines.DATASET_LABEL_COLNAME] == defines.DATASET_LABEL_BACKGROUND]

            # Append the data to the final
            data_all = pd.concat([data_all, data], ignore_index=True)

            # Break the loop if the batch size is big enough
            if len(data_all) >= self.min_batch_size:
                break
            else:
                # If the collected batch is not big enough, generate new index and go again
                idx = random.randrange(len(self.data_files))

        # Shuffle data within the DataFrame
        data_all = data_all.sample(frac=1)

        # Prepare X and Y and return them
        x = data_all.drop(columns=defines.DATASET_COLS_DROP).values.astype(np.float32)
        y = (data_all[defines.DATASET_LABEL_COLNAME] !=
                defines.DATASET_LABEL_BACKGROUND).astype(np.uint8).values

        return torch.tensor(x), torch.tensor(y)
