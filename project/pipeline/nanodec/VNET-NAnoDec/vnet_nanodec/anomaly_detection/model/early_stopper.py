"""Early stopper module.
Inspired by: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch"""


import numpy as np
import torch


class EarlyStopper:
    def __init__(self, patience: int, min_delta: int = 0) -> None:
        self.patience     = patience        # Number of epochs to wait without improvement
        self.min_delta    = min_delta       # Minimum validation loss improvement
        self.epochs_cnt   = 0               # Number of epochs counter without improvement
        self.min_val_loss = np.inf          # Minimum achieved validation loss so far
        self.best_model   = None            # Best achieved model weights

    def early_stop(self, val_loss: np.float32, model_params : dict) -> bool:
        """Determines whether to perform early stopping.  If there is an improvement, saves
        the best model configuration and logs new validation loss.

        Parameters:
            val_loss     -- Value of the achieved validation loss to consider
            model_params -- Current model parameters as a state dictionary

        Returns:
            True  -- early stop should be performed
            False -- do not perform early stopping"""

        if val_loss < self.min_val_loss:
            # The model has improved, reset counters and save its parameters
            self.min_val_loss = val_loss
            self.counter = 0
            self.best_model = model_params
        elif val_loss > (self.min_val_loss + self.min_delta):
            # The model has not improved, increase the counter
            self.counter += 1

            if self.counter >= self.patience:
                # The early stop condition has been reached
                return True

        return False

    def get_best_model(self) -> torch.Tensor:
        """Returns the best model logged through early stopping mechanism."""

        return self.best_model
