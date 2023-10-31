"""Anomaly detection models and relevant functions."""

import torch
import torch.nn as nn
from tqdm import tqdm

import defines
from early_stopper import EarlyStopper


class Generic_AE(torch.nn.Module):
    """Generic Autoencoder for anomaly detection in the NanoDec project.
    The class is supposed to be abstract, and all inheriting children need to specify its
    model, loss function and optimizer in the constructor as well as the forward() method."""

    def __init__(self, device, config : dict) -> None:
        super(Generic_AE, self).__init__()

        # Has to be specified by inherited classes
        self.loss_fn   = None
        self.optimizer = None

        # Local variables
        es_patience = config[defines.CONFIG_KEY_MODEL]['early_stopping']['patience']
        es_delta    = config[defines.CONFIG_KEY_MODEL]['early_stopping']['min_delta']

        # Instance variables
        self.device = device
        self.epochs = config[defines.CONFIG_KEY_MODEL]['epochs']
        self.early_stop = config[defines.CONFIG_KEY_MODEL]['early_stopping']['use']
        self.early_stopper = EarlyStopper(es_patience, es_delta) if self.early_stop else None
        self.decision_threshold = config[defines.CONFIG_KEY_MODEL]['threshold']
        self.print_info = config[defines.CONFIG_KEY_MODEL]['print_info']
        self.log = {
            'train' : {
                'loss' : []
            },
            'valid' : {
                'loss' : [],
                'accuracy' : []
            },
            'test' : {
                'y_true' : [],
                'y_preds': [],
                'losses' : []
            }
        }

    def fit(self, loader_train, loader_valid=None) -> None:
        for epoch_cnt in range(self.epochs):
            print(f"----------   Epoch {epoch_cnt+1}/{self.epochs}   ----------")
            self._train_loop(loader_train)

            if loader_valid is not None:
                valid_loss = self._eval_loop(loader_valid, 'valid')

                # Use early stopping if desired
                if self.early_stop:
                    if self.early_stopper.early_stop(valid_loss, self.state_dict()):
                        # Early stopping condition reached - restore the model
                        print("Early stopping condition reached! Restoring the best parameters "
                              f"from epoch {epoch_cnt - self.early_stopper.patience + 1}.\n")

                        best_model = self.early_stopper.get_best_model()
                        self.load_state_dict(best_model)

                        break

    def test(self, loader_test) -> tuple:
        return self._eval_loop(loader_test, 'test')


    def get_train_stats(self) -> dict:
        return self.log['train']


    def get_valid_stats(self) -> dict:
        return self.log['valid']

    ###############################################################################
    #############################   PRIVATE INTEFACE   ############################
    ###############################################################################
    def _train_loop(self, dataloader) -> None:
        num_batches = len(dataloader)
        running_loss = 0

        self.train()

        for (X, _) in tqdm(dataloader, unit='batch', desc='Train run  ', leave=False):
            X = X.to(self.device)

            # Compute prediction error
            pred = self(X)
            loss = self.loss_fn(pred, X).mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()

        # Compute average loss per epoch, log and print it
        avg_batch_loss = running_loss / num_batches

        self.log['train']['loss'].append(avg_batch_loss)

        if self.print_info:
            print(f"Train      - Accuracy: --     | Avg batch loss: {avg_batch_loss:>11.2f}")


    def _eval_loop(self, dataloader, mode):
        # Data size would normally be calculated as len(dataloader.dataset), but due to our specific
        # setup, it needs to be computed on-the-go due to variable batch size
        num_batches  = len(dataloader)
        data_size    = 0
        running_loss = 0
        correct      = 0

        self.eval()

        with torch.no_grad():
            for (X, y) in tqdm(dataloader, unit='batch', desc='Eval run   ', leave=False):
                data_size += len(y)

                # Obtain model reconstructions and compute their losses
                X, y        = X.to(self.device), y.to(self.device)
                X_hat       = self(X)
                per_x_loss  = self.loss_fn(X_hat, X).mean(dim=1)

                # Obtain the number of anomalies above threshold and determine correct predictions
                y_hat = (per_x_loss > self.decision_threshold).long()
                correct += (y_hat == y).sum().item()

                # Increment evaluation loss
                running_loss += per_x_loss.mean().item()

                # Log per-element loss for ROC computation
                if mode == 'test':
                    self.log['test']['y_true'].append(y)
                    self.log['test']['y_preds'].append(y_hat)
                    self.log['test']['losses'].append(per_x_loss)

        avg_batch_loss = running_loss / num_batches
        accuracy = correct / data_size

        # Perform logging and printing if desired
        if mode == 'valid':
            self.log['valid']['loss'].append(avg_batch_loss)
            self.log['valid']['accuracy'].append(accuracy)

            if self.print_info:
                print(f"Validation - Accuracy: {(100*accuracy):>5.2f}% | "
                      f"Avg batch loss: {avg_batch_loss:>11.2f}\n")

            return avg_batch_loss
        elif mode == 'test':
            return (
                torch.cat(self.log['test']['y_true'], dim=0).cpu().numpy(),
                torch.cat(self.log['test']['y_preds'], dim=0).cpu().numpy(),
                torch.cat(self.log['test']['losses'], dim=0).cpu().numpy()
            )

'''
class Autoencoder(Generic_AE):
    """Simple autoencoder model for anomaly detection on VNET project."""

    def __init__(self, device, config: dict) -> None:
        super().__init__(device, config)

        self.encoder = nn.Sequential(
            nn.Linear(defines.DATA_X_DIMENSIONALITY, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 96),
            nn.ReLU(),
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, defines.DATA_X_DIMENSIONALITY),
            nn.ReLU()
        )

        self.loss_fn   = torch.nn.MSELoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config[defines.CONFIG_KEY_MODEL]['lr'])

        # Send the model to desired device
        self.to(self.device)

    def forward(self, x):
        return self.decoder(self.encoder(x))
'''

class Autoencoder(Generic_AE):
    """Simple autoencoder model for anomaly detection on VNET project."""

    def __init__(self, device, config: dict) -> None:
        super().__init__(device, config)

        self.encoder = nn.Sequential(
            nn.Linear(defines.DATA_X_DIMENSIONALITY, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.Dropout(0.1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 96),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, defines.DATA_X_DIMENSIONALITY),
            nn.ReLU()
        )

        self.loss_fn   = torch.nn.MSELoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config[defines.CONFIG_KEY_MODEL]['lr'])

        # Send the model to desired device
        self.to(self.device)

    def forward(self, x):
        return self.decoder(self.encoder(x))
