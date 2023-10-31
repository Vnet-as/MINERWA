"""Trainer instance for the model"""

import numpy as np
import torch
from tqdm import tqdm

import defines


def model_train_validate(model, loader_train, loss_fn_train, loader_valid, loss_fn_valid,
        optimizer, device, config) -> None:
    epochs = config[defines.CONFIG_KEY_TRAIN]['epochs']
    decision_treshold = config[defines.CONFIG_KEY_EVAL]['threshold']

    for epoch_cnt in range(epochs):
        print(f"----------   Epoch {epoch_cnt+1}/{epochs}   ----------")
        _train_loop(model, loader_train, loss_fn_train, optimizer, device)
        _valid_loop(model, loader_valid, loss_fn_valid, device, decision_treshold, 'valid')


def model_test(model, loader_test, loss_fn, device, config) -> np.ndarray:
    decision_treshold = config[defines.CONFIG_KEY_EVAL]['threshold']

    return _valid_loop(model, loader_test, loss_fn, device, decision_treshold, 'test')


###############################################################################
#############################   PRIVATE INTEFACE   ############################
###############################################################################
def _train_loop(model, dataloader, loss_fn, optimizer, device: str) -> None:
    num_batches = len(dataloader)
    running_loss = 0

    model.train()

    for (X, _) in tqdm(dataloader, unit='batch', desc='Train run  ', leave=False):
        X = X.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, X)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss

    # Return average batch loss per epoch
    avg_batch_loss = running_loss / num_batches

    print(f"Train      - Accuracy: --     | Avg batch loss: {avg_batch_loss:>.2f}")



def _valid_loop(model, dataloader, loss_fn, device, threshold, mode):
    # Data size would normally be calculated as len(dataloader.dataset), but due to our specific
    # setup, it needs to be computed on-the-go due to variable batch size
    num_batches = len(dataloader)
    data_size   = 0
    valid_loss  = 0
    correct     = 0

    model.eval()

    with torch.no_grad():
        for (X, y) in tqdm(dataloader, unit='batch', desc='Eval run   ', leave=False):
            data_size += len(y)

            # Obtain model reconstructions and compute their losses
            X, y        = X.to(device), y.to(device)
            X_hat       = model(X)
            per_x_loss  = loss_fn(X_hat, X).mean(dim=1)
            valid_loss += per_x_loss.mean().item()

            # Obtain the number of anomalies above threshold and determine correct predictions
            y_hat = per_x_loss > threshold
            correct += (y_hat == y).sum().item()

        valid_loss /= num_batches
        correct /= data_size

    if mode == 'valid':
        print(f"Validation - Accuracy: {(100*correct):>0.2f}% | Avg batch loss: {valid_loss:>.2f}\n")
    elif mode == 'test':
        pass
