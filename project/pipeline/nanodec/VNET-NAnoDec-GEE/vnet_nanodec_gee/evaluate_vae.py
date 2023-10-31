from pathlib import Path
import logging
import pickle
from typing import Optional, List

import click
import psutil

import numpy as np
import pandas as pd
from petastorm import make_reader
from petastorm.pytorch import DataLoader
import torch
import torch.nn.functional as F

from .ml.vae import VAE
from .utils import get_uri
from . import config


def calc_recon_loss(recon_x, num_features, x, logvar=None, mu=None, loss_type: str = 'mse') -> list:
    """
    Return the reconstruction loss

    :param recon_x: reconstructed x, output from model
    :param num_features: number of input features
    :param x: original x
    :param logvar: variance, output from model, ignored when loss_type isn't 'bce+kd'
    :param mu: mean, output from model, ignored when loss_type isn't 'bce+kd'
    :param loss_type: method to compute loss, option: 'bce', 'mse', 'bce+kd'
    :return: list of reconstruct errors
    :rtype: list
    """
    loss_type = loss_type.lower()

    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none').view(-1, num_features).mean(dim=1)
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction='none').view(-1, num_features).mean(dim=1)
    elif loss_type == 'bce+kd':
        bce = F.binary_cross_entropy(recon_x, x, reduction='none').view(-1, num_features).mean(dim=1)
        kd = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_error = bce + kd
    else:
        raise Exception('Invalid loss type: only supporting the following: "mse", "bce", or "bce+kd"')

    return recon_error.tolist()


@click.command()
@click.option('-d', '--data_path', help='dir path containing model input parquet files for testing', required=True)
@click.option('-m', '--model_path', help='path to trained model', required=True)
@click.option('-f', '--filter_model_path', help='path to trained filter model', required=False)
@click.option('--dataset', help='format of the input dataset', type=click.Choice(['gee', 'vnet']), default='vnet')
@click.option('-o', '--output_path', help='dir path containing results', required=True)
@click.option('--gpu', help='GPU IDs to use for training', multiple=True, type=int)
@click.option('--max_workers', help='maximum number of workers to use', default=30, type=int)
@click.option('--add_labels_to_results',
              help=('whether to add labels to the output results;'
                    ' requires labels to be present in files in data_path'),
              is_flag=True)
@click.option('--add_ip_to_results',
              help='whether to add source and destination IPs to the output results',
              is_flag=True)
def main(
        data_path: str,
        model_path: str,
        filter_model_path: Optional[str],
        dataset: str,
        output_path: str,
        gpu: Optional[List[int]],
        max_workers: int,
        add_labels_to_results: bool,
        add_ip_to_results: bool,
):
    if not gpu:
        gpu = None
    else:
        gpu = list(gpu)

    # initialise logger
    logger = logging.getLogger(__file__)
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel('INFO')

    logger.info('Load model...')

    num_cores = min(psutil.cpu_count(logical=True), max_workers)

    if dataset == 'vnet':
        num_features = len([column for column in config.VNET_COLUMNS if column.column_type.startswith('feature_')])
    elif dataset == 'gee':
        num_features = config.NUM_FEATURES_GEE
    else:
        raise ValueError('invalid value for "dataset"')

    if gpu:
        gpu_str = ','.join([str(gpu_id) for gpu_id in gpu])
        model = VAE.load_from_checkpoint(
            checkpoint_path=model_path, map_location=torch.device(f'cuda:{gpu_str}'), n_features=num_features)
    else:
        model = VAE.load_from_checkpoint(
            checkpoint_path=model_path, map_location=torch.device('cpu'), n_features=num_features)

    model.eval()

    filter_model = None
    if filter_model_path:
        with open(filter_model_path, 'rb') as f:
            filter_model = pickle.load(f)

    logger.info('Initialise data loader...')
    # load data loader
    reader = make_reader(
        get_uri(data_path), reader_pool_type='process', workers_count=num_cores, num_epochs=1
    )
    dataloader = DataLoader(reader, batch_size=300, shuffling_queue_capacity=4096)

    logger.info('Calculating reconstruction error...')
    mse_loss_list = []
    bce_loss_list = []
    bce_kd_loss_list = []
    filter_preds_list = []
    filter_probs_list = []
    label_list = []
    src_ip_list = []
    dst_ip_list = []

    for data in dataloader:
        x = data['feature']
        recon_x, mu, logvar = model(x)

        mse_loss = calc_recon_loss(recon_x, num_features, x, loss_type='mse')
        bce_loss = calc_recon_loss(recon_x, num_features, x, loss_type='bce')
        bce_kd_loss = calc_recon_loss(recon_x, num_features, x, logvar, mu, loss_type='bce+kd')

        mse_loss_list.extend(mse_loss)
        bce_loss_list.extend(bce_loss)
        bce_kd_loss_list.extend(bce_kd_loss)

        if filter_model is not None:
            data_x = np.squeeze(x.numpy())
            pred = filter_model.predict(data_x)
            prob = filter_model.predict_proba(data_x).max(axis=1)
            filter_preds_list.extend(pred)
            filter_probs_list.extend(prob)
        
        if add_labels_to_results:
            label = data['label']
            label_list.extend(label)
        
        if add_ip_to_results:
            src_ip = data['src_ip']
            src_ip_list.extend(src_ip)
            dst_ip = data['dst_ip']
            dst_ip_list.extend(dst_ip)

    df_dict = {
        'mse_loss': mse_loss_list,
        'bce_loss': bce_loss_list,
        'bce+kd_loss': bce_kd_loss_list,
    }

    if filter_model is not None:
        df_dict['filter_pred'] = filter_preds_list
        df_dict['filter_prob'] = filter_probs_list

    if add_labels_to_results:
        df_dict['label'] = label_list

    if add_ip_to_results:
        df_dict['src_ip'] = src_ip_list
        df_dict['dst_ip'] = dst_ip_list
    
    # pandas dataframe for easier evaluation
    df = pd.DataFrame(df_dict)

    if add_labels_to_results:
        logger.info(df['label'].value_counts())
    logger.info(df['mse_loss'].describe())

    logger.info('Saving results...')
    results_path = Path(output_path)
    results_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(results_path / 'results_losses.parquet')

    logger.info('Done')


if __name__ == '__main__':
    main()
