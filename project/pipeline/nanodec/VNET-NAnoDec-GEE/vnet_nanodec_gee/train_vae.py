import logging
import os
from typing import List, Optional

import click
import psutil
from petastorm import make_reader
from petastorm.pytorch import DataLoader
from pytorch_lightning import Trainer, seed_everything

from ml.vae import VAE
from utils import get_uri
import config


@click.command()
@click.option('-d', '--data_path', help='dir path containing model input parquet files', required=True)
@click.option('-v', '--validation_path', help='dir path containing model input parquet files for validation', required=False)
@click.option('-m', '--model_path', help='output model directory path', required=True)
@click.option('-c', '--model_checkpoint_path', help='file path to a model checkpoint to resume training from')
@click.option('--final-model-filename', help='filename of the trained model', default='final_model')
@click.option('--dataset', help='format of the input dataset', type=click.Choice(['gee', 'vnet']), default='vnet')
@click.option('--gpu', help='GPU IDs to use for training', multiple=True, type=int)
@click.option('--max_epochs', help='maximum number of epochs', default=50, type=int)
@click.option('--max_workers', help='maximum number of workers to use', default=30, type=int)
@click.option('--log_loss', help='if specified, log train and validation loss', is_flag=True)
@click.option('--random_state', help='fixed random state; pass -1 to use a non-fixed random state', required=False, type=int)
@click.option('--deterministic', help='if specified, force PyTorch to use deterministic algorithms', is_flag=True)
def main(
        data_path: str,
        validation_path: Optional[str],
        model_path: str,
        model_checkpoint_path: Optional[str],
        dataset: str,
        final_model_filename: str,
        gpu: Optional[List[int]],
        max_epochs: int,
        max_workers: int,
        log_loss: bool,
        random_state: Optional[int],
        deterministic: bool,
):
    if gpu:
        accelerator = 'gpu'
        devices = gpu
    else:
        accelerator = 'cpu'
        devices = None

    if random_state and random_state != -1:
        seed_everything(random_state, workers=True)

    # initialise logger
    logger = logging.getLogger(__file__)
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel('INFO')

    logger.info('Initialise data loader...')

    num_cores = min(psutil.cpu_count(logical=True), max_workers)

    # load data loader
    reader = make_reader(
        get_uri(data_path),
        schema_fields=['feature'], reader_pool_type='process',
        workers_count=num_cores, shuffle_row_groups=True, shuffle_row_drop_partitions=2,
        num_epochs=1
    )
    dataloader = DataLoader(reader, batch_size=300, shuffling_queue_capacity=4096)

    if validation_path is not None:
        reader_val = make_reader(
            get_uri(validation_path),
            schema_fields=['feature'], reader_pool_type='process',
            workers_count=num_cores, shuffle_row_groups=True, shuffle_row_drop_partitions=2,
            num_epochs=1
        )
        dataloader_val = DataLoader(reader_val, batch_size=300, shuffling_queue_capacity=4096)
    else:
        dataloader_val = None

    if dataset == 'vnet':
        num_features = len([column for column in config.VNET_COLUMNS if column.column_type.startswith('feature_')])
    elif dataset == 'gee':
        num_features = config.NUM_FEATURES_GEE
    else:
        raise ValueError('invalid value for "dataset"')

    logger.info('Initialise model...')
    os.makedirs(model_path, exist_ok=True)
    model = VAE(num_features, validation=validation_path is not None, log_loss=log_loss)

    logger.info('Start Training...')
    trainer = Trainer(
        num_sanity_val_steps=0, val_check_interval=1.0, max_epochs=max_epochs,
        accelerator=accelerator, devices=devices, deterministic=deterministic,
        default_root_dir=model_path,
    )
    trainer.fit(model, dataloader, val_dataloaders=dataloader_val, ckpt_path=model_checkpoint_path)

    logger.info('Persisting...')
    trainer.save_checkpoint(os.path.join(model_path, final_model_filename))

    logger.info('Done')


if __name__ == '__main__':
    main()
