"""Data splitting script for training and evaluation purposes."""

import argparse
import joblib
import os
import sys

import numpy as np

# Append absolute path to of the 'src' directory to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)).rsplit('src', 1)[0])

import src.utils as utils


DEFAULT_SPLIT_RATIOS_TRAIN_VALIDATION_TEST = (0.85, 0.05, 0.1)
DEFAULT_SPLIT_RATIOS_TRAIN_VALIDATION = (0.9, 0.1)
DEFAULT_CSV_SEPARATOR = ','
DEFAULT_RANDOM_STATE = None
DEFAULT_N_JOBS = -1

FILE_SUFFIX_TRAIN = '_train'
FILE_SUFFIX_VALIDATION = '_valid'
FILE_SUFFIX_TEST = '_test'


def main(raw_args: list) -> None:
    args = parse_args(raw_args)

    filepaths = []
    for root, dirnames, filenames in os.walk(args.dirpath):
        for filename in filenames:
            filepaths.append(os.path.join(root, filename))

    joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(split_file)(
            filepath,
            args,
        )
        for filepath in filepaths
    )


def split_file(filepath, args):
    data = utils.load_df(filepath, csv_sep=DEFAULT_CSV_SEPARATOR)

    _process_split_ratios(args)

    split_size_train = args.split_ratios[0]
    split_size_validation = args.split_ratios[1]
    if args.test_dirpath is not None:
        split_size_test = args.split_ratios[2]
    else:
        split_size_test = None

    if args.random_state == -1:
        args.random_state = None

    # Perform splitting into train, test, and validation sets
    split_aux_array = np.random.default_rng(seed=args.random_state).uniform(size=len(data))
    if split_size_test is not None:
        split_mask_train = split_aux_array <= split_size_train
        split_mask_validation = ((split_aux_array > split_size_train)
                                 & (split_aux_array <= (split_size_train + split_size_validation)))
        split_mask_test = split_aux_array > (split_size_train + split_size_validation)
    else:
        split_mask_train = split_aux_array <= split_size_train
        split_mask_validation = ((split_aux_array > split_size_train)
                                 & (split_aux_array <= (split_size_train + split_size_validation)))
        split_mask_test = None

    filepath_root, file_ext = os.path.splitext(filepath)
    filename_root = os.path.basename(filepath_root)
    file_ext = file_ext[1:]

    data_train = data.iloc[split_mask_train]
    os.makedirs(args.train_dirpath, exist_ok=True)
    train_filepath_root = os.path.join(args.train_dirpath, filename_root) + FILE_SUFFIX_TRAIN
    utils.save_df(data_train, train_filepath_root, file_ext, csv_sep=DEFAULT_CSV_SEPARATOR)

    data_valid = data.iloc[split_mask_validation]
    os.makedirs(args.validation_dirpath, exist_ok=True)
    validation_filepath_root = os.path.join(
        args.validation_dirpath, filename_root) + FILE_SUFFIX_VALIDATION
    utils.save_df(data_valid, validation_filepath_root, file_ext, csv_sep=DEFAULT_CSV_SEPARATOR)

    if split_size_test is not None:
        data_test = data.iloc[split_mask_test]
        os.makedirs(args.test_dirpath, exist_ok=True)
        test_filepath_root = os.path.join(args.test_dirpath, filename_root) + FILE_SUFFIX_TEST
        utils.save_df(data_test, test_filepath_root, file_ext, csv_sep=DEFAULT_CSV_SEPARATOR)


def _process_split_ratios(args):
    if args.split_ratios is None:
        if args.test_dirpath is not None:
            args.split_ratios = DEFAULT_SPLIT_RATIOS_TRAIN_VALIDATION_TEST
        else:
            args.split_ratios = DEFAULT_SPLIT_RATIOS_TRAIN_VALIDATION
    else:
        if args.test_dirpath is not None and len(args.split_ratios) != 3:
            raise ValueError(
                '--split-ratio must contain 3 arguments if --test-filepath is specified')

        if args.test_dirpath is None and len(args.split_ratios) != 2:
            raise ValueError(
                '--split-ratio must contain 2 arguments if --test-filepath is not specified')


def parse_args(raw_args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Splits file in the given directory'
                    ' into train, validation and test files.')

    parser.add_argument(
        'dirpath',
        help='Input directory path containing features to split.',
    )
    parser.add_argument(
        'train_dirpath',
        help='Output directory path containing train files.',
    )
    parser.add_argument(
        'validation_dirpath',
        help='Output directory path containing validation files.',
    )
    parser.add_argument(
        '--test-dirpath',
        help='Output directory path containing test files.',
    )
    parser.add_argument(
        '--split-ratios',
        type=float,
        nargs='+',
        default=None,
        help=('Split ratios to use for train, validation and test sets.'
              f' If --test-dirpath is specified, the default is'
              f' {DEFAULT_SPLIT_RATIOS_TRAIN_VALIDATION_TEST}.'
              f' Otherwise, the default is {DEFAULT_SPLIT_RATIOS_TRAIN_VALIDATION}.'),
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=DEFAULT_N_JOBS,
        help=('Number of jobs to run in parallel.'
              f' Default: {DEFAULT_N_JOBS}'),
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=('Fixed random state to use for splitting the data. Pass -1 to not use a fixed random state.'
              f' Default: {DEFAULT_RANDOM_STATE}'),
    )
    parser.add_argument(
        '--csv-sep',
        default=DEFAULT_CSV_SEPARATOR,
        help=('Separator to use for input and output files if the file format is CSV.'
              f' Default: {DEFAULT_CSV_SEPARATOR}'),
    )

    return parser.parse_args(raw_args)


if __name__ == '__main__':
    main(sys.argv[1:])
