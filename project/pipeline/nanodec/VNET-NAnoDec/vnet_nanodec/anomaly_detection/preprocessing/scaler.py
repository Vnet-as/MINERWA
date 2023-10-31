"""
Feature scaling functionality - either standalone or as an interface.

Example usage:
    scaler.py data.csv data_scaled.csv scaling_config.yaml 90
"""

import argparse
import os
import sys

import yaml

import joblib
import numpy as np
import pandas as pd

# Append to system path for imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)).rsplit('src', 1)[0])

import vnet_nanodec.utils as utils


# Program settings
CONFIG_KEY_SCALING_ROOT = 'scaling_config'
FILE_SUFFIX_SCALED = '_scaled'

DEFAULT_N_JOBS = -1
DEFAULT_SCALING_MAX = None
DEFAULT_CSV_SEPARATOR = ','
DEFAULT_OUTPUT_FORMAT = 'csv'


def main(raw_args: list) -> None:
    args = parse_args(raw_args)
    unscaling_config = None

    # Load the scaling config
    with open(args.scaling_config_filepath, 'r') as scaling_config_file:
        scaling_config = yaml.safe_load(scaling_config_file)

    # Determine if unscaling has to be provided
    if args.unscaling_config is not None:
        with open(args.unscaling_config, 'r') as unscaling_config_file:
            unscaling_config = yaml.safe_load(unscaling_config_file)

    if os.path.isfile(args.input_path):
        _scale_features_per_file(
            args.input_path, args.output_path, args.output_format,
            args.csv_sep, scaling_config, args.scaling_max,
            unscaling_config, args.unscale_max, args.unscale_only)
    elif os.path.isdir(args.input_path):
        os.makedirs(args.output_path, exist_ok=True)

        filepaths = []
        for root, dirnames, filenames in os.walk(args.input_path):
            for filename in filenames:
                filepaths.append(
                    (os.path.join(root, filename), os.path.join(args.output_path, filename)))

        joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(_scale_features_per_file)(
                input_filepath, output_filepath, args.output_format,
                args.csv_sep, scaling_config, args.scaling_max,
                unscaling_config, args.unscale_max, args.unscale_only,
            )
            for input_filepath, output_filepath in filepaths
        )
    else:
        raise ValueError(f'Input path "{args.input_path}" must be a valid file or directory')


def _scale_features_per_file(
        input_filepath: str, output_filepath: str, output_format: str, csv_sep: str, config: dict,
        new_scaling_max: str, unscaling_config: dict, unscaling_max: str, unscale_only: bool):
    data = utils.load_df(input_filepath, csv_sep=csv_sep)

    # Perform unscaling if desired
    if unscaling_config is not None:
        data = scale_features(data, unscaling_config[CONFIG_KEY_SCALING_ROOT], unscaling_max,
                              True)

    # Perform scaling
    if not unscale_only:
        data = scale_features(data, config[CONFIG_KEY_SCALING_ROOT], new_scaling_max)

    # Save the scaled date
    output_file_root, _ = os.path.splitext(output_filepath)
    output_file_root += FILE_SUFFIX_SCALED

    utils.save_df(data, output_file_root, output_format, csv_sep=csv_sep)


def scale_features(data: pd.DataFrame, config: dict, new_scaling_max: str = None,
                   reverse_scaling: bool = False) -> pd.DataFrame:
    """Scales features in data according to the supplied configuration dictionary.
    If the value is not present in the dictionary, it is left intact.
    The structure of dictionary is as follows
        'feature_name' : {feature-specific dictionary for scaling}, see "_scale_feature"

    Parameters:
        data            -- DataFrame to scale
        config          -- configuration dictionary to use for scaling
        new_scaling_max -- adjust the maximum value to cut off outlier values
        reverse_scaling -- True if unscaling is desired
    """

    data_new = data
    scaled_cols = list(config.keys())

    # Determine whether to perform scaling or unscaling
    scaling_func = _unscale_feature if reverse_scaling else _scale_feature

    # Adjust the maximum scaling value if desired
    new_scaling_max = new_scaling_max + '%' if new_scaling_max is not None else 'max'

    for colname, colconfig in config.items():
        colconfig['max'] = colconfig[new_scaling_max]
        data_new[colname] = scaling_func(data[colname], colconfig)

    # Perform post-processing to avoid NaN values in the final DataFrame
    data_new[scaled_cols] = data_new[scaled_cols].fillna(value=0)

    return data_new


def _scale_feature(feature_col: pd.Series, config: dict) -> pd.Series:
    """Scales a column of features according to the provided configuration
    dictionary. The configuration dictionary is required to have a structure as follows:
        scaling   : {absmax, minmax, norm, standard, robust, bin}
        mean      : value,
        std       : value,
        min       : value,
        25%       : value,
        50%       : value,
        75%       : value,
        max       : value,
        threshold : value,

    All variables are not mandatory - e.g., when binning is not considered, threshold
    value does not need to be present.

    Parameters:
        feature_col -- column of features to scale
        config      -- configuration dictionary defining scaling parameters

    Returns:
        pd.Series -- Series processed by a specified scaling method"""

    scaling_type = config['scaling_type']

    if scaling_type == 'absmax':
        new_feature_col = feature_col / _get_safe_denominator(config['max'])
    elif scaling_type == 'minmax':
        new_feature_col = (feature_col - config['min']) / _get_safe_denominator(config['max'] - config['min'])
    elif scaling_type == 'norm':
        new_feature_col = (feature_col - config['mean']) / _get_safe_denominator(config['max'] - config['min'])
    elif scaling_type == 'standard':
        new_feature_col = (feature_col - config['mean']) / _get_safe_denominator(config['std'])
    elif scaling_type == 'robust':
        new_feature_col = (feature_col - config['50%']) / _get_safe_denominator(config['75%'] - config['25%'])
    elif scaling_type == 'bin':
        new_feature_col = (feature_col > config['threshold']).astype(np.uint8)
    else:
        new_feature_col = feature_col

    return new_feature_col


def _unscale_feature(feature_col: pd.Series, config: dict) -> pd.Series:
    """Similar syntax and usage than _scale_feature, but perform reverse scaling (unscaling)
    instead. Currently, only MinMax unscaling is supported."""

    scaling_type = config['scaling_type']

    if scaling_type == 'minmax':
        new_feature_col = feature_col * (config['max'] - config['min']) + config['min']
    else:
        new_feature_col = feature_col

    return new_feature_col


def _get_safe_denominator(denominator):
    """Returns 1 if the denominator is 0, and the original denominator otherwise. This avoids
    division by zero errors."""
    return denominator if denominator else 1


def parse_args(raw_args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Scales data in the specified directory using the provided scaler.')

    parser.add_argument(
        'input_path',
        help='Input path containing features to split. Can be either a single file or a directory.',
    )
    parser.add_argument(
        'output_path',
        help=('Output path. If input_path is a file, output_path must be a file.'
              ' If input_path is a directory, output_path must be a directory.'),
    )
    parser.add_argument(
        'scaling_config_filepath',
        help='File path to a scaling config.',
    )
    parser.add_argument(
        '--scaling-max',
        default=DEFAULT_SCALING_MAX,
        help=('Alternate maximum value to use when scaling. Useful for clipping outlier values.'
              f' Default: {DEFAULT_SCALING_MAX}'),
    )
    parser.add_argument(
        '--unscaling-config',
        default=None,
        help=('Path to the unscaling configuration file, if unscaling is desired. If --unscale-max'
              ' is not used, uses max as an unscaling factor for MinMax variant'),
    )
    parser.add_argument(
        '--unscale-max',
        default=None,
        help='Sets maximum value used during MinMax unscaling'
    )
    parser.add_argument(
        '--unscale-only',
        action='store_true',
        help='After unscaling, do not perform rescaling.'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=DEFAULT_N_JOBS,
        help=('Number of jobs to run in parallel if input_path is a directory.'
              f' Default: {DEFAULT_N_JOBS}'),
    )
    parser.add_argument(
        '--output-format',
        default=DEFAULT_OUTPUT_FORMAT,
        choices=utils.VALID_OUTPUT_FILE_FORMATS,
        help=('Output file format.'
              f' Default: {DEFAULT_OUTPUT_FORMAT}'),
    )
    parser.add_argument(
        '--csv-sep',
        default=DEFAULT_CSV_SEPARATOR,
        help=('Separator to use for input and output files if the file format is CSV or gzip.'
              f' Default: {DEFAULT_CSV_SEPARATOR}'),
    )

    return parser.parse_args(raw_args)


if __name__ == '__main__':
    main(sys.argv[1:])
