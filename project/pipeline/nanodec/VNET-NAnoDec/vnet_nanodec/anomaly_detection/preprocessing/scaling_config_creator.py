"""
Creates a scaling config in the YAML format to be used for data_preparer.py script.

Usage:
    python scaling_config_creator.py data.csv scaling_config.yaml
        - Reads file data.csv, computes statistics, sets scaling type to "standard" and saves
          to scaling_config.yaml file.

    python scaling_config_creator.py data.csv scaling_config.yaml "minmax" features_types.yaml
        - Reads file data.csv, computes statistics, and uses scaling type according to the
          feature_types.yaml file. If the feature is not listed within the file and is still
          eligible to be scaled, a default scaling type of "minmax" is used. The resulting
          scaling configuration is saved to scaling_config.yaml file.

    features_scaling_types_filepath -- YAML configuration in the format:
        scaling_type:
            <feature_name1> : scaling_type
            <feature_name2> : scaling_type
            ...
            <feature_nameN> : scaling_type
Whereas scaling types include {absmax, minmax, norm, standard, bin}.
"""

import argparse
import os
import sys

import yaml

import pandas as pd

from collections import defaultdict, OrderedDict

from . import vnet_preprocessing as vp
from vnet_nanodec.defines import FEATURES_CASTDICT
import vnet_nanodec.utils as utils


# Configure YAML module for dumping ordered dictionaries
yaml.add_representer(OrderedDict, lambda dumper,
    data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))

# Default top-level YAML configuration key
SCALING_TYPE_CONFIG_KEY_DEFAULT = 'scaling_type'
SCALING_STATS_CONFIG_DEFAULT = 'scaling_config'

# Default parameters
DEFAULT_SCALING_TYPE = 'minmax'
DEFAULT_SCALING_TYPES_FILEPATH = None
DEFAULT_INPUT_FORMAT = 'csv'

# Percentiles to compute when describing the DataFrame (default None -> [0.25, 0.5, 0.75])
#DESC_PERCENTILES = np.linspace(0, 1, 41)
DESC_PERCENTILES = [0.25, 0.5, 0.75, 0.85, 0.875, 0.9, 0.925, 0.95, 1.0]


def main(args: list) -> None:
    """Loads a file containing data to compute scaling statistics from. Computes min, max, mean,
    standard deviation, and percentiles for each specified feature name based on the list of
    features.

    Parameters:
        args -- command line arguments to be parsed"""

    # Parse arguments and turn them into variables for better readability
    parsed_args = parse_args(args)

    default_scaling_type = parsed_args.default_scaling_type         # Default scaling type to use
    out_config_filepath  = parsed_args.out_scaling_config           # Output config filepath
    data_path            = parsed_args.data_path                    # Processed data filepath
    features_scaling_types_filepath = parsed_args.scaling_types     # Input scaling types config
    process_base_only    = parsed_args.base_only
    process_window_only  = parsed_args.window_only

    data_descs = pd.DataFrame(None)                            # Computed data descriptions
    scaling_dict = defaultdict(lambda : default_scaling_type)  # Scaling type features dict
    out_dict = {SCALING_STATS_CONFIG_DEFAULT: OrderedDict()}  # Output dictionary dumped into file

    # Load the data - either a single file or all files in the folder
    if os.path.isfile(data_path):
        data = utils.load_df(data_path, dtypes=FEATURES_CASTDICT, csv_sep=',', use_dask=True)
    elif os.path.isdir(data_path):
        file_ext = _get_file_ext_from_directory(data_path)
        data = utils.load_df(
            os.path.join(data_path, '*' + file_ext), dtypes=FEATURES_CASTDICT, csv_sep=',', use_dask=True)
    else:
        print("Invalid input data file path", file=sys.stderr)
        sys.exit(1)

    # Open the file specifying feature scaling types if provided
    try:
        with open(features_scaling_types_filepath, 'r') as scaling_cfg_file:
            scaling_dict.update(yaml.safe_load(scaling_cfg_file)[SCALING_TYPE_CONFIG_KEY_DEFAULT])
    except TypeError:
        # Configuration for feature scaling is None, keep the dict empty
        pass
    except FileNotFoundError:
        print("The specified feature scaling configuration file {} does not exist.".format(
            features_scaling_types_filepath), file=sys.stderr)
        sys.exit(1)
    except KeyError:
        # The configuration file does not contain a required structure,
        print("The configuration file {} does not have a required structure.".format(
            features_scaling_types_filepath), file=sys.stderr)
        sys.exit(1)

    # Obtain descriptions for numerical features from all data if not window exclusive
    if not process_window_only:
        data_descs = data.loc[:, vp.FEATURES_NUMERICAL + vp.FEATURES_NUMERICAL_COMPUTED].describe(
            percentiles=DESC_PERCENTILES).compute()

    # Obtain descriptions for window-based features from non-zero values if not base exclusive
    if not process_base_only:
        data_descs = pd.concat(
            [
                data_descs,
                data[data[vp.FEATURES_NUMERICAL_WINDOWER] != 0][vp.FEATURES_NUMERICAL_WINDOWER].describe(
                    percentiles=DESC_PERCENTILES).compute(),
            ],
            axis=1)

    # Create a scaling_type row within the computed data descriptions dataframe
    scaling_types_ser = pd.Series(None, index=data_descs.columns, dtype=str,
        name=SCALING_TYPE_CONFIG_KEY_DEFAULT)

    for colname in data_descs.columns:
        scaling_types_ser[colname] = scaling_dict[colname]

    data_descs.loc[SCALING_TYPE_CONFIG_KEY_DEFAULT] = scaling_types_ser

    # Drop "count" index produced by describe() function but unimportant for our purposes
    data_descs = data_descs.drop(index='count')

    # Create a dictionary of computed statistics
    for feature_name in data_descs.columns:
        out_dict[SCALING_STATS_CONFIG_DEFAULT][feature_name] = data_descs[feature_name].to_dict()

    os.makedirs(os.path.dirname(out_config_filepath), exist_ok=True)

    # Dump the dictionary into the file
    with open(out_config_filepath, 'w') as out_config_file:
        yaml.dump(out_dict, out_config_file)


def _get_file_ext_from_directory(dirpath):
    filepaths = [os.path.join(root, filename) for root, _, filenames in os.walk(dirpath) for filename in filenames]
    if len(filepaths) > 0:
        return os.path.splitext(filepaths[0])[1]
    else:
        return DEFAULT_INPUT_FORMAT


def parse_args(args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Creates a scaling configuration based on the provided data.")

    parser.add_argument(
        'data_path',
        help=('Path to the data to be processed.'
              ' If a folder is passed, all files within are processed.'),
    )
    parser.add_argument(
        'out_scaling_config',
        help='Path to the output scaling config',
    )
    parser.add_argument(
        '--default-scaling-type',
        default=DEFAULT_SCALING_TYPE,
        choices=['absmax', 'minmax', 'norm', 'standard', 'bin'],
        help=('Scaling type to use when --scaling-types config is not given.'
              f' Default: {DEFAULT_SCALING_TYPE}'),
    )
    parser.add_argument(
        '--scaling-types',
        default=DEFAULT_SCALING_TYPES_FILEPATH,
        help=('Filepath to the configuration file containing feature scaling types.'
              f' Default: {DEFAULT_SCALING_TYPES_FILEPATH}'),
    )

    # Arguments mutually exclusive group for base/window features only
    subwork_group = parser.add_mutually_exclusive_group(required=False)
    subwork_group.add_argument(
        '--base-only',
        default=False,
        action='store_true',
        help='Process only base statistics (without window',
    )
    subwork_group.add_argument(
        '--window-only',
        default=False,
        action='store_true',
        help='Process only window-based statistics',
    )

    return parser.parse_args(args)


if __name__ == '__main__':
    main(sys.argv[1:])
