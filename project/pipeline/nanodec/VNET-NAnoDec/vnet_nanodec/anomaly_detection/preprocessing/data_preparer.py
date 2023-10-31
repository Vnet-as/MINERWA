"""
Preprocessing of a single network flow file.

Examples:
    python main.py flows.csv --windowing-settings windowing.yaml
        - Processes the flows.csv file within a single process, loads config from windowing.yaml
    python main.py flows.csv --windowing-settings windowing.yaml --n-jobs 1
        - Processes the flows.csv file using 1 worker process
    python main.py flows.csv --windowing-settings windowing.yaml --n-jobs 8 --output-windowers windowers.bin
        - Processes the flows.csv file with 8 worker processes, saving their windower contexts to
          the windowers.bin file
    python main.py flows.csv --windowing-settings windowing.yaml --n-jobs 8 --input-windowers windowers.bin --output-windowers windowers.bin
        - Processes the flows.csv file with 8 worker processes, loading their windower contexts
          from the windowers.bin file and replacing the file with new contexts in the end.
    python main.py flows.csv --windowing-settings windowing.yaml --n-jobs 8 --input-windowers windowers.bin --output-windowers windowers.bin --in-scaler scaler.bin
        - Processes the flows.csv file with 8 worker processes, loading their windower contexts
          from the windowers.bin file and replacing the file with new contexts in the end, using normalization
          scaler from the file scaler.bin.

Note about number of worker CPUs (--n-jobs):
    According to our empirical observations, the best results are achieved with 6-8 worker
    processes. Too few worker processes (1-2) cause heavy usage of a single process. Conversely, a
    very high number (10+) causes processes to be utilized inefficiently, while increasing overall
    computation time due to synchronization costs.

Note about output comparisons:
    In order to evalute and compare data processing results (such as comparison of this serial)
    version to parallel versions when ran through parallelproc.py wrapper, we set Merge sort as
    the sorting algorithm for sorting based on the flow ending timestamp. Merge sort is a stable
    sorting algorithm, thus producing consistent results among different runs with various sizes
    of input data.

Example of a config file:

window_size: 1
windower_params:
  win_min_entries: 4
  win_min_cnt: 4
  win_timeout: 500
  flow_winspan_max_len: 1000
  samples_cnt: 30
  win_max_cnt: 50
"""

import argparse
import ipaddress
import os
import pathlib
import pickle
import pandas as pd
import sys
import yaml
from typing import Optional

# Append to system path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'windowing'))


import vnet_nanodec.utils as utils
import vnet_nanodec.defines as defines
from . import vnet_feature_extraction
from . import vnet_preprocessing
from . import scaling_config_creator as scc


# Configuration keys defaults
CONFIG_KEY_WINDOW_SIZE = 'window_size'
CONFIG_KEY_WINDOWER_PARAMS = 'windower_params'

# Program parameters defaults
DEFAULT_N_JOBS = 0
DEFAULT_OUTPUT_DIRECTORY = '.'
DEFAULT_INPUT_WINDOWERS  = None
DEFAULT_OUTPUT_WINDOWERS = None
DEFAULT_OUTPUT_SUFFIX    = '_preproc'
DEFAULT_OUT_FILE_FORMAT  = 'csv'
DEFAULT_IP_FILTER_FILEPATH   = None
DEFAULT_SCALING_CFG_FILEPATH = None
DEFAULT_SCALING_MINMAX_MAX   = None


def data_prepare(
        src_flows_filepath: str,
        windowing_config_filepath: str,
        windowing_context_in_filepath: Optional[str] = DEFAULT_INPUT_WINDOWERS,
        windowing_context_out_filepath: Optional[str] = DEFAULT_OUTPUT_WINDOWERS,
        scaling_config_filepath: Optional[dict] = DEFAULT_SCALING_CFG_FILEPATH,
        scaling_minmax_new_max: Optional[float] = DEFAULT_SCALING_MINMAX_MAX,
        preproc_data_suffix: str = DEFAULT_OUTPUT_SUFFIX,
        out_dir: str = DEFAULT_OUTPUT_DIRECTORY,
        out_file_format: str = DEFAULT_OUT_FILE_FORMAT,
        ip_filter_filepath: Optional[str] = DEFAULT_IP_FILTER_FILEPATH,
        n_jobs: int = DEFAULT_N_JOBS,
        sync_timestamp: Optional[int] = None
) -> None:
    """Data preparation routine, includes IP filtration, preprocessing, windowing and sorting.

    The function loads the data and filters out entries with "uninteresting" IP addresses. Then,
    sorting by flow end timestamp is applied to faciliate the windowing procedure, performed in
    the next step. Finally, data are preprocessed by scaling, categorical variables encoding, and
    removal of unecessary columns.

    Parameters:
        sync_timestamp -- used for synchronization of parallel windowers, only for an external call
    """

    scaling_config       = None    # Scaling configuration to use during preprocessing
    ip_filter_df         = None    # IP filter to select IP addresses subset
    window_size_sec      = None    # Windowing size in seconds
    windower_params      = None    # Windower settings, ignored if contexts are loaded
    windowing_config     = None    # Windowing configuration loaded from the file
    windowing_context_in = None    # Windowing context for continuous processing

    # Import desired flow processor type based on the specified number of processes
    if n_jobs == 0:
        from windowing.flow_processor import process_flows
    elif _is_nonzero_n_jobs_valid(n_jobs):
        from windowing.flow_processor_multiproc import process_flows
    else:
        raise ValueError('Invalid number of worker processes')

    # Load IP filter if specified
    if ip_filter_filepath:
        ip_filter_df = utils.load_df(ip_filter_filepath)

    # Load windowing configuration
    with open(windowing_config_filepath, 'r') as windowing_config_file:
        windowing_config = yaml.safe_load(windowing_config_file)

    window_size_sec = windowing_config[CONFIG_KEY_WINDOW_SIZE]
    windower_params = windowing_config[CONFIG_KEY_WINDOWER_PARAMS]

    # Load the data and perform IP filtering (if any)
    dset = utils.load_df(src_flows_filepath, dtypes=defines.FEATURES_CASTDICT)
    dset = _filter_flows_by_ip(dset, ip_filter_df)

    # Sort all entries based on their flow ending timestamp for the windowing purposes
    dset = dset.sort_values(by=defines.COLNAME_FLOW_END_TSTAMP, kind='mergesort',
        ignore_index=True)

    # Set the default windowing context if desired
    # Synchronization timestamp for parallel windowers synchronization
    if sync_timestamp is not None:
        windowing_context_in = (None,
            sync_timestamp + utils.seconds_to_milliseconds(window_size_sec))

    # Load windower context if specified
    if windowing_context_in_filepath is not None:
        try:
            with open(windowing_context_in_filepath, 'rb') as infile:
                # Replace the default windowing context value
                windowing_context_in = pickle.load(infile)
        except FileNotFoundError:
            # If the file does not exist, keep the default context
            # This allows to process multiple files with passing context without the first one
            pass

    # Open file specifying feature scaling types if provided
    try:
        with open(scaling_config_filepath, 'r') as scaling_cfg_file:
            scaling_config = yaml.safe_load(scaling_cfg_file)[scc.SCALING_STATS_CONFIG_DEFAULT]
    except TypeError:
        # Configuration for feature scaling is None, keep the scaling config as None
        scaling_config = None
    except FileNotFoundError:
        print("The specified feature scaling configuration file {} does not exist.".format(
            scaling_config_filepath), file=sys.stderr)
        sys.exit(1)
    except KeyError:
        # The configuration file does not contain a required structure,
        print("The configuration file {} does not have a required structure.".format(
            scaling_config_filepath), file=sys.stderr)
        sys.exit(1)

    # Perform windowing
    flows_winstats, windowing_context_out = process_flows(
        dset, vnet_feature_extraction.extract_features, windower_params,
        window_size_sec, windowing_context_in, n_jobs)

    # Perform preprocessing of both dataset and window data
    dset_with_winstats = dset.join(flows_winstats)
    dset_with_winstats = vnet_preprocessing.preprocess(dset_with_winstats, scaling_config,
        scaling_minmax_new_max)

    # Save the preprocessed and windowed data
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    dst_preproc_filepath_root = os.path.join(out_dir,
        os.path.splitext(os.path.basename(src_flows_filepath))[0] + preproc_data_suffix)

    utils.save_df(dset_with_winstats, dst_preproc_filepath_root, out_file_format)

    # Save windowing context if desired
    if windowing_context_out_filepath is not None:
        with open(windowing_context_out_filepath, 'wb') as outfile:
            pickle.dump(windowing_context_out, outfile)


def _is_nonzero_n_jobs_valid(n_jobs: int) -> bool:
    if os.name == 'posix':
        return 0 < n_jobs <= len(os.sched_getaffinity(0))
    else:
        return n_jobs > 0


def _filter_flows_by_ip(dset: pd.DataFrame, ip_filter: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Reduces the input DataFrame according to the provided DataFrame with  with the
    column-separated list of IP addresses.  If the source or destination IP from the original
    DataFrame is contained within the provided IP list - the dataframe column is kept.
    Otherwise, it is removed from the dataset.

    Parameters:
        dset      -- DataFrame to process
        ip_filter -- Path to the file containing IP addresses in the CSV ormat

    Returns:
        pd.DataFrame -- Pandas DataFrame with entries containing only given IP addresses
    """

    if ip_filter is None:
        return dset

    # Assume that the first column contains IP addresses
    ips_to_keep = ip_filter[ip_filter.columns[0]].apply(
        lambda x: ipaddress.ip_address(x)).tolist()
    ipv4_set_to_keep = set(
        ip_address for ip_address in ips_to_keep
        if isinstance(ip_address, ipaddress.IPv4Address))
    ipv6_set_to_keep = set(
        ip_address for ip_address in ips_to_keep
        if isinstance(ip_address, ipaddress.IPv6Address))

    cond = (
        dset['IPV4_SRC_ADDR'].apply(ipaddress.IPv4Address).isin(ipv4_set_to_keep)
        | dset['IPV4_DST_ADDR'].apply(ipaddress.IPv4Address).isin(ipv4_set_to_keep)
        | dset['IPV6_SRC_ADDR'].apply(ipaddress.IPv6Address).isin(ipv6_set_to_keep)
        | dset['IPV6_DST_ADDR'].apply(ipaddress.IPv6Address).isin(ipv6_set_to_keep)
    )

    dset_filtered = dset[cond].reset_index(drop=True)

    return dset_filtered


def parse_args(raw_args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Preprocesses network flows in the specified file.')

    parser.add_argument(
        'flows_filepath',
        help='File path containing flows to process.',
    )
    parser.add_argument(
        '--windowing-config',
        required=True,
        help='Path to the windowing YAML configuration file.'
    )
    parser.add_argument(
        '--input-windowers',
        default=DEFAULT_INPUT_WINDOWERS,
        help='Optional path to load windowing context from.',
    )
    parser.add_argument(
        '--output-windowers',
        default=DEFAULT_OUTPUT_WINDOWERS,
        help=('Optional path to write windowing contexts to.'
              ' Can be identical to --input-windowers to replace the loaded windower contexts.'),
    )
    parser.add_argument(
        '--scaling-config',
        default=DEFAULT_SCALING_CFG_FILEPATH,
        help='Path to load the scaling config from. If not provided, no scaling is performed.',
    )
    parser.add_argument(
        '--scaling-minmax-max',
        default=DEFAULT_SCALING_MINMAX_MAX,
        help='New maximum value for the minmax scaling.',
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=DEFAULT_N_JOBS,
        help=('Number of worker processes to use.'
              ' Value of 0 indicates to use a single process.'
              f' Default: {DEFAULT_N_JOBS}'),
    )
    parser.add_argument(
        '--suffix',
        default=DEFAULT_OUTPUT_SUFFIX,
        help=('Suffix for filenames containing preprocessed data.'
              f' Default: {DEFAULT_OUTPUT_SUFFIX}'),
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=('Directory path to write preprocessed output to, defaults to current directory.')
    )
    parser.add_argument(
        '--out-file-format',
        default=DEFAULT_OUT_FILE_FORMAT,
        choices=utils.VALID_OUTPUT_FILE_FORMATS,
        help=('File extension (and file format) of the output (preprocessed) file.'
              f' Default: {DEFAULT_OUT_FILE_FORMAT}'),
    )
    parser.add_argument(
        '--ip-filter',
        default=DEFAULT_IP_FILTER_FILEPATH,
        help=('Path to file containing a list of IP addresses to filter flows by.'
              ' This is assumed to be a CSV file with a header and one column.'
              ' Flows with source of destination IP address not matching any IP address in the list'
              ' will be removed.'),
    )

    return parser.parse_args(raw_args)


def main(raw_args: list) -> None:
    args = parse_args(raw_args)

    data_prepare(
        src_flows_filepath=args.flows_filepath,
        windowing_config_filepath=args.windowing_config,
        windowing_context_in_filepath=args.input_windowers,
        windowing_context_out_filepath=args.output_windowers,
        scaling_config_filepath=args.scaling_config,
        scaling_minmax_new_max=args.scaling_minmax_max,
        preproc_data_suffix=args.suffix,
        out_dir=args.out_dir,
        out_file_format=args.out_file_format,
        ip_filter_filepath=args.ip_filter,
        n_jobs=args.n_jobs,
        sync_timestamp=None
    )

if __name__ == '__main__':
    main(sys.argv[1:])
