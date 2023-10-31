"""
Computation of statistics from network flows for clustering ISP-owned IP addresses.
"""

import argparse
import collections
import datetime
import ipaddress
import os
import pathlib
import sys

import hyperloglog
import joblib
import pandas as pd

# Append to system path for imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)).rsplit('src', 1)[0])

import config
import ipcategories
import ipnetworks
import src.utils as utils


DEFAULT_INPUT_CSV_SEPARATOR = '|'
DEFAULT_N_JOBS = 1
DEFAULT_HYPERLOGLOG_ERROR_RATE = config.HYPERLOGLOG_ERROR_RATE

_HYPERLOGLOG_TUPLE_DELIMITER = '_'

COLUMNS_TO_KEEP = [
    'IN_BYTES',
    'L4_SRC_PORT',
    'IPV4_SRC_ADDR',
    'IPV6_SRC_ADDR',
    'L4_DST_PORT',
    'IPV4_DST_ADDR',
    'IPV6_DST_ADDR',
    'OUT_BYTES',
    'DIRECTION',
]

_num_flows_statistics_attrs = [
    'num_original_flows', 'num_non_isp_flows_removed', 'num_zero_out_bytes_removed',
    'num_flows_src_only_ip', 'num_flows_dst_only_ip', 'num_flows_both_ip', 'num_flows']
_NumFlowsStatistics = collections.namedtuple('_NumFlowsStatistics', _num_flows_statistics_attrs)


def main(raw_args: list) -> None:
    args = parse_args(raw_args)

    os.makedirs(args.output_dir, exist_ok=True)

    ip_categories = ipcategories.load_ip_categories(args.ip_categories_paths)

    flow_filepaths_per_directory = list_flow_files_per_directory(args.input_dir)

    joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(compute_statistics)(
            dirpath,
            flow_filepaths,
            ip_categories,
            args.output_dir,
            args,
        )
        for dirpath, flow_filepaths in flow_filepaths_per_directory.items()
    )

    print('Done')


def compute_statistics(
        flows_dirpath: str,
        flow_filepaths: list[str],
        ip_categories: ipcategories.IPCategories,
        output_dirpath: str,
        args: argparse.Namespace,
) -> None:
    flows_dirpath_parts = pathlib.Path(flows_dirpath).parts
    flows_date = datetime.datetime.strptime('-'.join(flows_dirpath_parts[-4:-1]), '%Y-%m-%d')
    flows_hour = int(flows_dirpath_parts[-1])

    output_filepath = os.path.join(output_dirpath, '-'.join(flows_dirpath_parts[-4:]) + '.pkl-zip')

    if os.path.isfile(output_filepath) and not args.force:
        print(f'Output path {output_filepath} already exists, skipping')
        return

    accumulated_data_per_ip = {}

    for flow_filepath in flow_filepaths:
        print(f'Processing {flow_filepath}', flush=True)

        df_flows = utils.load_df(flow_filepath, csv_sep=args.input_csv_sep)
        df_flows, num_flows_stats = process_flows(df_flows, ip_categories, args.use_direction)

        df_flows.groupby(by=config.ISP_OWNED_IP_ADDR_COLUMN, sort=False, as_index=False).apply(
            accumulate_data_per_ip,
            accumulated_data_per_ip, num_flows_stats, args.hyperloglog_error_rate)

    df_statistics = get_statistics(accumulated_data_per_ip, flows_date, flows_hour)
    df_statistics.index.name = config.ISP_OWNED_IP_ADDR_COLUMN

    print(f'Saving statistics to {output_filepath}', flush=True)

    df_statistics.to_pickle(output_filepath, compression='zip')


def process_flows(
        df_flows: pd.DataFrame,
        ip_categories: ipcategories.IPCategories,
        use_direction: bool,
) -> tuple[pd.DataFrame, _NumFlowsStatistics]:
    df_flows_processed = df_flows

    df_flows_processed = df_flows_processed.loc[:, COLUMNS_TO_KEEP]
    df_flows_processed = process_and_optimize_dtypes(df_flows_processed)
    df_flows_processed = remove_non_background_flows(df_flows_processed)

    df_flows_processed, num_flows_stats = process_ips_and_columns(
        df_flows_processed, ip_categories, use_direction)

    df_flows_processed = add_port_category(df_flows_processed, 'L4_SRC_PORT')
    df_flows_processed = add_port_category(df_flows_processed, 'L4_DST_PORT')

    return df_flows_processed, num_flows_stats


def list_flow_files_per_directory(dirpath: str) -> collections.defaultdict:
    flow_filepaths_per_directory = collections.defaultdict(list)

    for root_dirpath, dirnames, filenames in os.walk(dirpath, followlinks=True):
        root_abs_dirpath = os.path.abspath(root_dirpath)
        for filename in filenames:
            flow_filepaths_per_directory[root_abs_dirpath].append(
                os.path.join(root_abs_dirpath, filename))

    return flow_filepaths_per_directory


def process_and_optimize_dtypes(df_flows: pd.DataFrame) -> pd.DataFrame:
    ipv4_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']
    ipv6_cols = ['IPV6_SRC_ADDR', 'IPV6_DST_ADDR']
    int64_cols = [
        'IN_BYTES',
        'OUT_BYTES',
        'FLOW_START_MILLISECONDS',
        'FLOW_END_MILLISECONDS',
        'SRC_FRAGMENTS',
        'DST_FRAGMENTS',
        'SRC_TO_DST_AVG_THROUGHPUT',
        'DST_TO_SRC_AVG_THROUGHPUT',
        'DURATION_IN',
        'DURATION_OUT',
    ]
    other_cols = ['Label', 'SRC_DENY', 'DST_DENY', 'Background_filter']

    for col in ipv4_cols:
        if col in df_flows:
            df_flows[col] = df_flows[col].apply(ipaddress.IPv4Address)

    for col in ipv6_cols:
        if col in df_flows:
            df_flows[col] = df_flows[col].apply(ipaddress.IPv6Address)

    int32_cols = [
        col for col in df_flows.columns if
        col not in (ipv4_cols + ipv6_cols + int64_cols + other_cols)
    ]

    df_flows.loc[:, int32_cols] = df_flows.loc[:, int32_cols].astype('int32')

    return df_flows


def remove_non_background_flows(
        df_flows: pd.DataFrame, label_col: str = 'Label', background_label: str = 'background',
) -> pd.DataFrame:
    if label_col in df_flows.columns:
        return df_flows.loc[df_flows[label_col].str.lower() == background_label, :]
    else:
        return df_flows


def process_ips_and_columns(
        df_flows: pd.DataFrame,
        ip_categories: ipcategories.IPCategories,
        use_direction: bool,
) -> tuple[pd.DataFrame, _NumFlowsStatistics]:
    df_flows_src_only_ip, df_flows_dst_only_ip, df_flows_both_ip = filter_ip_addresses(
        df_flows, ip_categories, use_direction=use_direction)

    num_original_flows = len(df_flows)
    num_non_isp_flows_removed = (
        num_original_flows
        - (len(df_flows_src_only_ip) + len(df_flows_dst_only_ip) + len(df_flows_both_ip)))

    df_flows_dst_only_ip_processed = remove_flows_with_zero_out_bytes(df_flows_dst_only_ip)

    num_zero_out_bytes_removed = len(df_flows_dst_only_ip) - len(df_flows_dst_only_ip_processed)
    num_flows_src_only_ip = len(df_flows_src_only_ip)
    num_flows_dst_only_ip = len(df_flows_dst_only_ip_processed)
    num_flows_both_ip = len(df_flows_both_ip)
    num_flows = num_flows_src_only_ip + num_flows_dst_only_ip + num_flows_both_ip

    df_flows_dst_only_ip_processed_swapped = swap_src_dst_columns(df_flows_dst_only_ip_processed)

    df_flows_both_ip_copy = df_flows_both_ip.copy()
    df_flows_both_ip_copy_processed = remove_flows_with_zero_out_bytes(df_flows_both_ip_copy)
    df_flows_both_ip_copy_processed_swapped = swap_src_dst_columns(df_flows_both_ip_copy_processed)

    df_flows_processed = pd.concat(
        [df_flows_src_only_ip,
         df_flows_dst_only_ip_processed_swapped,
         df_flows_both_ip,
         df_flows_both_ip_copy_processed_swapped],
        ignore_index=True,
    )

    df_flows_processed = merge_ip_address_columns(df_flows_processed)

    # Convert IPs back to strings so that they could be used as keys in sets/dicts
    df_flows_processed['IP_SRC_ADDR'] = df_flows_processed['IP_SRC_ADDR'].astype(str)
    df_flows_processed['IP_DST_ADDR'] = df_flows_processed['IP_DST_ADDR'].astype(str)

    num_flows_statistics = _NumFlowsStatistics(
        num_original_flows, num_non_isp_flows_removed, num_zero_out_bytes_removed,
        num_flows_src_only_ip, num_flows_dst_only_ip, num_flows_both_ip, num_flows,
    )

    return df_flows_processed, num_flows_statistics


def filter_ip_addresses(
        df_flows: pd.DataFrame,
        ip_categories: ipcategories.IPCategories,
        use_direction: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    int_ipv4_networks, ipv4_network_prefixlens, int_ipv6_networks, ipv6_network_prefixlens = (
        ipnetworks.get_int_ip_networks_and_prefixlens(ip_categories))

    src_cond = (
            df_flows['IPV4_SRC_ADDR'].isin(ip_categories.ipv4_set)
            | df_flows['IPV6_SRC_ADDR'].isin(ip_categories.ipv6_set)
            | df_flows['IPV4_SRC_ADDR'].apply(
                lambda x: ipnetworks.is_ip_in_networks(x, int_ipv4_networks, ipv4_network_prefixlens, 4))
            | df_flows['IPV6_SRC_ADDR'].apply(
                lambda x: ipnetworks.is_ip_in_networks(x, int_ipv6_networks, ipv6_network_prefixlens, 6))
    )
    dst_cond = (
            df_flows['IPV4_DST_ADDR'].isin(ip_categories.ipv4_set)
            | df_flows['IPV6_DST_ADDR'].isin(ip_categories.ipv6_set)
            | df_flows['IPV4_DST_ADDR'].apply(
                lambda x: ipnetworks.is_ip_in_networks(x, int_ipv4_networks, ipv4_network_prefixlens, 4))
            | df_flows['IPV6_DST_ADDR'].apply(
                lambda x: ipnetworks.is_ip_in_networks(x, int_ipv6_networks, ipv6_network_prefixlens, 6))
    )
    src_and_dst_cond = src_cond & dst_cond
    src_or_dst_cond = src_cond | dst_cond

    if use_direction:
        direction_swapped_cond = df_flows['DIRECTION'] == 1

        df_flows_src_only_ip = df_flows[
            direction_swapped_cond & src_or_dst_cond & ~src_and_dst_cond]
        df_flows_dst_only_ip = df_flows[
            ~direction_swapped_cond & src_or_dst_cond & ~src_and_dst_cond]
        df_flows_both_ip = df_flows[src_and_dst_cond]
    else:
        df_flows_src_only_ip = df_flows[src_cond & ~src_and_dst_cond]
        df_flows_dst_only_ip = df_flows[dst_cond & ~src_and_dst_cond]
        df_flows_both_ip = df_flows[src_and_dst_cond]

    return df_flows_src_only_ip, df_flows_dst_only_ip, df_flows_both_ip


def remove_flows_with_zero_out_bytes(df_flows: pd.DataFrame) -> pd.DataFrame:
    return df_flows[df_flows['OUT_BYTES'] != 0]


def swap_src_dst_columns(df_flows: pd.DataFrame) -> pd.DataFrame:
    new_columns = []

    for col_name in df_flows.columns:
        if col_name.startswith('IN_'):
            new_col_name = 'OUT_' + col_name[len('IN_'):]
        elif col_name.startswith('OUT_'):
            new_col_name = 'IN_' + col_name[len('OUT_'):]
        elif col_name.startswith('CLIENT_'):
            new_col_name = 'SERVER_' + col_name[len('CLIENT_'):]
        elif col_name.startswith('SERVER_'):
            new_col_name = 'CLIENT_' + col_name[len('SERVER_'):]
        elif col_name.startswith('DST_TO_SRC'):
            new_col_name = 'SRC_TO_DST' + col_name[len('DST_TO_SRC'):]
        elif col_name.startswith('SRC_TO_DST'):
            new_col_name = 'DST_TO_SRC' + col_name[len('SRC_TO_DST'):]
        elif '_SRC_' in col_name:
            new_col_name = col_name.replace('_SRC_', '_DST_')
        elif '_DST_' in col_name:
            new_col_name = col_name.replace('_DST_', '_SRC_')
        else:
            new_col_name = col_name

        new_columns.append(new_col_name)

    # Make a copy in case `df_flows` is reused later.
    df_flows_swapped = df_flows.copy()
    df_flows_swapped.columns = new_columns
    df_flows_swapped = df_flows_swapped[df_flows.columns]

    return df_flows_swapped


def merge_ip_address_columns(df_flows: pd.DataFrame) -> pd.DataFrame:
    df_flows_processed = df_flows

    df_flows_processed['IP_SRC_ADDR'] = df_flows_processed['IPV4_SRC_ADDR']
    df_flows_processed.loc[
        df_flows_processed['IPV6_SRC_ADDR'] != ipaddress.ip_address('::'), 'IP_SRC_ADDR'
    ] = df_flows_processed['IPV6_SRC_ADDR']

    df_flows_processed['IP_DST_ADDR'] = df_flows_processed['IPV4_DST_ADDR']
    df_flows_processed.loc[
        df_flows_processed['IPV6_DST_ADDR'] != ipaddress.ip_address('::'), 'IP_DST_ADDR'
    ] = df_flows_processed['IPV6_DST_ADDR']

    df_flows_processed = df_flows_processed.drop(
        ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'IPV6_SRC_ADDR', 'IPV6_DST_ADDR'], axis=1)

    return df_flows_processed


def add_port_category(df_flows: pd.DataFrame, port_column: str) -> pd.DataFrame:
    df_flows[port_column + '_CATEGORY'] = df_flows[port_column].apply(
        lambda x: config.PORT_SEARCHDICT[x]).astype('int32')

    return df_flows


def accumulate_data_per_ip(
        df_flows_per_ip: pd.DataFrame,
        data: dict,
        num_flows_stats: _NumFlowsStatistics,
        hyperloglog_error_rate: float,
) -> None:
    # Avoids unnecessary overhead and an exception (for some reason, a group from an empty dataframe
    # has no `name` attribute).
    if len(df_flows_per_ip) == 0:
        return

    use_hyperloglog = hyperloglog_error_rate > 0

    ip_str = str(df_flows_per_ip.name)

    if ip_str not in data:
        data[ip_str] = {}
        data_per_ip = data[ip_str]

        for category in config.PORT_CATEGORIES_LIST:
            data_per_ip[f'num_src_ports_{category}'] = 0
            data_per_ip[f'num_dst_ports_{category}'] = 0

        data_per_ip['num_in_bytes'] = 0
        data_per_ip['num_out_bytes'] = 0
        data_per_ip['num_flows_per_ip'] = 0

        for attr_name in _num_flows_statistics_attrs:
            data_per_ip[attr_name] = 0

        if not use_hyperloglog:
            data_per_ip['uniques_src_port_dst_port'] = set()
            data_per_ip['uniques_dst_ip_src_port_category_dst_port_category'] = set()
        else:
            data_per_ip['uniques_src_port_dst_port'] = (
                hyperloglog.HyperLogLog(hyperloglog_error_rate))
            data_per_ip['uniques_dst_ip'] = (
                hyperloglog.HyperLogLog(hyperloglog_error_rate))
            data_per_ip['uniques_dst_ip_src_port_category'] = (
                hyperloglog.HyperLogLog(hyperloglog_error_rate))
            data_per_ip['uniques_dst_ip_dst_port_category'] = (
                hyperloglog.HyperLogLog(hyperloglog_error_rate))
            data_per_ip['uniques_dst_ip_src_port_category_dst_port_category'] = (
                hyperloglog.HyperLogLog(hyperloglog_error_rate))
    else:
        data_per_ip = data[ip_str]

    for category in config.PORT_CATEGORIES_LIST:
        data_per_ip[f'num_src_ports_{category}'] += (
            df_flows_per_ip['L4_SRC_PORT_CATEGORY'] == config.PORT_CATEGORIES_AND_IDS[category]
        ).sum()
        data_per_ip[f'num_dst_ports_{category}'] += (
            df_flows_per_ip['L4_DST_PORT_CATEGORY'] == config.PORT_CATEGORIES_AND_IDS[category]
        ).sum()

    # `num_in_bytes` refers to incoming traffic, which is counterintuitively represented by
    # `OUT_BYTES`. Same for `num_out_bytes` and `IN_BYTES`.
    data_per_ip['num_in_bytes'] += df_flows_per_ip['OUT_BYTES'].sum()
    data_per_ip['num_out_bytes'] += df_flows_per_ip['IN_BYTES'].sum()
    data_per_ip['num_flows_per_ip'] += len(df_flows_per_ip)

    for attr_name in _num_flows_statistics_attrs:
        data_per_ip[attr_name] += getattr(num_flows_stats, attr_name)

    if not use_hyperloglog:
        data_per_ip['uniques_src_port_dst_port'].update(
            df_flows_per_ip[
                ['L4_SRC_PORT', 'L4_DST_PORT']
            ].drop_duplicates().itertuples(index=False, name=None))
        data_per_ip['uniques_dst_ip_src_port_category_dst_port_category'].update(
            df_flows_per_ip[
                ['IP_DST_ADDR', 'L4_SRC_PORT_CATEGORY', 'L4_DST_PORT_CATEGORY']
            ].drop_duplicates().itertuples(index=False, name=None))
    else:
        _update_hyperloglog_set(
            df_flows_per_ip,
            ['L4_SRC_PORT', 'L4_DST_PORT'],
            data_per_ip['uniques_src_port_dst_port'])
        _update_hyperloglog_set(
            df_flows_per_ip,
            ['IP_DST_ADDR'],
            data_per_ip['uniques_dst_ip'])
        _update_hyperloglog_set(
            df_flows_per_ip,
            ['IP_DST_ADDR', 'L4_SRC_PORT_CATEGORY'],
            data_per_ip['uniques_dst_ip_src_port_category'])
        _update_hyperloglog_set(
            df_flows_per_ip,
            ['IP_DST_ADDR', 'L4_DST_PORT_CATEGORY'],
            data_per_ip['uniques_dst_ip_dst_port_category'])
        _update_hyperloglog_set(
            df_flows_per_ip,
            ['IP_DST_ADDR', 'L4_SRC_PORT_CATEGORY', 'L4_DST_PORT_CATEGORY'],
            data_per_ip['uniques_dst_ip_src_port_category_dst_port_category'])


def _update_hyperloglog_set(
        df: pd.DataFrame, cols: list[str], set_: hyperloglog.HyperLogLog,
) -> None:
    uniques = df[cols].drop_duplicates().astype(str)

    if len(uniques) > 0 and len(cols) > 0:
        uniques_str = uniques[cols[0]]

        for col in cols[1:]:
            uniques_str = uniques_str + _HYPERLOGLOG_TUPLE_DELIMITER + uniques[col]

        for _, value in uniques_str.items():
            set_.add(value)


def get_statistics(
        accumulated_data_per_ip: dict, date: datetime.datetime, hour: int,
) -> pd.DataFrame:
    statistics_per_ip = collections.defaultdict(dict)

    for ip, data_per_ip in accumulated_data_per_ip.items():
        for key in data_per_ip:
            if key.startswith('uniques_'):
                statistics_per_ip[ip][key] = data_per_ip[key]

        for category in config.PORT_CATEGORIES_LIST:
            statistics_per_ip[ip][f'num_src_ports_{category}'] = data_per_ip[
                f'num_src_ports_{category}']
            statistics_per_ip[ip][f'num_dst_ports_{category}'] = data_per_ip[
                f'num_dst_ports_{category}']

        statistics_per_ip[ip]['num_in_bytes'] = data_per_ip['num_in_bytes']
        statistics_per_ip[ip]['num_out_bytes'] = data_per_ip['num_out_bytes']
        statistics_per_ip[ip]['num_total_bytes'] = (
            statistics_per_ip[ip]['num_in_bytes'] + statistics_per_ip[ip]['num_out_bytes'])
        statistics_per_ip[ip]['num_flows_per_ip'] = data_per_ip['num_flows_per_ip']

        for attr_name in _num_flows_statistics_attrs:
            statistics_per_ip[ip][attr_name] = data_per_ip[attr_name]

        statistics_per_ip[ip]['date'] = date
        statistics_per_ip[ip]['day_type'] = (
            'workday' if date.strftime('%w') not in ['0', '6'] else 'weekend')
        statistics_per_ip[ip]['hour'] = hour

    return pd.DataFrame(statistics_per_ip).T


def parse_args(raw_args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=('Compute statistics from flows to be further used for clustering.'
                     ' Flows are processed such that ISP-owned IP addresses are always source'
                     ' addresses, along with ports and additional fields. This behavior can be'
                     ' adjusted via --use-direction.'
                     ' Statistics are computed per each subdirectory corresponding to an hour of'
                     ' captured data.'))

    parser.add_argument(
        'input_dir',
        help=('Directory path containing network flows.'
              ' Statistics are computed from all flow files in a single directory'),
    )
    parser.add_argument(
        'output_dir',
        help='Output directory.',
    )
    parser.add_argument(
        'ip_categories_paths',
        nargs='+',
        help=('Paths to CSV files containing ISP IP addresses or networks including their'
              ' categories.'),
    )
    parser.add_argument(
        '--use-direction',
        action='store_true',
        help=('If specified, use the DIRECTION field to determine which flows should have their'
              ' source and destination fields swapped.'),
    )
    parser.add_argument(
        '--hyperloglog-error-rate',
        type=float,
        default=DEFAULT_HYPERLOGLOG_ERROR_RATE,
        help=('HyperLogLog set error rate (between 0 and 1) for statistics based on unique tuple'
              ' counts. If this argument is 0 or less than 0, regular sets are used instead. Be'
              ' aware that regular sets require a substantial amount of memory and storage compared'
              ' to HyperLogLog sets.'
              f' Default: {DEFAULT_HYPERLOGLOG_ERROR_RATE}'),
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=DEFAULT_N_JOBS,
        help=('Number of jobs to run in parallel.'
              ' Value of -1 indicates to use all available CPUs.'
              f' Default: {DEFAULT_N_JOBS}'),
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help=('If specified, force computing statistics for a subdirectory even if the'
              ' corresponding output file already exists.'),
    )
    parser.add_argument(
        '--input-csv-sep',
        default=DEFAULT_INPUT_CSV_SEPARATOR,
        help=('Separator to use for input files if the file format is CSV or gzip.'
              f' Default: {DEFAULT_INPUT_CSV_SEPARATOR}'),
    )

    return parser.parse_args(raw_args)


if __name__ == '__main__':
    main(sys.argv[1:])
