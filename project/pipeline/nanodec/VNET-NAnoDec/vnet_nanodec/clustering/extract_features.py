"""
Extraction of features representing behavior of ISP-owned IP addresses.
"""

import argparse
import os
import sys
from typing import Union

import hyperloglog
import pandas as pd
from scipy import stats as scipy_stats

# Append to system path for imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)).rsplit('src', 1)[0])

import config
import src.utils as utils


def main(raw_args: list) -> None:
    args = parse_args(raw_args)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f'Loading statistics from "{args.input_dir}"')

    df_statistics = load_statistics(args.input_dir)

    print('Extracting features')

    df_features = df_statistics.groupby(config.ISP_OWNED_IP_ADDR_COLUMN, sort=False).apply(
        extract_features_per_ip)

    df_features = adjust_dtypes(df_features)

    print('Saving features')

    df_features.to_parquet(args.output_path)


def load_statistics(dirpath: str) -> pd.DataFrame:
    return pd.concat([
        utils.load_df(os.path.join(dirpath, filename))
        for filename in os.listdir(dirpath)
    ])


def extract_features_per_ip(statistics_per_ip: pd.DataFrame) -> pd.Series:
    if len(statistics_per_ip) <= 0:
        print('Empty dataframe, skipping')
        return

    print(f'Extracting features for IP {statistics_per_ip.name}')

    features_per_ip = {}

    num_flows_per_ip = statistics_per_ip['num_flows_per_ip'].sum()
    features_per_ip['num_flows_per_ip'] = num_flows_per_ip

    num_total_src_port_categories = {}
    num_total_dst_port_categories = {}
    for category in config.PORT_CATEGORIES_LIST:
        num_total_src_port_categories[category] = statistics_per_ip[
            f'num_src_ports_{category}'].sum()
        num_total_dst_port_categories[category] = statistics_per_ip[
            f'num_dst_ports_{category}'].sum()

    features_per_ip['src_port_category_entropy'] = scipy_stats.entropy(
        list(num_total_src_port_categories.values()))
    features_per_ip['dst_port_category_entropy'] = scipy_stats.entropy(
        list(num_total_dst_port_categories.values()))

    for category in config.PORT_CATEGORIES_LIST:
        features_per_ip[f'num_src_ports_{category}'] = num_total_src_port_categories[category]
        features_per_ip[f'num_dst_ports_{category}'] = num_total_dst_port_categories[category]

    for category in config.PORT_CATEGORIES_LIST:
        features_per_ip[f'src_port_probability_{category}'] = (
            num_total_src_port_categories[category] / num_flows_per_ip)
        features_per_ip[f'dst_port_probability_{category}'] = (
            num_total_dst_port_categories[category] / num_flows_per_ip)

    features_per_ip = add_unique_count_features(features_per_ip, statistics_per_ip)

    features_per_ip.update(get_total_bytes_per_hour_per_day_type(statistics_per_ip))

    features_per_ip['in_out_bytes_ratio'] = (
        statistics_per_ip['num_in_bytes'].sum()
        / (statistics_per_ip['num_out_bytes'].sum() + 1)  # Avoid division by zero
    )

    return pd.Series(features_per_ip)


def add_unique_count_features(features_per_ip: dict, statistics_per_ip: pd.DataFrame) -> dict:
    # Automatically infer how to compute unique counts
    if len(statistics_per_ip['uniques_src_port_dst_port']) > 0:
        if isinstance(statistics_per_ip['uniques_src_port_dst_port'].iloc[0], set):
            use_hyperloglog = False
        else:
            use_hyperloglog = True
    else:
        use_hyperloglog = False

    if not use_hyperloglog:
        features_per_ip['num_uniques_src_port_dst_port'] = len(
            set.union(*statistics_per_ip['uniques_src_port_dst_port'].tolist()))
        uniques_dst_ip_src_port_category_dst_port_category = set.union(
            *statistics_per_ip['uniques_dst_ip_src_port_category_dst_port_category'].tolist())
        features_per_ip['num_uniques_dst_ip'] = len(
            {item[0] for item in uniques_dst_ip_src_port_category_dst_port_category})
        features_per_ip['num_uniques_dst_ip_src_port_category'] = len(
            {(item[0], item[1]) for item in uniques_dst_ip_src_port_category_dst_port_category})
        features_per_ip['num_uniques_dst_ip_dst_port_category'] = len(
            {(item[0], item[2]) for item in uniques_dst_ip_src_port_category_dst_port_category})
        features_per_ip['num_uniques_dst_ip_src_port_category_dst_port_category'] = len(
            uniques_dst_ip_src_port_category_dst_port_category)
    else:
        for stat_name in statistics_per_ip.columns:
            if stat_name.startswith('uniques_'):
                features_per_ip['num_' + stat_name] = len(
                    _merge_hyperloglog_sets(statistics_per_ip[stat_name]))

    return features_per_ip


def _merge_hyperloglog_sets(
    sets: list[hyperloglog.HyperLogLog],
) -> Union[hyperloglog.HyperLogLog, set]:
    if len(sets) > 0:
        merged_set = sets[0]
        if len(sets) > 1:
            merged_set.update(*sets[1:])

        return merged_set
    else:
        return set()


def get_total_bytes_per_hour_per_day_type(statistics_per_ip: pd.DataFrame) -> dict:
    total_bytes_per_hour_per_day_type = statistics_per_ip[
        ['num_total_bytes', 'day_type', 'hour']
    ].groupby(by=['day_type', 'hour'], sort=True, as_index=False).sum()

    total_bytes_per_hour_per_day_type.index = (
        'num_bytes_'
        + total_bytes_per_hour_per_day_type['day_type']
        + '_'
        + total_bytes_per_hour_per_day_type['hour'].astype(str))

    total_bytes_per_hour_per_day_type = total_bytes_per_hour_per_day_type.drop(
        ['day_type', 'hour'], axis=1)
    total_bytes_per_hour_per_day_type = total_bytes_per_hour_per_day_type.T
    total_bytes_per_hour_per_day_type = total_bytes_per_hour_per_day_type.reset_index(drop=True)

    # Ensure all columns are present if some hours/day types are missing
    keys = [
        f'num_bytes_{day_type}_{hour}'
        for day_type in ['workday', 'weekend']
        for hour in range(0, 24)
    ]

    computed_total_bytes = total_bytes_per_hour_per_day_type.iloc[0].to_dict()
    total_bytes_per_hour_per_day_type_dict = {}

    # Ensure consistent order of columns when later creating pandas Series
    for key in keys:
        if key in computed_total_bytes:
            total_bytes_per_hour_per_day_type_dict[key] = computed_total_bytes[key]
        else:
            total_bytes_per_hour_per_day_type_dict[key] = 0

    return total_bytes_per_hour_per_day_type_dict


def adjust_dtypes(df_features: pd.DataFrame) -> pd.DataFrame:
    column_prefixes = [
        'num_src_ports_',
        'num_dst_ports_',
        'num_uniques_',
        'num_bytes_',
    ]

    for column in df_features.columns:
        if any(column.startswith(prefix) for prefix in column_prefixes):
            df_features[column] = df_features[column].astype(int)

    df_features['num_flows_per_ip'] = df_features['num_flows_per_ip'].astype(int)

    return df_features


def parse_args(raw_args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=('Extracts features from network flow statistics. A feature vector represents'
                     ' the typical behavior of an ISP-owned node.'))

    parser.add_argument(
        'input_dir',
        help='Directory path containing computed statistics.',
    )
    parser.add_argument(
        'output_path',
        help='File path to write features to.',
    )

    return parser.parse_args(raw_args)


if __name__ == '__main__':
    main(sys.argv[1:])
