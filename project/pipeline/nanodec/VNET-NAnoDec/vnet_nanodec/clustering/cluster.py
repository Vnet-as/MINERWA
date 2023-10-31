"""
Clustering of samples, each representing behavior of an ISP-owned IP address.
"""

import argparse
import datetime
import inspect
import ipaddress
import json
import os
import pickle
import sys
from typing import Any, Union
import warnings

import joblib
import pandas as pd

from sklearn import metrics as sklearn_metrics
from sklearn import model_selection as sklearn_model_selection

# Append to system path for imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)).rsplit('src', 1)[0])

import ipcategories
import ipnetworks
import src.utils as utils


DEFAULT_N_JOBS = 1

_FEATURE_SETS = {
    'num_bytes_per_hour': lambda names: [
        name for name in names
        if name.startswith('num_bytes_workday_') or name.startswith('num_bytes_weekend_')],
    'in_out_bytes_ratio': lambda names: ['in_out_bytes_ratio'],
    'port_entropy': lambda names: ['src_port_category_entropy', 'dst_port_category_entropy'],
    'port_probability': lambda names: [
        name for name in names
        if (name.startswith('src_port_probability_') or name.startswith('dst_port_probability_'))],
    'num_unique_tuples': lambda names: [
        name for name in names if name.startswith('num_uniques_')],
}


def main(raw_args: list) -> None:
    args = parse_args(raw_args)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    ip_categories = ipcategories.load_ip_categories(args.ip_categories_paths)

    print(f'Loading features from "{args.features_path}"')

    df_features, df_labels = load_features(args.features_path, ip_categories)

    if args.filter:
        df_features, df_labels = filter_features(df_features, df_labels)

    df_features = remove_irrelevant_columns(df_features)

    param_grid = load_param_grid(args.params_path)
    params_list = process_param_grid(param_grid)

    results = joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(cluster_samples)(
            df_features,
            df_labels,
            params,
            i,
            len(params_list),
        )
        for i, params in enumerate(params_list)
    )

    with open(args.output_path, 'wb') as f:
        pickle.dump(results, f)


def load_param_grid(filepath: str) -> sklearn_model_selection.ParameterGrid:
    with open(filepath, 'r') as f:
        param_grid_raw = json.load(f)

    return sklearn_model_selection.ParameterGrid(param_grid_raw)


def load_features(
        filepath: str, ip_categories: ipcategories.IPCategories, remove_unknown_ips: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_features = utils.load_df(filepath)

    df_labels = get_labels(df_features, ip_categories).reset_index(drop=True)

    if remove_unknown_ips:
        df_labels = df_labels[df_labels['category'] != 'Unknown'].reset_index(drop=True)
        df_features = df_features.iloc[df_labels.index].copy()

    return df_features, df_labels


def filter_features(
        df_features: pd.DataFrame, df_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_features_filtered = df_features[df_features['num_flows_per_ip'] > 1].copy()
    df_labels_filtered = df_labels[
        df_labels['ip_address'].isin(df_features_filtered.index)
    ].reset_index(drop=True)

    return df_features_filtered, df_labels_filtered


def remove_irrelevant_columns(df_features: pd.DataFrame) -> pd.DataFrame:
    return df_features.drop(
        [col for col in df_features.columns
         if (col.startswith('num_src_ports')
             or col.startswith('num_dst_ports')
             or col == 'num_flows_per_ip')],
        axis=1,
        errors='ignore')


def get_labels(df_features: pd.DataFrame, ip_categories: ipcategories.IPCategories) -> pd.DataFrame:
    df_labels = pd.DataFrame()
    df_labels['ip_address'] = df_features.index.tolist()
    df_labels['ip_address_obj'] = df_labels['ip_address'].apply(ipaddress.ip_address)

    df_labels = df_labels.merge(
        ip_categories.ip_categories[['ip', 'type']], how='left', left_on='ip_address', right_on='ip')
    del df_labels['ip']
    df_labels = df_labels.rename(columns={'type': 'category'})
    df_labels = df_labels.drop_duplicates(subset='ip_address', keep='first')

    int_ip_networks_and_prefixlens = ipnetworks.get_int_ip_networks_and_prefixlens(ip_categories)

    df_labels.loc[df_labels['category'].isnull(), 'category'] = df_labels.loc[
        df_labels['category'].isnull()
    ].apply(
        get_category_from_ip_address, args=(ip_categories, int_ip_networks_and_prefixlens), axis=1)

    return df_labels


def get_category_from_ip_address(
        row: pd.Series,
        ip_categories: ipcategories.IPCategories,
        int_ip_networks_and_prefixlens: tuple,
) -> str:
    ip_addr_obj = row['ip_address_obj']

    if ip_addr_obj in ip_categories.ip_set:
        return ip_categories.ip_categories.loc[
            ip_categories.ip_categories['ip'] == row['ip_address'], 'type'].iloc[0]
    else:
        if isinstance(ip_addr_obj, ipaddress.IPv4Address):
            int_ip_networks, ip_network_prefixlens, version = (
                int_ip_networks_and_prefixlens[0], int_ip_networks_and_prefixlens[1], 4)
        else:
            int_ip_networks, ip_network_prefixlens, version = (
                int_ip_networks_and_prefixlens[2], int_ip_networks_and_prefixlens[3], 6)

        result = ipnetworks.get_smallest_matching_network_ip_address(
            ip_addr_obj, int_ip_networks, ip_network_prefixlens, version)

        if result is not None:
            if version == 4:
                ip_network = ipaddress.IPv4Network(result)
            else:
                ip_network = ipaddress.IPv6Network(result)

            return ip_categories.ip_categories.loc[
                ip_categories.ip_categories['ip'] == str(ip_network), 'type'].iloc[0]
        else:
            return 'Unknown'


def process_param_grid(param_grid: sklearn_model_selection.ParameterGrid) -> list:
    params_list = list(param_grid)

    _expand_all_feature_set_in_feature_sets_and_scalers(params_list)
    params_list = _remove_feature_sets_and_scalers_not_in_feature_set(params_list)
    params_list = _deduplicate_params_list(params_list)

    return params_list


def _expand_all_feature_set_in_feature_sets_and_scalers(params_list: list) -> None:
    for params in params_list:
        if 'all' not in params['feature_sets_and_scalers']:
            raise ValueError('"all" must be always specified in "feature_sets_and_scalers" field')

        all_feature_set_and_scaler = params['feature_sets_and_scalers']['all']
        new_feature_sets_and_scalers = {
            set_name: all_feature_set_and_scaler for set_name in _FEATURE_SETS}
        for set_name, scaler in params['feature_sets_and_scalers'].items():
            if set_name == 'all':
                continue
            new_feature_sets_and_scalers[set_name] = scaler

        params['feature_sets_and_scalers'] = new_feature_sets_and_scalers


def _remove_feature_sets_and_scalers_not_in_feature_set(params_list: list) -> list:
    for params in params_list:
        params['feature_sets_and_scalers'] = {
            set_name: scaler for set_name, scaler in params['feature_sets_and_scalers'].items()
            if params['feature_sets'] == 'all' or set_name in params['feature_sets']}

    return params_list


def _deduplicate_params_list(params_list: list) -> list:
    return list({str(params): params for params in params_list}.values())


def cluster_samples(
        df_features: pd.DataFrame,
        df_labels: pd.DataFrame,
        params: dict,
        param_index: int,
        len_param_grid: int,
) -> dict:
    print(f'Evaluating {param_index + 1} out of {len_param_grid} clustering hyperparameters')

    df_features_processed = df_features[
        get_feature_subset(df_features.columns, params['feature_sets'])].copy()

    if len(df_features_processed.columns) == 0:
        warnings.warn(f'Feature sets {params["feature_sets"]} did not resolve to any columns')
        return None

    scale_features(df_features_processed, params)

    _, model_class = utils.load_module_and_obj(params['clustering_method'])
    model = model_class(**get_params(params, 'clustering_method', model_class))
    cluster_labels = model.fit_predict(df_features_processed.to_numpy())

    # Create a copy to avoid overriding by other threads/processes.
    df_labels_with_result = df_labels.copy()
    df_labels_with_result['cluster_label'] = cluster_labels

    metrics = get_clustering_metrics(df_labels_with_result)
    model_attributes = get_model_attributes(model)

    results = assemble_results(df_labels_with_result, metrics, model_attributes, params)

    return results


def get_feature_subset(
        feature_names: Union[list, pd.Index], feature_sets: Union[str, list[str]],
) -> list[str]:
    if feature_sets == 'all':
        return list(feature_names)
    else:
        subset_names = []
        for set_name in feature_sets:
            if set_name in _FEATURE_SETS:
                subset_names += _FEATURE_SETS[set_name](feature_names)

        return subset_names


def get_params(params: dict, estimator_name: str, estimator_class: Any) -> dict:
    estimator_params_prefix = estimator_name + '__'
    estimator_signature_params = inspect.signature(estimator_class).parameters
    estimator_params = {
        name[len(estimator_params_prefix):]: value
        for name, value in params.items()
        if (name.startswith(estimator_params_prefix)
            and name[len(estimator_params_prefix):] in estimator_signature_params)
    }

    if 'random_state' in estimator_signature_params and 'random_state' in params:
        estimator_params['random_state'] = params['random_state']

    return estimator_params


def scale_features(df_features_processed: pd.DataFrame, params: dict) -> None:
    for feature_set, scaler_name in params['feature_sets_and_scalers'].items():
        _, scaler_class = utils.load_module_and_obj(scaler_name)
        scaler = scaler_class(**get_params(params, 'feature_scaler', scaler_class))

        feature_subset_names = get_feature_subset(df_features_processed.columns, [feature_set])

        if not feature_subset_names:
            warnings.warn(f'Feature set {feature_set} did not resolve to any columns')
            continue

        df_features_processed.loc[:, feature_subset_names] = scaler.fit_transform(
            df_features_processed[feature_subset_names].to_numpy())


def get_clustering_metrics(df_labels: pd.DataFrame) -> dict:
    supervised_metrics = {
        'homogeneity': sklearn_metrics.homogeneity_score(
            df_labels['category'], df_labels['cluster_label']),
        'completeness': sklearn_metrics.completeness_score(
            df_labels['category'], df_labels['cluster_label']),
        'v_measure': sklearn_metrics.v_measure_score(
            df_labels['category'], df_labels['cluster_label']),
    }

    return supervised_metrics


def get_model_attributes(model: Any) -> dict:
    return {
        attr_name.rstrip('_'): getattr(model, attr_name)
        for attr_name in dir(model)
        if attr_name.endswith('_') and not attr_name.startswith('_')
    }


def assemble_results(
        df_labels: pd.DataFrame, metrics: dict, model_attributes: dict, params: dict,
) -> dict:
    cluster_counts = df_labels['cluster_label'].value_counts()
    cluster_counts_per_category = df_labels[['category', 'cluster_label']].value_counts()

    return dict(
        params,
        **{f'metric__{name}': value for name, value in metrics.items()},
        **{f'model__{name}': value for name, value in model_attributes.items()},
        cluster_counts=cluster_counts.to_dict(),
        cluster_counts_per_category=cluster_counts_per_category.to_dict(),
        cluster_labels=df_labels['cluster_label'].to_numpy(),
        date=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'),
        num_samples=len(df_labels),
    )


def parse_args(raw_args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Performs clustering on features computed from network flow statistics.')

    parser.add_argument(
        'features_path',
        help='File path containing extracted features.',
    )
    parser.add_argument(
        'params_path',
        help='Path to a JSON file containing hyperparameters.',
    )
    parser.add_argument(
        'output_path',
        help='File path to write output to.',
    )
    parser.add_argument(
        'ip_categories_paths',
        nargs='+',
        help=('Paths to CSV files containing ISP IP addresses or networks including their'
              ' categories.'),
    )
    parser.add_argument(
        '--filter',
        action='store_true',
        help=('If specified, filter irrelevant feature vectors'
              ' (with one flow only).'),
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=DEFAULT_N_JOBS,
        help=('Number of jobs to run in parallel.'
              ' Value of -1 indicates to use all available CPUs.'
              f' Default: {DEFAULT_N_JOBS}'),
    )

    return parser.parse_args(raw_args)


if __name__ == '__main__':
    main(sys.argv[1:])
