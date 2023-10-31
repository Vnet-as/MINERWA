"""
Selection of the best hyperparameters for clustering using k-means on network profiles.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn import preprocessing as sklearn_preprocessing

# Append to system path for imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)).rsplit('src', 1)[0])

import src.utils as utils


MIN_NUM_CLUSTERS = 5
MAX_NUM_CLUSTERS = 9
FEATURE_SETS = 'all'
# Higher weight = lower significance
METRIC_INERTIA_WEIGHT = 1.0
METRIC_INVERSE_CLUSTER_COUNT_ENTROPY_WEIGHT = 2.0

DEFAULT_CSV_SEPARATOR = '|'


def main(raw_args: list) -> None:
    args = parse_args(raw_args)

    df_features = load_features(args.features_path)
    df_results = load_clustering_results(args.results_path)

    best_result = get_best_result(df_results)

    if args.filter:
        df_features = df_features[df_features['num_flows_per_ip'] > 1]

    save_output(df_features, best_result, args.output_path, args.csv_sep)


def load_clustering_results(filepath):
    with open(filepath, 'rb') as f:
        results = pickle.load(f)

    df_results = pd.DataFrame.from_records(results)

    # Allow grouping by feature sets
    if 'feature_set' in df_results.columns:
        df_results = df_results.rename(columns={'feature_set': 'feature_sets'})

    if 'feature_sets' in df_results.columns:
        df_results['feature_sets'] = df_results['feature_sets'].apply(
            lambda x: x if isinstance(x, str) else tuple(x))

    if 'feature_sets_and_scalers' in df_results.columns:
        df_results['feature_sets_and_scalers'] = df_results['feature_sets_and_scalers'].apply(
            lambda x: tuple(x.items()))

    if 'cluster_counts' in df_results.columns:
        df_results['cluster_count_entropy'] = df_results['cluster_counts'].apply(
            lambda x: scipy_stats.entropy(list(x.values())))
        df_results['inverse_cluster_count_entropy'] = 1 / (df_results['cluster_count_entropy'] + 1)

    return df_results


def load_features(features_filepath):
    df_features = utils.load_df(features_filepath)

    df_features = df_features.reset_index()
    df_features = df_features.rename(columns={'IP_SRC_ADDR': 'ip_address'})

    return df_features


def get_best_result(df_results):
    df_results_filtered = df_results.loc[
        ((df_results['clustering_method__n_clusters'] >= MIN_NUM_CLUSTERS)
         & (df_results['clustering_method__n_clusters'] <= MAX_NUM_CLUSTERS))
        & (df_results['feature_sets'] == FEATURE_SETS),
    ].copy()

    df_results_filtered['model__inertia_scaled'] = sklearn_preprocessing.minmax_scale(
        df_results_filtered['model__inertia'])
    df_results_filtered[
        'inverse_cluster_count_entropy_scaled'] = sklearn_preprocessing.minmax_scale(
        df_results_filtered['inverse_cluster_count_entropy'])

    df_results_filtered['best_cluster_metric'] = (
            ((df_results_filtered['model__inertia_scaled'] * METRIC_INERTIA_WEIGHT)
             + (df_results_filtered[
                    'inverse_cluster_count_entropy_scaled'] * METRIC_INVERSE_CLUSTER_COUNT_ENTROPY_WEIGHT))
            / 2
    )

    good_result_index = df_results_filtered.sort_values(by=['best_cluster_metric']).index[0]

    good_results = df_results_filtered.loc[
        good_result_index,
        [
            *[col for col in df_results.columns if col.startswith('clustering_method')],
            *[col for col in df_results.columns if col.startswith('feature_scaler')],
            'feature_sets',
            'feature_sets_and_scalers',
        ]
    ].to_dict()

    good_results.pop('clustering_method__n_clusters')

    # Find the "elbow" from inertia values - we will use that number of clusters.
    inertias = df_results_filtered[
        (df_results_filtered[good_results.keys()] == good_results.values()).all(axis=1)
    ].sort_values('clustering_method__n_clusters')['model__inertia']

    inertia_second_derivatives = np.gradient(np.gradient(inertias))
    best_result_index = inertias.index[np.argmax(inertia_second_derivatives)]
    best_result = df_results_filtered.loc[best_result_index]

    return best_result


def save_output(df_features, best_result, output_filepath, csv_sep):
    df_output = df_features['ip_address'].copy().to_frame()

    df_output['cluster_label'] = best_result['cluster_labels']

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    utils.save_df(df_output, output_filepath, csv_sep=csv_sep)


def parse_args(raw_args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=('Creates a list of IP addresses and cluster labels based on the best'
                     ' k-means clustering results.'))

    parser.add_argument(
        'features_path',
        help='File path containing extracted features - the output of "extract_features.py".',
    )
    parser.add_argument(
        'results_path',
        help='File path containing clustering results - the output of "cluster.py".',
    )
    parser.add_argument(
        'output_path',
        help='Output file path containing a list of IP addresses and clustering labels.',
    )
    parser.add_argument(
        '--csv-sep',
        default=DEFAULT_CSV_SEPARATOR,
        help=('Separator to use for output_path if the file format is CSV or gzip.'
              f' Default: {DEFAULT_CSV_SEPARATOR}'),
    )
    parser.add_argument(
        '--filter',
        action='store_true',
        help='If specified, filter irrelevant feature vectors (with one flow only).',
    )

    return parser.parse_args(raw_args)


if __name__ == '__main__':
    main(sys.argv[1:])
