import argparse
import ipaddress
import os
import sys

import joblib
import pandas as pd

# Append to system path for imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)).rsplit('src', 1)[0])
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'windowing'))

import src.utils as utils


DEFAULT_N_JOBS = -1
DEFAULT_OUTPUT_FORMAT = 'gz'
DEFAULT_INPUT_FILE_CSV_SEPARATOR = ','
DEFAULT_FILTER_CSV_SEPARATOR = '|'
DEFAULT_CSV_SEPARATOR_SAVE = ','
DEFAULT_OUTPUT_FILENAME_AFFIX = 'cluster_'


def _get_ip_addresses_cluster_labels_and_indexes(df_ip_addresses, df_for_filtering, ip_col_name):
    orig_cluster_label_dtype = df_for_filtering['cluster_label'].dtype

    df_ip_addresses_with_clusters = df_ip_addresses.merge(
        df_for_filtering, left_on=ip_col_name, right_on='ip_address_obj', how='left').dropna()
    df_ip_addresses_with_clusters['cluster_label'] = (
        df_ip_addresses_with_clusters['cluster_label'].astype(orig_cluster_label_dtype))

    return df_ip_addresses_with_clusters


def _deduplicate_indexes_for_ip_addresses_with_clusters(df_ip_addresses_with_clusters):
    # Use the first IP address in case both source and destination IP address are present in the IP
    # filter list, even if the other IP address belongs to a different cluster.
    return df_ip_addresses_with_clusters.loc[
        ~df_ip_addresses_with_clusters.index.duplicated(keep='first')]


def main(raw_args: list) -> None:
    args = parse_args(raw_args)

    df_for_filtering = utils.load_df(args.filter_path, csv_sep=args.filter_csv_sep)
    df_for_filtering['ip_address_obj'] = df_for_filtering['ip_address'].apply(ipaddress.ip_address)

    if os.path.isfile(args.input_path):
        _split_file(
            args.input_path,
            args.input_file_csv_sep,
            df_for_filtering,
            args.output_path,
            args.output_affix,
            args.output_csv_sep,
            args.output_format,
        )
    elif os.path.isdir(args.input_path):
        os.makedirs(args.output_path, exist_ok=True)

        filepaths = []
        for root, dirnames, filenames in os.walk(args.input_path):
            for filename in filenames:
                filepaths.append(os.path.join(root, filename))

        joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(_split_file)(
                input_filepath,
                args.input_file_csv_sep,
                df_for_filtering,
                args.output_path,
                args.output_affix,
                args.output_csv_sep,
                args.output_format,
            )
            for input_filepath in filepaths
        )
    else:
        raise ValueError(f'Input path "{args.input_path}" must be a valid file or directory')


def _split_file(
        filepath: str, input_csv_sep: str,
        df_for_filtering,
        output_dirpath: str, output_affix: str, output_csv_sep: str, output_format):
    df = utils.load_df(filepath, csv_sep=input_csv_sep)

    df_ip_addresses_with_clusters_list = []
    for ip_col_name in ['IP_SRC', 'IP_DST']:
        df_ip_addresses = df[ip_col_name].apply(ipaddress.ip_address).to_frame()
        df_ip_addresses_with_clusters_list.append(_get_ip_addresses_cluster_labels_and_indexes(
            df_ip_addresses, df_for_filtering, ip_col_name))

    df_ip_addresses_with_clusters = pd.concat(df_ip_addresses_with_clusters_list)
    df_ip_addresses_with_clusters = _deduplicate_indexes_for_ip_addresses_with_clusters(
        df_ip_addresses_with_clusters)

    cluster_labels = df_ip_addresses_with_clusters['cluster_label'].unique()

    for cluster_label in cluster_labels:
        indexes = df_ip_addresses_with_clusters[
            df_ip_addresses_with_clusters['cluster_label'] == cluster_label].index

        output_filepath_root = os.path.join(
            output_dirpath,
            output_affix + str(cluster_label),
            os.path.splitext(os.path.basename(filepath))[0] + '_' + output_affix + str(cluster_label))
        os.makedirs(os.path.dirname(output_filepath_root), exist_ok=True)

        utils.save_df(
            df.iloc[indexes].reset_index(drop=True),
            output_filepath_root,
            csv_sep=output_csv_sep,
            file_format=output_format,
        )


def parse_args(raw_args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=('Splits each file in the specified directory containing features'
                     ' into separate files per cluster label, according to the source IP address.'))

    parser.add_argument(
        'input_path',
        help='Input path containing features. Can be either a single file or a directory.',
    )
    parser.add_argument(
        'filter_path',
        help=('File path containing IP addresses and clusters to split feature files by.'
              ' The file must contain the following columns: "ip_address", "cluster_label".'),
    )
    parser.add_argument(
        'output_path',
        help='Output directory path.',
    )
    parser.add_argument(
        '--output-affix',
        default=DEFAULT_OUTPUT_FILENAME_AFFIX,
        help=('Affix for filenames and directory names.'
              f' Default: {DEFAULT_OUTPUT_FILENAME_AFFIX}'),
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
        help=('Format of output files.'
              f' Default: {DEFAULT_OUTPUT_FORMAT}'),
    )
    parser.add_argument(
        '--input-file-csv-sep',
        default=DEFAULT_INPUT_FILE_CSV_SEPARATOR,
        help=('Separator for input files if their format is CSV.'
              f' Default: {DEFAULT_INPUT_FILE_CSV_SEPARATOR}'),
    )
    parser.add_argument(
        '--filter-csv-sep',
        default=DEFAULT_FILTER_CSV_SEPARATOR,
        help=('Separator for the file containing IP addresses and cluster label if the format is CSV.'
              f' Default: {DEFAULT_FILTER_CSV_SEPARATOR}'),
    )
    parser.add_argument(
        '--output-csv-sep',
        default=DEFAULT_CSV_SEPARATOR_SAVE,
        help=('Separator for output CSV files.'
              f' Default: {DEFAULT_CSV_SEPARATOR_SAVE}'),
    )

    return parser.parse_args(raw_args)


if __name__ == '__main__':
    main(sys.argv[1:])
