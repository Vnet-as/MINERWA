import argparse
import os
import sys

import pandas as pd

# Append absolute path to of the 'src' directory to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)).rsplit('src', 1)[0])

import src.utils as utils


DEFAULT_CSV_SEPARATOR = ','
DEFAULT_SUBSAMPLE_SIZE = 5000

CHUNK_SIZE = 500000
COLUMNS_TO_BE_REMOVED = [
    'IP_SRC', 'IPV4_SRC_ADDR', 'IPV6_SRC_ADDR',
    'IP_DST', 'IPV4_DST_ADDR', 'IPV6_DST_ADDR',
    'SRC_DENY', 'DST_DENY']


def main(raw_args: list) -> None:
    args = parse_args(raw_args)

    data = pd.DataFrame()

    df = utils.load_df(args.input_filepath, csv_sep=args.csv_sep)

    num_chunks = (len(df) // CHUNK_SIZE) + 1
    start_ranges = list(i * CHUNK_SIZE for i in range(num_chunks))
    end_ranges = start_ranges[1:] + [None]

    for start_range, end_range in zip(start_ranges, end_ranges):
        chunk = df.iloc[start_range:end_range]
        for label in chunk['Label'].unique():
            temp = chunk[chunk['Label'] == label].sample(
                min(DEFAULT_SUBSAMPLE_SIZE, len(chunk[chunk['Label'] == label])), random_state=0)
            data = pd.concat([data, temp], ignore_index=True, copy=False)

    data.drop(columns=COLUMNS_TO_BE_REMOVED, inplace=True, errors='ignore')

    print('Counts per label before balancing:')
    print(data['Label'].value_counts())

    # balance data in binary manner (background vs attacks)
    attacks = data[~data['Label'].str.contains('background')].groupby(['Label']).sample(
        DEFAULT_SUBSAMPLE_SIZE, random_state=0, replace=True).sample(frac=1, random_state=0).reset_index(drop=True)

    background = data[data['Label'].str.contains('background')].sample(
        len(attacks), random_state=0, replace=True).sample(frac=1, random_state=0).reset_index(drop=True)

    data = pd.concat([attacks, background], ignore_index=True, copy=False).sample(
        frac=1, random_state=0).reset_index(drop=True)

    data.loc[data['Label'].str.contains('background'), 'Label'] = 0
    data.loc[data['Label'] != 0, 'Label'] = 1

    print('Counts per label after balancing:')
    print(data['Label'].value_counts())

    utils.save_df(data, args.output_filepath, csv_sep=args.csv_sep)


def parse_args(raw_args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=('Balances attack data for training in a binary manner (the same amount of'
                     ' attacks vs. background, with balanced inclusion of attacks).'))

    parser.add_argument(
        'input_filepath',
        help='Input file path containing known attacks.',
    )
    parser.add_argument(
        'output_filepath',
        help='Output file path containing balanced known attacks.',
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
