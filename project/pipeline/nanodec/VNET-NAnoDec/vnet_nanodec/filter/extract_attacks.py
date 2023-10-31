import argparse
import os
import sys

import pyarrow as pa
import pyarrow.csv as csv

# Append absolute path to of the 'src' directory to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)).rsplit('src', 1)[0])

import src.utils as utils


DEFAULT_INPUT_CSV_SEPARATOR = '|'
DEFAULT_OUTPUT_CSV_SEPARATOR = ','
DEFAULT_NUM_SAMPLES_PER_LABEL = 1000
DEFAULT_RANDOM_STATE = -1

CHUNK_SIZE = 500000
OUTPUT_FILE_EXTENSION = 'gz'


def main(raw_args: list) -> None:
    args = parse_args(raw_args)

    if args.random_state == -1:
        args.random_state = None

    flow_filepaths = [
        os.path.join(root, filename)
        for root, _, filenames in os.walk(args.input_filepath)
        for filename in filenames
    ]

    if not args.output_filepath.endswith('.' + OUTPUT_FILE_EXTENSION):
        args.output_filepath += '.' + OUTPUT_FILE_EXTENSION

    header = True
    with pa.CompressedOutputStream(args.output_filepath, 'gzip') as out:
        for flow_filepath in flow_filepaths:
            df = utils.load_df(flow_filepath, csv_sep=args.input_csv_sep)

            num_chunks = (len(df) // CHUNK_SIZE) + 1
            start_ranges = list(i * CHUNK_SIZE for i in range(num_chunks))
            end_ranges = start_ranges[1:] + [None]

            for start_range, end_range in zip(start_ranges, end_ranges):
                chunk = df.iloc[start_range:end_range]
                for label in chunk['Label'].unique():
                    temp = chunk[chunk['Label'] == label]
                    if args.sample_size != -1:
                        temp = temp.sample(
                            min(args.sample_size, len(chunk[chunk['Label'] == label])),
                            random_state=args.random_state)
                    csv.write_csv(pa.Table.from_pandas(temp, preserve_index=False), out,
                                  csv.WriteOptions(delimiter=args.output_csv_sep, include_header=header))
                    header = False


def parse_args(raw_args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extracts known attacks from network flows.')

    parser.add_argument(
        'input_filepath',
        help='Input directory containing network flow files.',
    )
    parser.add_argument(
        'output_filepath',
        help='File path containing extracted attacks.',
    )
    parser.add_argument(
        '--input-csv-sep',
        default=DEFAULT_INPUT_CSV_SEPARATOR,
        help=('Separator to use for input files if the file format is CSV or gzip.'
              f' Default: {DEFAULT_INPUT_CSV_SEPARATOR}'),
    )
    parser.add_argument(
        '--output-csv-sep',
        default=DEFAULT_OUTPUT_CSV_SEPARATOR,
        help=('Separator to use for output files if the file format is CSV or gzip.'
              f' Default: {DEFAULT_OUTPUT_CSV_SEPARATOR}'),
    )
    parser.add_argument(
        '--sample-size',
        default=DEFAULT_NUM_SAMPLES_PER_LABEL,
        help=('Number of samples per label to obtain from each flow file. Use -1 to disable sampling.'
              f' Default: {DEFAULT_NUM_SAMPLES_PER_LABEL}'),
    )
    parser.add_argument(
        '--random-state',
        default=DEFAULT_RANDOM_STATE,
        help=('Fixed random state to use for sampling. Use -1 to not use a fixed random state.'
              f' Default: {DEFAULT_RANDOM_STATE}'),
    )

    return parser.parse_args(raw_args)


if __name__ == '__main__':
    main(sys.argv[1:])
