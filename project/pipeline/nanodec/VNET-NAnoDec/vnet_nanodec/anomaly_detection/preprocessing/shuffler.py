import argparse
import random
import os
import shutil
import string
import sys

import joblib

# Append to system path for imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)).rsplit('src', 1)[0])

import src.utils as utils


DEFAULT_CSV_SEPARATOR = ','
DEFAULT_N_JOBS = -1
DEFAULT_RANDOM_PREFIX_LENGTH = 24
DEFAULT_OUT_FILE_FORMAT = 'gz'


def main(raw_args: list) -> None:
    args = parse_args(raw_args)

    os.makedirs(args.output_dirpath, exist_ok=True)

    input_filepaths = [
        os.path.join(root, filename)
        for root, _, filenames in os.walk(args.input_dirpath) for filename in filenames]

    joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(_shuffle_and_rename_file)(
            filepath, args.output_dirpath, args.csv_sep, args.random_prefix_length, args.out_file_format,
        )
        for filepath in input_filepaths
    )

    if args.remove_input:
        shutil.rmtree(args.input_dirpath)


def _shuffle_and_rename_file(
        filepath: str, output_dirpath: str, csv_sep: str, random_prefix_length: int, out_file_format: str):
    df = utils.load_df(filepath, csv_sep=csv_sep)

    df = df.sample(frac=1).reset_index(drop=True)

    new_filename = _get_renamed_filepath(os.path.basename(filepath), random_prefix_length)
    filepath_root = os.path.splitext(os.path.join(output_dirpath, new_filename))[0]

    utils.save_df(df, filepath_root, out_file_format, csv_sep=csv_sep)


def _get_renamed_filepath(filename: str, random_prefix_length: int):
    # Taken from: https://stackoverflow.com/a/2257449
    return (
        ''.join(
            random.SystemRandom().choice(string.ascii_lowercase + string.digits)
            for _ in range(random_prefix_length))
        + '_'
        + filename
    )


def parse_args(raw_args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Shuffles rows within files and randomly renames the files in the specified directory.')
    )

    parser.add_argument(
        'input_dirpath',
        help='Directory path containing input files.',
    )
    parser.add_argument(
        'output_dirpath',
        help='Output directory path containing shuffled and renamed files.',
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=DEFAULT_N_JOBS,
        help=('Number of jobs to run in parallel.'
              f' Default: {DEFAULT_N_JOBS}'),
    )
    parser.add_argument(
        '--csv-sep',
        default=DEFAULT_CSV_SEPARATOR,
        help=('Separator to use for input and output files if the file format is CSV or gzip.'
              f' Default: {DEFAULT_CSV_SEPARATOR}'),
    )
    parser.add_argument(
        '--random-prefix-length',
        default=DEFAULT_RANDOM_PREFIX_LENGTH,
        help=('Length of a random string to prepend to each file.'
              f' Default: {DEFAULT_RANDOM_PREFIX_LENGTH}'),
    )
    parser.add_argument(
        '--remove-input',
        action='store_true',
        help='If specified, remove the input directory upon completion.',
    )
    parser.add_argument(
        '--out-file-format',
        default=DEFAULT_OUT_FILE_FORMAT,
        choices=utils.VALID_OUTPUT_FILE_FORMATS,
        help=('File extension (and file format) of the output files.'
              f' Default: {DEFAULT_OUT_FILE_FORMAT}'),
    )

    return parser.parse_args(raw_args)


if __name__ == '__main__':
    main(sys.argv[1:])
