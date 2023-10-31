"""
Splitting a network flow file into multiple smaller files.

Example:
    python splitter.py flows.csv split --n-files 35
        - Splits the `flows.csv` file into 35 files and saves them in the `split` directory.
"""

import argparse
import os
import sys
import pandas as pd

# Append absolute path to of the 'src' directory to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)).rsplit('src', 1)[0])

import vnet_nanodec.utils as utils
import vnet_nanodec.defines as defines


DEFAULT_N_FILES = 16
DEFAULT_OUT_FILE_FORMAT = 'csv'
DEFAULT_OUT_CSV_SEP = '|'


def main(raw_args: list) -> None:
    args = parse_args(raw_args)

    split_file(args.flows_filepath, args.n_files, args.output_dir, args.out_file_format)


def split_file(filepath: str, n_files: int, output_dirpath: str, out_file_format: str) -> list:
    """Splits the flow file defined by filepath into n_files files and saving into the
    output_dirpath folder. The output format can be specified by the out_file_format parameter.

    Parameters:
        filepath        -- path to the file to split
        n_files         -- number of files to split the file into
        output_dirpath  -- Path to the directory to save splitted files into
        out_file_format -- Output format of the splitted files

    Returns:
        list -- Pathnames of splitted files
    """
    filenames = []      # List of produced filenames
    data = utils.load_df(filepath, dtypes=defines.FEATURES_CASTDICT)

    # Assign the worker process number
    data['worker'] = pd.util.hash_array(data['IPV4_SRC_ADDR'].to_numpy()).astype('uint32') % n_files
    data.loc[(data.IPV4_SRC_ADDR == '0.0.0.0'), 'worker'] = pd.util.hash_array(
        data.loc[(data.IPV4_SRC_ADDR == '0.0.0.0'), 'IPV6_SRC_ADDR'].to_numpy()).astype('uint32') % n_files

    filename_root = os.path.splitext(os.path.basename(filepath))[0]

    # Save the data for each worker into separate files
    for idx in range(n_files):
        data_part = data[data['worker'] == idx].drop(columns='worker').reset_index(drop=True)
        filepath_root = os.path.join(output_dirpath, filename_root + '_' + str(idx))
        saved_file = utils.save_df(data_part, filepath_root, out_file_format, DEFAULT_OUT_CSV_SEP)

        filenames.append(saved_file)

    return filenames


def parse_args(raw_args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=('Splits the input flow file to multiple files according to source IP address'
                     ' (the same source IP address is in the same output file).'))

    parser.add_argument(
        'flows_filepath',
        help='File path containing flows to process.',
    )
    parser.add_argument(
        'output_dir',
        help='Directory path to save split files to.',
    )
    parser.add_argument(
        '--n-files',
        type=int,
        required=True,
        help=('Number of files to split the input file to.'
              f' Default: {DEFAULT_N_FILES}'),
    )
    parser.add_argument(
        '--out-file-format',
        default=DEFAULT_OUT_FILE_FORMAT,
        choices=utils.VALID_OUTPUT_FILE_FORMATS,
        help=('File extension (and file format) of the output (preprocessed) file.'
              f' Default: {DEFAULT_OUT_FILE_FORMAT}'),
    )

    return parser.parse_args(raw_args)


if __name__ == '__main__':
    main(sys.argv[1:])
