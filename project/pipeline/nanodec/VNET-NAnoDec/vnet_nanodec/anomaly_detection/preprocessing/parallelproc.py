"""
Preprocessing of network flow files in parallel.
Due to its parallel nature, the final dataframe is NOT in the order of the original input flow
file. Nevertheless, this should not have any consequences, as after windowing, temporal context
is already incorporated into the data, and the data points itself become standalone from that
point on.  If other models (such as recurrent NNs), requiring temporal dependencies between flows
to be kept, are used, the final dataframe should be sorted by ending/starting flow timestamp
before exporting to the file.
"""

import argparse
import multiprocessing as mp
import os
import pandas as pd
import pathlib
import shutil
import sys
import tempfile
from typing import Optional
from tqdm import tqdm

# Append absolute path to of the 'src' directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)) + 'windowing'))

from . import data_preparer
from . import splitter
import vnet_nanodec.utils as utils
import vnet_nanodec.defines as defines


# Default program settings
DEFAULT_SCALING_CFG_FILEPATH = None
DEFAULT_SCALING_MINMAX_MAX = None
DEFAULT_N_JOBS = splitter.DEFAULT_N_FILES
DEFAULT_OUTPUT_DIRPATH = 'preproc'
DEFAULT_TEMPDIR = '/tmp'
DEFAULT_WINDOWERS_DIRNAME  = 'wincontexts'

DEFAULT_OUTPUT_SUFFIX = data_preparer.DEFAULT_OUTPUT_SUFFIX
DEFAULT_OUT_FILE_FORMAT = 'csv'
WINDOWERS_FILENAME = 'wincontext-'
DEFAULT_IP_FILTER_FILEPATH = data_preparer.DEFAULT_IP_FILTER_FILEPATH

# Multiprocessing synchronization
SYNC_QUEUE_SIZE = 5
SIGNAL_SPLITTING_FINISHED = '__FINISHED'


def main(raw_args: list) -> None:
    args = parse_args(raw_args)

    # Transform arguments to variables for better readability
    in_flows_dirpath   = args.flows_dir          # Input flows directory
    in_flows_file_ext  = args.flows_file_ext     # Input flows file extension
    windowing_config   = args.windowing_config   # Windowing settings
    out_dirpath        = args.out_dir            # Output directory to save processed data to
    out_file_suffix    = args.suffix             # Output preprocessed files suffix
    out_file_format    = args.out_file_format    # Output file format
    scaling_conf_path  = args.scaler_config      # Configuration file path for the scaler
    scaling_minmax_max = args.scaling_minmax_max # New maximum percentile for MinMax scaling
    ip_filter_path     = args.ip_filter          # Path to IP address filter to include/ignore
    n_jobs             = args.n_jobs             # Number of processes to process flows in
    remove_wincontexts = args.remove_wincontexts # Remove windowing contexts after finish
    do_not_merge_files = args.do_not_merge       # Do not merge files after parallel processing
    temp_dirpath       = args.temp_dir           # Temporary directory for windowing contexts

    sync_timestamp = None           # Synchronization timestamp for windowing

    # Determine directory to save windower contexts to
    windowers_dir = (os.path.join(temp_dirpath, DEFAULT_WINDOWERS_DIRNAME) if temp_dirpath is
      not None else os.path.join(DEFAULT_TEMPDIR, DEFAULT_WINDOWERS_DIRNAME))

    # Producer consumer queue for processes synchronization
    # -1, of the specified size because files are actually created before added to queue
    dirs_ready = mp.Queue(maxsize=SYNC_QUEUE_SIZE - 1 if SYNC_QUEUE_SIZE != 1 else 1)

    # Obtain list of all files in a given directory that will be processed
    flow_files = [
        os.path.join(root, filename)
        for root, _, filenames in os.walk(in_flows_dirpath)
        for filename in filenames if filename.endswith(in_flows_file_ext)
    ]
    flow_files.sort()

    # Create a folder to save the preprocessed flows to if it does not exist
    pathlib.Path(out_dirpath).mkdir(parents=True, exist_ok=True)

    # Create a folder to store windower contexts in
    pathlib.Path(windowers_dir).mkdir(parents=True, exist_ok=True)

    # Obtain the very first flow ending timestamp from the file for windowing synchronization
    sync_timestamp = get_first_timestamp(os.path.join(in_flows_dirpath, flow_files[0]))

    # Perform splitting and parallel processing within a temporary directory
    with tempfile.TemporaryDirectory(dir=temp_dirpath) as tmpdir:
        # Run parallel process for file splitting per IP address hash
        file_splitting_proc = mp.Process(
            target=_split_by_ips,
            kwargs={
                'tasks_queue': dirs_ready,
                'op_dirpath': tmpdir,
                'in_dirpath': in_flows_dirpath,
                'in_files': flow_files,
                'n_splits': n_jobs,
            }
        )
        file_splitting_proc.start()

        # Run parallel process for file preprocessing and windowing
        flows_processor_proc = mp.Process(
            target=_process_flows,
            kwargs={
                'tasks_queue': dirs_ready,
                'op_dirpath': tmpdir,
                'windowing_config_filepath': windowing_config,
                'ip_filter_filepath': ip_filter_path,
                'scaling_conf_filepath': scaling_conf_path,
                'scaling_minmax_max' : scaling_minmax_max,
                'in_files_total': len(flow_files),
                'out_flows_dirpath': out_dirpath,
                'out_file_format': out_file_format,
                'out_file_suffix': out_file_suffix,
                'sync_timestamp': sync_timestamp,
                'merge_outputs': not do_not_merge_files,
                'windowers_dir': windowers_dir,
            }
        )
        flows_processor_proc.start()

        # Wait for the processes to finish
        file_splitting_proc.join()
        flows_processor_proc.join()

    # Remove windowing contexts if desired
    if remove_wincontexts:
        [filename.unlink() for filename in pathlib.Path(windowers_dir).iterdir()]


def _split_by_ips(op_dirpath: str, in_dirpath: str, in_files: list, n_splits: int,
    tasks_queue: mp.Queue) -> None:
    """Splits all in_files into n_jobs parts and saves with specified output format.
    Each split file is saved into separate directory, which is appended to tasks_queue
    queue to signalize the process.

    Parameters:
        op_dirpath      -- Operational directory to save split files into
        in_dirpath      -- Input directory that contains file specified within in_files
        in_files        -- List of files to split
        n_splits        -- Number of splits to make
        tasks_queue     -- Queue for tasks synchronization
    """

    for flow_file in in_files:
        # Create a directory for a given file to save its split parts into and clean it
        filename_noext     = flow_file.split('.')[0]
        filename_split_dir = os.path.join(op_dirpath, filename_noext)

        pathlib.Path(filename_split_dir).mkdir(exist_ok=True)
        [filename.unlink() for filename in pathlib.Path(filename_split_dir).iterdir()]

        # Perform file splitting and save to the desired directory
        # Make splits into CSVs to conserve computing costs, they will be deleted later anyway
        splitter.split_file(os.path.join(in_dirpath, flow_file), n_splits,
            filename_split_dir, 'csv')

        # Add directory path to the queue to signalize data are split
        tasks_queue.put(filename_split_dir)

    # Add a constant to signalize that processing is finished
    tasks_queue.put(SIGNAL_SPLITTING_FINISHED)


def _process_flows(tasks_queue: mp.Queue, op_dirpath: str, windowing_config_filepath: str,
    ip_filter_filepath: str, scaling_conf_filepath: Optional[str],
    scaling_minmax_max: Optional[float], in_files_total: int, out_flows_dirpath: str,
    out_file_format: str, out_file_suffix: str, sync_timestamp: Optional[int],
    merge_outputs: bool, windowers_dir: str) -> None:
    """Processes flow files by firstly splitting them, and then processing in parallel by running
    multiple data_preparer.py instances for each split file. Files are split by IP addresses,
    allowing windowing context to be passed across the computation. Finally, preprocessed files
    are merged back together.

    Parameters:
        tasks_queue           -- Queue for tasks synchronization
        op_dirpath            -- Path to the operational directory, where to perform operations in
        flow_files            -- Path to the flow file to be processed
        windowing_config_filepath -- Path to the windowing configuration file
        ip_filter_filepath    -- Path to the IP address filter file
        scaling_conf_filepath -- Path to the scaling configuration file
        scaling_minmax_max    -- New percentile for the MinMax scaling value
        in_files_total        -- The total number of processed files for progress keeping
        out_flows_dirpath     -- Path to the output directory to save the processed file to
        out_file_format       -- Processed file output format
        out_file_suffix       -- Processed file suffix
        sync_timestamp        -- Synchronization timestamp for parallel processes without context
        do_not_merge          -- Do not merge files after preprocessing
        windowers_dir         -- Directory in which windowing contexts are stored
    """
    flows_folder = tasks_queue.get()
    progress_bar = tqdm(total=in_files_total, unit='file')
    out_file_format_partial = 'csv'     # Use CSVs for computing efficiency for partial results

    # If no merging happens, save partial files straight into output dir with the specified format
    if not merge_outputs:
        op_dirpath = out_flows_dirpath
        out_file_format_partial = out_file_format

    # Loop through contents of the tasks queue until a terminating string is received
    while flows_folder != SIGNAL_SPLITTING_FINISHED:
        workers = []
        flows_basefilename = os.path.basename(flows_folder)

        progress_bar.set_description(f'{flows_basefilename}')

        # Run a separate data preprocessing + windowing process for each split file
        for flows_splitfile in os.listdir(flows_folder):
            # Determine the file index to allow windower contexts passing
            file_index = os.path.splitext(flows_splitfile)[0].split('_')[-1]
            windower_contexts_filename = os.path.join(windowers_dir,
                f'{WINDOWERS_FILENAME}{file_index}.bin')

            # Spawn and run the process
            process = mp.Process(
                target=data_preparer.data_prepare,
                kwargs={
                    'src_flows_filepath': os.path.join(flows_folder, flows_splitfile),
                    'windowing_config_filepath': windowing_config_filepath,
                    'preproc_data_suffix': out_file_suffix,
                    'out_file_format': out_file_format_partial,
                    'out_dir': op_dirpath,
                    'ip_filter_filepath': ip_filter_filepath,
                    'n_jobs': 0,    # Always 0, so we do not spawn any more processes inside
                    'scaling_config_filepath' : scaling_conf_filepath,
                    'scaling_minmax_new_max' : scaling_minmax_max,
                    'windowing_context_in_filepath'  : windower_contexts_filename,
                    'windowing_context_out_filepath' : windower_contexts_filename,
                    'sync_timestamp' : sync_timestamp
                },
            )
            process.start()
            workers.append(process)

        for worker in workers:
            worker.join()

        # Perform merging from the temporary directory to the final output dir
        if merge_outputs:
            processed_flow_files = [preproc_file for preproc_file in os.listdir(op_dirpath) if
                out_file_suffix in preproc_file]
            final_flow_df = pd.concat([
                utils.load_df(os.path.join(op_dirpath, processed_flow_file),
                    csv_sep=utils.DEFAULT_CSV_SEPARATOR_SAVE)
                for processed_flow_file in processed_flow_files])

            final_filename = flows_basefilename + out_file_suffix
            final_filepath_root = os.path.join(out_flows_dirpath, final_filename)

            # Shuffle the DataFrame to minimize bias during training and save
            final_flow_df = final_flow_df.sample(frac=1).reset_index(drop=True)

            utils.save_df(final_flow_df, final_filepath_root, out_file_format)

            # Clear the temporary directory to conserve space
            [filename.unlink() for filename in pathlib.Path(op_dirpath).iterdir() if filename.is_file()]

        # Remove the temporary directory with split data to conserve space
        shutil.rmtree(flows_folder)

        progress_bar.update(1)

        # Retrieve new folder to process from the queue
        flows_folder = tasks_queue.get()

    progress_bar.close()


def get_first_timestamp(flowfile_path : str) -> int:
    """Obtain the first flow timestamp of the specified DataFrame

    Parameters:
        flowfile_path -- Path to the flow file to obtain the first timestamp from

    Return:
        First COLNAME_FLOW_END_TSTAMP value of the specified DataFrame flowfile_path"""

    flow_data = utils.load_df(flowfile_path, dtypes=defines.FEATURES_CASTDICT)
    flow_data = flow_data.sort_values(by=defines.COLNAME_FLOW_END_TSTAMP,
        ignore_index=True)

    return flow_data.iloc[0][defines.COLNAME_FLOW_END_TSTAMP]


def parse_args(raw_args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Preprocesses network flows within the specified directory in parallel.')

    parser.add_argument(
        '--flows-dir',
        required=True,
        help='Directory path containing flows to process in parallel.',
    )
    parser.add_argument(
        '--flows-file-ext',
        required=True,
        help='File extension of input flow files. '
        'Options: \{gz, parquet, feather, pkl, pkl-zip, csv, flows\}',
    )
    parser.add_argument(
        '--windowing-config',
        required=True,
        help='Path to the windowing YAML configuration file.'
    )
    parser.add_argument(
        '--ip-filter',
        default=DEFAULT_IP_FILTER_FILEPATH,
        help=('Path to file containing a list of IP addresses to filter flows by.'
              ' This is assumed to be a CSV file with a header and one column.'
              ' Flows with source of destination IP address not matching any IP address in the list'
              ' will be removed.'),
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=DEFAULT_N_JOBS,
        help=('Number of worker processes to use.'
              f' Default: {DEFAULT_N_JOBS}'),
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIRPATH,
        help=('Directory path to write output to.'
              f' Default: {DEFAULT_OUTPUT_DIRPATH}'),
    )
    parser.add_argument(
        '--out-file-format',
        default=DEFAULT_OUT_FILE_FORMAT,
        choices=utils.VALID_OUTPUT_FILE_FORMATS,
        help=('File extension (and file format) of the output (preprocessed) files.'
              f' Default: {DEFAULT_OUT_FILE_FORMAT}'),
    )
    parser.add_argument(
        '--scaler-config',
        default=DEFAULT_SCALING_CFG_FILEPATH,
        help='Path to load the scaling config from. If not provided, no scaling is performed.',
    )
    parser.add_argument(
        '--scaling-minmax-max',
        default=DEFAULT_SCALING_MINMAX_MAX,
        help='New maximum value for the minmax scaling.',
    )
    parser.add_argument(
        '--suffix',
        default=DEFAULT_OUTPUT_SUFFIX,
        help=('Suffix for filenames containing preprocessed data.'
              f' Default: {DEFAULT_OUTPUT_SUFFIX}'),
    )
    parser.add_argument(
        '--remove-wincontexts',
        action='store_true',
        help="Remove windowing contexts after the program finishes"
    )
    parser.add_argument(
        '--do-not-merge',
        action='store_true',
        help="Do not merge preprocessed files after parallel processing into a single one."
    )
    parser.add_argument(
        '--temp-dir',
        help=('Temporary directory to store windowing contexts in.'
              f' Default: {DEFAULT_TEMPDIR}')
    )

    return parser.parse_args(raw_args)


if __name__ == '__main__':
    main(sys.argv[1:])
