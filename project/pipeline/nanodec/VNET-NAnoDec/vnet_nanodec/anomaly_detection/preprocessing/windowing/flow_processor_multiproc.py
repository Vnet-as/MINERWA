"""
Flow processing engine.
"""

import math
import multiprocessing
import numpy as np
import pandas as pd
from typing import Optional

from .windower import Windower


def process_func(windower_instance: Optional[Windower],
    windower_settings: dict, data_queue: multiprocessing.Queue, connection) -> None:
    """Function for worker thread to log statistics into the window and produce outputs

    Parameters:
        windower_settings -- Dictionary to use when creating windower
        data_queue        -- Queue for receiving work from the main thread
        connection        -- Connection pipe to send computed data back after computing

    Returns:
        None - Indirectly returns computed np.ndarray through the connection pipe"""

    windower  = Windower(**windower_settings) if windower_instance is None else \
        windower_instance
    computed_stats = []

    for command, data in iter(data_queue.get, 'STOP'):
        if command == 'END_WINDOW':
            windower.end_window(data)
        elif command == 'LOG':
            windower.log(data)
            computed_stats.append(windower.retrieve_stats(data.ip_src))

    connection.send(np.vstack(computed_stats))
    connection.send(windower)


# Set of functions for timestamps conversion
def seconds_to_milliseconds(sec: float) -> int:
    return int(sec * 1000)


def process_flows(flows: pd.DataFrame, features_extract, windower_settings: dict,
    window_size: int, windowing_context : Optional[tuple] = None,
    worker_processes_num: int = 6) -> pd.DataFrame:
    """Processes flows from a given dataframe in a simulated online mode in a
    multi-process manner with a specified number of worker processes.

    Parameters:
        features_extract     -- function for flow feature extraction
        flows                -- DataFrame containing flow statistics
        windower_settings    -- settings for the windower to use for processing
        window_size          -- window size to use in seconds
        windowing_context    -- context to continue windowing from the previous file
        worker_processes_num -- number of worker processes

    Returns:
        pd.DataFrame -- DataFrame of the same size as input flow DataFrame
    """

    window_size = seconds_to_milliseconds(window_size)
    tstamp_next_win = None          # Next timestamp for window start
    windowers       = None          # Windower instance to perform statistics logging

    # Unpack the windowing context tuple and assign to variables
    if windowing_context is not None:
        windowers, tstamp_next_win = windowing_context

    if windowers is None:
        windowers = [None for _ in range(worker_processes_num)]

    # Make an empty call to Windower stats retrieval routine to get stats datatype
    stats_sample = Windower(**windower_settings).retrieve_stats('')

    # Preallocate statistics array with given datatype to optimize processing
    computed_stats = np.zeros(len(flows), dtype=stats_sample.dtype)

    queues      = [multiprocessing.Queue() for _ in range(worker_processes_num)]
    connections = [multiprocessing.Pipe() for _ in range(worker_processes_num)]
    processes   = [multiprocessing.Process(target=process_func, args=(windowers[idx],
         windower_settings, queues[idx], connections[idx][1])) for idx in
        range(worker_processes_num)]
    indices     = [[] for _ in range(worker_processes_num)]

    for process in processes:
        process.start()

    # Iterate through all rows in the dataframe containing flows
    for idx, flow in flows.iterrows():
        # Extract flow features
        flow_features = features_extract(flow)

        # Perform windowing
        flow_end = flow_features.tstamp_end

        if tstamp_next_win is None:
            tstamp_next_win = flow_end + window_size
        elif tstamp_next_win < flow_end:
            win_delta = int(math.ceil((flow_end - tstamp_next_win) / window_size))
            tstamp_next_win += win_delta * window_size

            # Put a window end signalization to the process queues
            for queue in queues:
                queue.put(('END_WINDOW', win_delta))

        # Log the flow into Windower based on source-IP hash
        proc_idx = hash(flow_features.ip_src) % worker_processes_num

        queues[proc_idx].put(('LOG', flow_features))
        indices[proc_idx].append(idx)

    # Tell all the processes to stop computing
    for queue in queues:
        queue.put('STOP')

    # Reassemble the array with the data computed by worker processes
    for idx in range(worker_processes_num):
        # Receive worker data
        worker_result_data = connections[idx][0].recv()
        worker_windower_context = connections[idx][0].recv()

        # Update main process variables
        np.put(computed_stats, indices[idx], worker_result_data)
        windowers[idx] = worker_windower_context

    # Wait for processes to finish
    for process in processes:
        process.join()

    # Build up windowing context
    new_windowing_context = (windowers, tstamp_next_win)

    return pd.DataFrame(computed_stats).drop(columns='window_id'), new_windowing_context
