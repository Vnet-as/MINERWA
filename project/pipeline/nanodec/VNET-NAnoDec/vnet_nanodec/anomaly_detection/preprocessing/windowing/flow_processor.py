"""
Flow processing engine.
"""

import numpy as np
import math
import pandas as pd
from typing import Optional

from windower import Windower


# Set of functions for timestamps conversion
def seconds_to_milliseconds(sec: float) -> int:
    return int(sec * 1000)


def process_flows(flows: pd.DataFrame, features_extract, windower_settings: dict,
    window_size: int, windowing_context : Optional[tuple] = None, *args, **kwargs
    ) -> pd.DataFrame:
    """Processes flows from a given dataframe in a simulated online mode in a
    single-process manner.

    Parameters:
        features_extract     -- function for flow feature extraction
        flows                -- DataFrame containing flow statistics
        windower_settings    -- settings for the windower to use for processing
        window_size          -- window size to use in seconds
        windowing_context    -- context to continue windowing from the previous file
        args, kwargs         -- eat other arguments not used in the serial program

    Returns:
        pd.DataFrame -- DataFrame of the same size as input flow DataFrame
        tuple        -- windowing context as a 2-tuple containing windower instance with
                        its internal settings and time of the next window start
    """

    window_size     = seconds_to_milliseconds(window_size)
    tstamp_next_win = None          # Next timestamp for the window start
    windower        = None          # Windower instance to perform statistics logging

    # Unpack the windowing context tuple and assign to variables
    if windowing_context is not None:
        windower, tstamp_next_win = windowing_context

    if windower is None:
        windower = Windower(**windower_settings)

    # Make an empty call to Windower stats retrieval routine to get stats datatype
    stats_sample = windower.retrieve_stats('')

    # Preallocate statistics array with given datatype to optimize processing
    computed_stats = np.zeros(len(flows), dtype=stats_sample.dtype)

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

            windower.end_window(win_delta)

        # Log the flow into Windower
        windower.log(flow_features)

        # Retrieve statistics for the current flow
        flow_window_stats = windower.retrieve_stats(flow_features.ip_src)
        computed_stats[idx] = flow_window_stats

    # Build up windowing context
    new_windowing_context = (windower, tstamp_next_win)

    return pd.DataFrame(computed_stats).drop(columns='window_id'), new_windowing_context
