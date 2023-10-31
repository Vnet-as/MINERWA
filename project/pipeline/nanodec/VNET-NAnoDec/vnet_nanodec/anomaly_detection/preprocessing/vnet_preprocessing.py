"""
VNET dataset preprocessing algorithms.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from . import scaler
from .windowing import traffic_stats as winstats
import vnet_nanodec.defines as defines


###############################################################################
####################     Constants definitions     ############################
###############################################################################
# Features not directly used for classification - provide additional information, but drop them
FEATURES_DROP = ['IPV4_SRC_ADDR', 'IPV6_SRC_ADDR',  # IPs are not directly useful, only as metadata
    'IPV4_DST_ADDR', 'IPV6_DST_ADDR',               # IPs are not directly useful, only as metadata
    'DIRECTION',                                    # Not useful for intrusion detection purposes
    'FLOW_VERDICT',                                 # These are all 0s
    'APPLICATION_ID',                               # Interesting metdata, but we cannot find an use for ID
    'window_total',                                 # Total number of windows used for window summary
    'window_span'                                   # Span of the summarized windows
]

# Features which were processed in the preprocessing function and are no longer needed for classification
FEATURES_DROP_PROCESSED = ['TCP_FLAGS', 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'CLIENT_TCP_FLAGS',
    'SERVER_TCP_FLAGS']

# Features created during the preprocessing stage
FEATURES_DROP_AUXILLIARY = ['IN_DUR_S', 'OUT_DUR_S']

# Numerical variables which should be scaled
FEATURES_NUMERICAL = ['IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
    'MIN_TTL', 'MAX_TTL', 'SRC_FRAGMENTS', 'DST_FRAGMENTS', 'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
    'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES',
    'NUM_PKTS_1024_TO_1514_BYTES', 'NUM_PKTS_OVER_1514_BYTES', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
    'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_PKTS', 'OOORDER_IN_PKTS', 'OOORDER_OUT_PKTS', 'DURATION_IN',
    'DURATION_OUT', 'TCP_WIN_MIN_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_MSS_IN', 'TCP_WIN_SCALE_IN', 'TCP_WIN_MIN_OUT',
    'TCP_WIN_MAX_OUT', 'TCP_WIN_MSS_OUT', 'TCP_WIN_SCALE_OUT', 'SRC_TO_DST_IAT_MIN', 'SRC_TO_DST_IAT_MAX',
    'SRC_TO_DST_IAT_AVG', 'SRC_TO_DST_IAT_STDDEV', 'DST_TO_SRC_IAT_MIN', 'DST_TO_SRC_IAT_MAX', 'DST_TO_SRC_IAT_AVG',
    'DST_TO_SRC_IAT_STDDEV']

# Numerical variables to be scaled computed during the preprocessing phase
FEATURES_NUMERICAL_COMPUTED = ['IN_BPS', 'IN_BPP', 'IN_PPS', 'OUT_BPS', 'OUT_BPP', 'OUT_PPS']

# Windower features to be scaled as well except the ones we are dropping
FEATURES_NUMERICAL_WINDOWER = list(winstats.STATS_NPDTYPE_WINDOW_COMPLETED.names) + \
    list(winstats.STATS_NPDTYPE_INTERWINDOW.names)
FEATURES_NUMERICAL_WINDOWER = [feature for feature in FEATURES_NUMERICAL_WINDOWER if feature
    not in ['window_id', 'window_total', 'window_span']]

# Categorical features and their possible values for manual one-hot encoding
FEATURES_CATEGORICAL = ['PROTOCOL', 'L4_SRC_PORT', 'L4_DST_PORT', 'ICMP_TYPE']

FEATURE_PROTOCOL_VALUES = ['ICMP', 'TCP', 'UDP', 'GRE', 'ESP', 'ICMP6']
FEATURE_L4_PORT_VALUES = ['OTHER', 'NOPORT', 'WEB_NOENC', 'TLS', 'DNS', 'EMAIL', 'VPN', 'DATA', 'SHELL',
    'PLAYSTATION', 'CHAT', 'QUERY', 'DYNAMIC']
FEATURE_ICMP_CODE_VALUES = ['OTHER', 'ECHO_REPLY', 'DEST_UNREACHABLE', 'REDIRECT', 'ECHO_REQUEST',
    'TIME_EXCEEDED']

# Dictionary to map all categorical features to their possible values
FEATURES_CATEGORICAL_VALUES_MAPPER = {
    'PROTOCOL'    : FEATURE_PROTOCOL_VALUES,
    'L4_SRC_PORT' : FEATURE_L4_PORT_VALUES,
    'L4_DST_PORT' : FEATURE_L4_PORT_VALUES,
    'ICMP_TYPE'   : FEATURE_ICMP_CODE_VALUES,
}

# Protocol to text data default dictionary
PROTO_MAPPER = defaultdict(lambda : 'OTHER', {
    1   : 'ICMP',
    6   : 'TCP',
    17  : 'UDP',
    47  : 'GRE',
    50  : 'ESP',
    58  : 'ICMP6'
})

# ICMP type to textual representation simplification
# Other possible would include 11 : TIME_EXCEEDED
ICMP_CODE_MAPPER = defaultdict(lambda : 'OTHER', {
    0   : 'ECHO_REPLY',
    3   : 'DEST_UNREACHABLE',
    5   : 'REDIRECT',
    8   : 'ECHO_REQUEST',
    11  : 'TIME_EXCEEDED',
})

###############################################################################
##########################     Functions     ##################################
###############################################################################

def _select_nonempty_ip(ips: pd.Series) -> str:
    """Selects non-empty IPv4/IPv6 address to unify data columns. Supposes that IPv4 is at the
    first place."""

    return ips[ips.index[0]] if ips[ips.index[1]] == '::' else ips[ips.index[1]]


def _map_tcp_flags(tcp_flags_ser : pd.Series, col_prefix: str = '') -> pd.DataFrame:
    """Map a series of TCP flags into DataFrame, with most important flags parsed into separate columns.

    Parameters:
        tcp_flags_ser -- Pandas series containing TCP flags to be processed
        col_prefix    -- Prefix to be used for new columns creation

    Returns:
        pd.DataFrame -- Dataframe of size (len(tcp_flags_ser), 8), having each of FIN, SYN, RST, PSH, ACK,
                        URG, ECE, and CWR flags parsed into their separate column using value masking"""

    col_prefix = col_prefix + '_' if col_prefix else col_prefix
    colnames = [col_prefix + 'FLAG_' + colname for colname in ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'ECE', 'CWR']]
    deenc_flags_df = pd.DataFrame(np.zeros((len(tcp_flags_ser), 8), dtype=np.int8), columns=colnames)

    # Set the value of the flag to 1 if it is present after AND operation
    for idx in range(len(colnames)):
        colname = colnames[idx]
        deenc_flags_df.loc[(tcp_flags_ser & 1 << idx != 0), colname] = 1

    return deenc_flags_df


def _categorize_port(port : int) -> str:
    """Performs port categorization into one of 12 pre-specified groups

    Parameters:
        port -- Integer value representing port to be categorized

    Returns:
        str -- String representing categorization of a given port"""

    portcat = defines.PCAT_SEARCHDICT[port]

    # Perform correction to other special, dynamic port category
    # Dynamic ports are defined from 49152 to 65535, since ports are 16-bit, upper bound is not necessary
    if portcat == 'OTHER' and port >= 49152:
        portcat = 'DYNAMIC'

    return portcat


def _manual_onehot(data : pd.DataFrame, cols : list, mapper: dict) -> pd.DataFrame:
    """Performs a manual one-hot encoding of the specified columns.  One-hot encoding is performed
    manually due to processing the dataset in batches, and not every value needs to be necessarily
    present in a categorical variable dataset's subset.

    Parameters
        data   -- Pandas DataFrame to be one-hot encoded
        cols   -- Columns to be one-hot encoded
        mapper -- Mapping function to be used for the encoding process

    Returns
        pd.DataFrame -- Pandas DataFrame with columns to be appropriately one-hot encoded"""

    all_colnames = []       # List of all created column names for data casting

    for feature in cols:
        for feature_value in mapper[feature]:
            # Create a new column for each potential value
            new_colname = feature + '_' + feature_value
            data[new_colname] = 0

            # Write 1 into the column for such categorical value
            data.loc[data[feature] == feature_value, new_colname] = 1

            all_colnames.append(new_colname)

    # Remove already-encoded original categorical columns
    data = data.drop(columns=cols)

    # Cast one hot encoded columns to smaller datatype to save space
    data[new_colname] = data[new_colname].astype('uint8')

    return data


def preprocess(data : pd.DataFrame, scaling_config : dict = None,
    scaling_minmax_new_max : float = None) -> pd.DataFrame:
    """Preprocesses dataset exported by nProbe into ML-acceptable form.

    Parameters
        data           -- DataFrame containing data to be preprocessed
        scaling_config -- Configuration for feature scaling to use. If None, performs no scaling.

    Returns:
        pd.DataFrame -- DataFrame with preprocessed data"""

    # Only remove columns when an empty DataFrame is passed
    if len(data) == 0:
        data = data.drop(columns=FEATURES_DROP + FEATURES_DROP_PROCESSED)
        return data

    # Unify IPv4 and IPv6 columns from 4 features into 2
    data['IP_SRC'] = data[['IPV4_SRC_ADDR', 'IPV6_SRC_ADDR']].apply(_select_nonempty_ip, axis=1)
    data['IP_DST'] = data[['IPV4_DST_ADDR', 'IPV6_DST_ADDR']].apply(_select_nonempty_ip, axis=1)

    # Compute "normalized" flow durations for each side in seconds or future fields computation
    # This means that if the duration is 0 (0 or 1 packet sent), it it set to 1ms to allow divisions
    data['IN_DUR_S']  = data['DURATION_IN'].apply(lambda x : x / 1000.0 if x > 0 else 0.001)
    data['OUT_DUR_S'] = data['DURATION_OUT'].apply(lambda x : x / 1000.0 if x > 0 else 0.001)

    # Compute other statistics
    data['IN_BPS'] = data['IN_BYTES'] / data['IN_DUR_S']    # Client's bytes-per-second
    data['IN_BPP'] = data['IN_BYTES'] / data['IN_PKTS']     # Client's bytes-per-packet
    data['IN_PPS'] = data['IN_PKTS'] / data['IN_DUR_S']     # Client's packets-per-second
    data['OUT_BPS'] = data['OUT_BYTES'] / data['OUT_DUR_S'] # Server's bytes-per-second
    data['OUT_BPP'] = data['OUT_BYTES'] / data['OUT_PKTS']  # Server's bytes-per-packet
    data['OUT_PPS'] = data['OUT_PKTS'] / data['OUT_DUR_S']  # Server's bytes-per-second

    # Fix bytes-per-packet values if number of packets was 0. This is a pretty ugly
    # principle, as many divisions by 0 occur. Maybe the computation itself would need fix
    data['IN_BPP']  = data['IN_BPP'].apply(lambda x : 0 if np.isnan(x) else x)
    data['OUT_BPP'] = data['OUT_BPP'].apply(lambda x : 0 if np.isnan(x) else x)

    # Decode ICMP type and map it to string representation
    # Nprobe encodes ICMP type and code into a single variable as: ICMP Type * 256 + ICMP Code
    # In our case, we decided to reconstruct only the ICMP code for further encoding.
    data['ICMP_TYPE'] = data['ICMP_TYPE'].apply(lambda type : ICMP_CODE_MAPPER[type // 256])

    # Map protocols into their respective categories to limit their number for one-hot encoding
    data['PROTOCOL'] = data['PROTOCOL'].apply(lambda x : PROTO_MAPPER[x])

    # Map TCP flags
    data = pd.concat([data, _map_tcp_flags(data['TCP_FLAGS'], '')], axis=1)
    data = pd.concat([data, _map_tcp_flags(data['CLIENT_TCP_FLAGS'], 'CLIENT')], axis=1)
    data = pd.concat([data, _map_tcp_flags(data['SERVER_TCP_FLAGS'], 'SERVER')], axis=1)

    # Perform port categorization
    data['L4_SRC_PORT'] = data['L4_SRC_PORT'].apply(_categorize_port)
    data['L4_DST_PORT'] = data['L4_DST_PORT'].apply(_categorize_port)

    # Perform categorical variables one-hot encoding and correct it due to ICMP specific case
    data = _manual_onehot(data, FEATURES_CATEGORICAL, FEATURES_CATEGORICAL_VALUES_MAPPER)
    data.loc[data['PROTOCOL_ICMP'] == 0, 'ICMP_TYPE_ECHO_REPLY'] = 0

    # Perform scaling if its config is specified
    if scaling_config is not None:
        data = scaler.scale_features(data, scaling_config, scaling_minmax_new_max)

    # Drop features which cannot be used for classification and other auxilliary features
    data = data.drop(columns=FEATURES_DROP + FEATURES_DROP_PROCESSED + FEATURES_DROP_AUXILLIARY)

    return data
