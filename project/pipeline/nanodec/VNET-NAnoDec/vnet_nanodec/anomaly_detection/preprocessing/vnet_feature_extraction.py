"""
Feature extraction function from VNET data flows in CSV format from nProbe.
"""

from collections import defaultdict

import pandas as pd

import vnet_nanodec.utils as utils
from .windowing.flow_features import FlowFeatures


PROTO_MAPPER = defaultdict(lambda: 'other', {
    0   : 'other',
    1   : 'icmp',
    6   : 'tcp',
    17  : 'udp',
    132 : 'stcp'
})


def extract_features(flow: pd.Series) -> FlowFeatures:
    """Features extraction from VNET dataset.

    Parameters:
        flow -- Single flow (line from dataset) as Pandas Series

    Returns:
        FlowFeatures -- Dataclass containing extracted features"""

    # Determine flow duration, if only 1 packet is sent, consider duration to be 10ms
    flow_dur_s = utils.milliseconds_to_seconds(
        flow['FLOW_END_MILLISECONDS'] - flow['FLOW_START_MILLISECONDS'])
    flow_dur_s = flow_dur_s if flow_dur_s != 0 else 0.01

    # Determine source and destination IPs
    flow_src_ip = flow['IPV4_SRC_ADDR'] if flow['IPV6_SRC_ADDR'] == '::' else flow['IPV6_SRC_ADDR']
    flow_dst_ip = flow['IPV4_DST_ADDR'] if flow['IPV6_DST_ADDR'] == '::' else flow['IPV6_DST_ADDR']

    features = FlowFeatures(
        tstamp_start = flow['FLOW_START_MILLISECONDS'],
        tstamp_end   = flow['FLOW_END_MILLISECONDS'],
        packets      = flow['IN_PKTS'],
        bytes        = flow['IN_BYTES'],
        bpp          = flow['IN_BYTES'] / flow['IN_PKTS'],
        bps          = flow['IN_BYTES']  / flow_dur_s,
        pps          = flow['IN_PKTS'] / flow_dur_s,
        ip_src       = flow_src_ip,
        ip_dst       = flow_dst_ip,
        proto        = PROTO_MAPPER[flow['PROTOCOL']],
        port_src     = flow['L4_DST_PORT'],
        port_dst     = flow['L4_SRC_PORT'],
        ttl          = flow['MIN_TTL'],
        flag_syn     = (flow['CLIENT_TCP_FLAGS'] & 0x2) > 0,
        flag_ack     = (flow['CLIENT_TCP_FLAGS'] & 0x10) > 0,
        flag_fin     = (flow['CLIENT_TCP_FLAGS'] & 0x1) > 0
    )

    return features
