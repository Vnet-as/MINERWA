"""
Computed traffic statistics for the windowed model.
"""

import numpy as np

STATS_NPDTYPE_WINDOW_OPEN = np.dtype([
    # Data that will be transferred directly to the completed window
    ('flows_total',             np.uint32),     # Total number of flows
    ('dur_total',               np.float32),    # Total flow duration
    ('dur_avg',                 np.float32),    # Flow duration average
    ('pkt_total',               np.uint32),     # Total number of packets
    ('ppf_avg',                 np.float32),    # Packets per flow average
    ('bytes_total',             np.uint64),     # Total number of bytes
    ('bpf_avg',                 np.float32),    # Bytes per flow average
    ('bpp_avg',                 np.float32),    # Bytes per packet average
    ('bps_avg',                 np.float32),    # Bytes per second average
    ('pps_avg',                 np.float32),    # Packets per second average
    ('proto_all_or',            np.uint8),      # OR value of all encountered L4 protocols
    ('hdr_payload_ratio_avg',   np.float32),    # Packets' header to payload ratio estimate

    # Auxilliary data structures
    ('flow_last_tstamp',        np.float32),    # Start of the lastly processed flows
    ('dur_std_aux',             np.float32),    # Streaming std for flow duration
    ('ppf_std_aux',             np.float32),    # Streaming std for packets per flow
    ('bpf_std_aux',             np.float32),    # Streaming std for bytes per flow
    ('bpp_std_aux',             np.float32),    # Streaming std for bpp
    ('bps_std_aux',             np.float32),    # Streaming std for bps
    ('pps_std_aux',             np.float32),    # Streaming std for bpp
    ('ttl_avg',                 np.float32),    # Streaming average of TTL
    ('ttl_std_aux',             np.float32),    # Streaming std for TTL computation
    ('flows_syn',               np.uint32),     # Flows with SYN flag active
    ('flows_ack',               np.uint32),     # Flows with ACK flag active
    ('flows_fin',               np.uint32),     # Flows with FIN flag active
], align=True)

STATS_NPDTYPE_WINDOW_COMPLETED = np.dtype([
    ('window_id',               np.uint32),     # Window identifier
    # Flow information
    ('flows_total',             np.uint32),     # Total number of flows
    ('flows_concurrent_max',    np.uint16),     # Number of maximum concurrent flows
    ('flows_itimes_avg',        np.float32),    # Average time between flows starts
    ('flows_itimes_std',        np.float32),    # Standard deviation between flows starts
    ('flows_itimes_min',        np.float32),    # Minimum time between two flow starts
    ('flows_itimes_max',        np.float32),    # Maximum time between two flow starts
    ('dur_total',               np.float32),    # Sum of flow durations within the window
    ('dur_avg',                 np.float32),    # Flow duration average
    ('dur_std',                 np.float32),    # Flow duration std
    # Packet count & bytes information over time
    ('pkt_total',               np.uint32),     # Total number of packets
    ('ppf_avg',                 np.float32),    # Packets per flow average
    ('ppf_std',                 np.float32),    # Packets per flow std
    ('bytes_total',             np.uint64),     # Total number of bytes
    ('bpf_avg',                 np.float32),    # Bytes per flow average
    ('bpf_std',                 np.float32),    # Bytes per flow std
    ('bpp_avg',                 np.float32),    # Bytes per packet average
    ('bpp_std',                 np.float32),    # Bytes per packet std
    ('bps_avg',                 np.float32),    # Bytes per second average
    ('bps_std',                 np.float32),    # Bytes per second std
    ('pps_avg',                 np.float32),    # Packets per second average
    ('pps_std',                 np.float32),    # Packets per second std
    # Ports
    ('port_src_uniq_cnt',       np.uint16),     # Number of unique source ports
    ('port_src_entropy',        np.float32),    # Source port entropy
    # Protocols and their flags
    ('proto_all_or',            np.uint8),      # OR value of all encountered L4 protocols
    ('flag_syn_ratio',          np.float32),    # Ratio of SYN flags in flows
    ('flag_ack_ratio',          np.float32),    # Ratio of ACK flags in flows
    ('flag_fin_ratio',          np.float32),    # Ratio of FIN flags in flows
    # Miscellaneous
    ('ip_dst_uniq',             np.uint16),     # Number of unique destination IP addresses
    ('hdr_payload_ratio_avg',   np.float32),    # Packets' header to payload ratio estimate
    ('ttl_std',                 np.float32),    # Time-to-live std
], align=True)

STATS_NPDTYPE_INTERWINDOW = np.dtype([
    # Interwindow statistics parameters - these are not particularly useful for classification
    ('window_total',                np.uint16),     # Total number of summarized windows
    ('window_span',                 np.uint16),     # Last window ID - first window ID
    # Additional statistics produced by processing several windows
    ('window_active_ratio',         np.float32),        # Summarized windows actity ratio
    # Interwindow standard deviations of underlying variables in STATS_NPDTYPE_WINDOW_COMPLETED
    ('flows_total_std',             np.float32),
    ('flows_concurrent_max_std',    np.float32),
    ('flows_itimes_avg_std',        np.float32),
    ('flows_itimes_std_std',        np.float32),
    ('flows_itimes_min_std',        np.float32),
    ('flows_itimes_max_std',        np.float32),
    ('dur_total_std',               np.float32),
    ('dur_avg_std',                 np.float32),
    ('dur_std_std',                 np.float32),
    ('pkt_total_std',               np.float32),
    ('ppf_avg_std',                 np.float32),
    ('ppf_std_std',                 np.float32),
    ('bytes_total_std',             np.float32),
    ('bpf_avg_std',                 np.float32),
    ('bpf_std_std',                 np.float32),
    ('bps_avg_std',                 np.float32),
    ('bps_std_std',                 np.float32),
    ('bpp_avg_std',                 np.float32),
    ('bpp_std_std',                 np.float32),
    ('port_src_uniq_cnt_std',       np.float32),
    ('port_src_entropy_std',        np.float32),
    ('flag_syn_ratio_std',          np.float32),
    ('flag_ack_ratio_std',          np.float32),
    ('flag_fin_ratio_std',          np.float32),
    ('ip_dst_uniq_std',             np.float32),
    ('hdr_payload_ratio_avg_std',   np.float32),
    ('ttl_std_std',                 np.float32),
], align=True)
