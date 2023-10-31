"""
Extracted flow features statistics.
"""

from dataclasses import dataclass


@dataclass
class FlowFeatures:
    """Features extracted from the flow for the purpose of statistics logging."""
    tstamp_start : int          # Millisecond timestamp of the flow start
    tstamp_end   : int          # Millisecond timestamp of the flow end
    packets      : int          # Number of packets in the flow
    bytes        : int          # Number of bytes in the flow
    bpp          : float        # Bytes per packet
    bps          : float        # Bytes per second
    pps          : float        # Packets per second
    ip_src       : str          # Source IP address
    ip_dst       : str          # Destination IP address
    proto        : int          # L4 protocol identifier
    port_src     : int          # Source L4 port number
    port_dst     : int          # Destination L4 port number
    ttl          : int          # Time-to-live value
    flag_syn     : bool         # SYN flag present in the flow
    flag_ack     : bool         # ACK flag present in the flow
    flag_fin     : bool         # FIN flag present in the flow
