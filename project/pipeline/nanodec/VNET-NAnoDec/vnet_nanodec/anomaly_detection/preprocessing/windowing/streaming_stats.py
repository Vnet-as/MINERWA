"""
Algorithms for streaming statistics computation.
"""

from collections import defaultdict


IPV4_HDR_SIZE = 20
IPV6_HDR_SIZE = 40

PROTO_TO_HDR_SIZE = defaultdict(lambda: 20, {
    'tcp'  : 20,
    'udp'  : 8,
    'sctp' : 12,
    'icmp' : 8
})


def streaming_mean(new_elem_val, prev_avg, new_elems_cnt):
    """(Re)computes a mean of the stream.
    Parameters:
        new_elem_val   -- Value of the new element to include
        prev_avg       -- Previously computed stream average
        new_elems_cnt  -- Number of elements including new_elem_val"""

    return prev_avg + (new_elem_val - prev_avg) / new_elems_cnt


def streaming_variance(stream_var_aux_val, elems_cnt):
    """Computes stream (running) variance according to Welford's algorithm.
    Requires an auxiliary value S{k} computed by stream_var_aux() function. Afterwards, the
    function computes variance s^{2} as:
            s^{2} = S_{k} / (k-1)
    Parameters:
            stream_var_aux_val -- Auxiliary value for streaming variance computation
            elems_cnt          -- Number of elements included in stream_var_aux_val."""

    return (stream_var_aux_val / (elems_cnt - 1)) if elems_cnt > 1 else 0


def streaming_variance_aux(new_elem_val, prev_var_aux, prev_avg, new_avg):
    """Welford's running variance auxiliary value recomputation.
    Auxiliary value in k-th step S_{k} for stream variance is computed as:
        S_{k} = S_{k-1} + (x_{k} - m_{k-1}) * (x_{k} - m_{k})
    where: k    Computation step
        x_{k}   New element to include in variance computation
        m_{k}   Mean with element x_{k} already included
        m_{k-1} Previously computed mean without element x_{k}
    Parameters:
        new_elem_val -- Value of the new element to include
        prev_var_aux -- Previous auxiliary value for variance computation
        prev_avg     -- Previously computed average without new_elem_val included
        new_avg      -- Average with new_elem_val included"""

    return prev_var_aux + (new_elem_val - prev_avg) * (new_elem_val - new_avg)


def hdr_payload_estimate(ip: str, proto: str, packets: int, bytes: int) -> int:
    """Estimates header to payload packet ratio of the underlying flow based on
    the provided 4-tuple. Does not consider extension headers, options etc.
    Just pure headers in their basic size.

    Parameters:
        ip      -- IPv4 or IPv6 address of the flow
        proto   -- L4 protocol representation based on PROTO_TO_HDR_SIZE dictionary
        packets -- Number of packets in the flow
        bytes   -- Number of bytes in the flow
    """

    # Determine size based on the type of address
    hdrsize  = IPV4_HDR_SIZE if '.' in ip else IPV6_HDR_SIZE
    hdrsize += PROTO_TO_HDR_SIZE[proto]

    return (hdrsize * packets) / bytes
