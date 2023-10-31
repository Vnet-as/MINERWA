# Brief: Packets extraction from PCAP file, given flows
# Author: Patrik Goldschmidt, patrik.goldschmidt@kinit.sk
# Date: 08-08-2022
#
# Usage:
# python f2p.py <inputPCAP> <outputPCAP> <referenceFlows>
#
# What's it good for:
# The program extracts packets from a PCAP file based on a corresponding flow file, selecting only
# packets contained within flows. Many datasets label their PCAP data with flows, but when we want
# to extract packets from those PCAPs, a problem begins - how do we want to extract them? Consider
# an example when flows labeled as "DoS" are interesting for us. However, the corresponding PCAP is
# intermixed with benign traffic and other types of attacks. According to our brief research, no
# tools to extract packets from flows in a simple manner exist. And by a pure strike of luck, this
# tool exactly does that. It extracts all packets CORRESPONDING TO ANY FLOW within the
# <referenceFlows> file. Therefore, to extract a particular class type, the pandas library has to
# be used first. Firstly, filter out the flows of interest, e.g., data[data["label"] == DoS], and
# then use this tool to extract relevant packets from the selected flows.
#
# Tweaking for other datasets:
# The tool is currently customized for the NDSec-1 dataset. However, it can be adapted to any
# dataset with brief changes to code. Variables just under the imports commented as "Flows dataset
# column names" need to be changed to represent column names of the flow CSV file. Furthermore,
# variables with the comment "Dataset properties" should be adapted. And that should be it!
# However, the dataset is required to contain all the columns specified by the variables previously
# set up. However, the flow end-time column may sometimes not be present. In this case, modify the
# function "prepare_dataset" to compute such a row by adding a flow start timestamp (already
# converted to epoch) with the flow duration. Voila!


import decimal
import pandas as pd
import math
import scapy.packet
import scapy.utils
import sys

from dataclasses import dataclass
from scapy.layers.sctp import SCTP
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.inet6 import IPv6, _ICMPv6


# Flows dataset column names
FLOWS_COL_TSTAMP_START = 'start-time'
FLOWS_COL_TSTAMP_END   = 'end-time'
FLOWS_COL_IP_SRC       = 'srcip'
FLOWS_COL_IP_DST       = 'dstip'
FLOWS_COL_PORT_SRC     = 'srcport'
FLOWS_COL_PORT_DST     = 'dstport'
FLOWS_COL_PROTO        = 'protocol'

# Dataset properties
DATASET_BIFLOW = True
DATASET_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
DATASET_TIMESTAMP_PRECISION = 3


# Internal computations and variables, do not modify
TIMESTAMP_MODIF_CONST  = float(decimal.Decimal(1).shift(DATASET_TIMESTAMP_PRECISION))
FLOWS_COL_EPOCH_TSTAMP_START = 'stime'
FLOWS_COL_EPOCH_TSTAMP_END   = 'etime'


@dataclass
class PacketInfo:
    """A simple structure-like class for passing extracted data from packets."""

    timestamp : int = 0
    src_ip    : str = ""
    dst_ip    : str = ""
    src_port  : int = 0
    dst_port  : int = 0
    proto     : int = 0


def prepare_dataset(dataset: pd.DataFrame):
    """Prepares a dataset for flow membership determination by converting string timestamp to
    the epoch-like format while creating new columns based on names specified in the program
    header."""

    dataset[FLOWS_COL_EPOCH_TSTAMP_START] = dataset[FLOWS_COL_TSTAMP_START].apply(
        lambda x: pd.to_datetime(x, format=DATASET_TIMESTAMP_FORMAT).timestamp())
    dataset[FLOWS_COL_EPOCH_TSTAMP_END]   = dataset[FLOWS_COL_TSTAMP_END].apply(
        lambda x: pd.to_datetime(x, format=DATASET_TIMESTAMP_FORMAT).timestamp())

    return dataset


def extract_packet_info(pkt: scapy.packet.Packet):
    """Parses packet information and returns a PacketInfo structure with filled information or None
    if the packet is of non-interest."""

    pkt_info = PacketInfo()

    # Packet timestamp is extracted directly
    pkt_info.timestamp = pkt.time

    # Determine L3 layer
    if pkt.haslayer(IP):
        pkt_info.src_ip = pkt[IP].src
        pkt_info.dst_ip = pkt[IP].dst
    elif pkt.haslayer(IPv6):
        pkt_info.src_ip = pkt[IPv6].src
        pkt_info.dst_ip = pkt[IPv6].dst

    # Determine L4 layer
    if pkt.haslayer(TCP):
        pkt_info.proto = 6
        pkt_info.src_port = pkt[TCP].sport
        pkt_info.dst_port = pkt[TCP].dport
    elif pkt.haslayer(UDP):
        pkt_info.proto = 17
        pkt_info.src_port = pkt[UDP].sport
        pkt_info.dst_port = pkt[UDP].dport
    elif pkt.haslayer(ICMP):
        pkt_info.proto = 1
    elif pkt.haslayer(_ICMPv6):
        pkt_info.proto = 58
    elif pkt.haslayer(SCTP):
        pkt_info.proto = 132
        pkt_info.src_port = pkt[SCTP].sport
        pkt_info.dst_port = pkt[SCTP].dport
    else:
        # Other protocols than TCP/UDP/ICMPv4 (v6) and SCTP are not considered for flows anyway
        return None

    return pkt_info


def search_for_flow_inclusion(pkt_info: PacketInfo, flows: pd.DataFrame):
    """Searches whether the packet defined by pkt_info is included within the flows dataframe."""

    filt_flows = flows[
        (flows[FLOWS_COL_PROTO] == pkt_info.proto) &
        (flows[FLOWS_COL_IP_SRC] == pkt_info.src_ip) &
        (flows[FLOWS_COL_PORT_SRC] == pkt_info.src_port) &
        (flows[FLOWS_COL_IP_DST] == pkt_info.dst_ip) &
        (flows[FLOWS_COL_PORT_DST] == pkt_info.dst_port)
    ]

    # Out of all selected flows, perform a look based on a timestamp
    # Ceil and floors are included to make sure the packet will get matched to the flow, if the
    # dataset uses rounding/truncating timestamps on a certain number of decimal places
    matched_flow = filt_flows[
        (filt_flows[FLOWS_COL_EPOCH_TSTAMP_START] <= math.ceil(pkt_info.timestamp * TIMESTAMP_MODIF_CONST) / TIMESTAMP_MODIF_CONST) &
        (filt_flows[FLOWS_COL_EPOCH_TSTAMP_END]   >= math.floor(pkt_info.timestamp * TIMESTAMP_MODIF_CONST) / TIMESTAMP_MODIF_CONST)
    ]

    return not matched_flow.empty


def is_in_flows(pkt_info: PacketInfo, flows: pd.DataFrame):
    """Searches whether the packet is included in the flows dataframe, in uni-flow mode by default, but also
    reverses the fields if the dataset uses bi-flows."""

    # Search the flows dataframe and get any row that mathces the given 5-column
    retval = False

    if search_for_flow_inclusion(pkt_info, flows):
        retval = True
    elif DATASET_BIFLOW:
        # If the dataset is composed of biflows, search once againt with swapped IPs and ports
        reversed_pkt_info = PacketInfo(
            timestamp = pkt_info.timestamp,
            src_ip    = pkt_info.dst_ip,
            src_port  = pkt_info.dst_port,
            dst_ip    = pkt_info.src_ip,
            dst_port  = pkt_info.src_port,
            proto     = pkt_info.proto)

        retval = search_for_flow_inclusion(reversed_pkt_info, flows)

    return retval


def main(args: list):
    # Check if the script is run correctly
    if len(args) != 4:
        raise Exception("Invalid number of arguments provided.")

    # Open file handles
    in_pcap_reader  = scapy.utils.PcapReader(args[1])
    out_pcap_writer = scapy.utils.PcapWriter(args[2], nano=True)
    flows           = pd.read_csv(args[3])

    # Prepare dataset by converting timestamps into epochs
    flows = prepare_dataset(flows)

    # Process the PCAP file, writing packets existing in the flow file to the output PCAP
    for pkt in in_pcap_reader:
        # Extract packet distinguishing features determining correspondence to a certain flow
        pkt_info = extract_packet_info(pkt)

        if pkt_info is not None and is_in_flows(pkt_info, flows):
            out_pcap_writer.write(pkt)

        #sys.exit(0)

    # Close the opened file handles
    in_pcap_reader.close()
    out_pcap_writer.close()


if __name__ == '__main__':
    main(sys.argv)
