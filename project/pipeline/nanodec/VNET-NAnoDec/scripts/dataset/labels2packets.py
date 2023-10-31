"""Extracts packets into separate files based on the labelling file provided.

Author: Patrik Goldschmidt
Date: 19.09.2022

Usage:
python labels2packets.py <in_PCAP> <in_labels>
"""

import os
import scapy
import scapy.utils
import sys

# This needs to be included even if IP and IPv6 are not used or otherwise scappy will behave weird
from scapy.layers.inet import IP
from scapy.layers.inet6 import IPv6


# Which class to extract into a separate file
# If set to None, creates a separate PCAP for each class as <originalFile>_<className>.pcap
EXTRACT_CLASS = None


def main(args : list) -> None:
    pcap_filename   = args[1]   # Filename with packet data
    labels_filename = args[2]   # Filename containing labels
    labels_ordered  = []        # List of labels loaded from the file
    labels_pktmap   = {}        # Map of labels to individual output files
    pkt_cnt         = 0         # PCAP reader packet counter
    pcap_reader     = scapy.utils.PcapReader(pcap_filename)

    # Open the file with labels
    with open(labels_filename, 'r') as labels_file:
        labels_ordered = labels_file.read().split()

    # Create a map of labels into output PCAP files
    if EXTRACT_CLASS == None:
        # Extraction of all classes
        labels_set = set(labels_ordered)

        for label in labels_set:
            labels_pktmap[label] = scapy.utils.PcapWriter(os.path.splitext(
            pcap_filename)[0] + '_' + label + '.pcap', nano=True)
    else:
        # Extraction of a particular class
        labels_pktmap[EXTRACT_CLASS] = scapy.utils.PcapWriter(os.path.splitext(
        pcap_filename)[0] + '_' + EXTRACT_CLASS + '.pcap', nano=True)

    # Read packet-by-packet and extract
    for pkt in pcap_reader:
        if labels_ordered[pkt_cnt] in labels_pktmap:
            labels_pktmap[labels_ordered[pkt_cnt]].write(pkt)

        pkt_cnt += 1

    # Close all opened readers and writers
    pcap_reader.close()

    for pcap_writer in labels_pktmap.values():
        pcap_writer.close()


if __name__ == '__main__':
    main(sys.argv)
