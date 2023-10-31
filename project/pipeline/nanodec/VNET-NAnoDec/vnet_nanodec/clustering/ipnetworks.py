"""
Fast matching of IP addresses to IP networks.
"""

import os
import sys
import ipaddress
from typing import Union

# Append to system path for imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)).rsplit('src', 1)[0])

import ipcategories


def is_ip_in_networks(
        ip_addr: Union[ipaddress.IPv4Address, ipaddress.IPv6Address],
        int_ip_networks: dict[int, int],
        ip_network_prefixlens: set[int],
        version: int,
) -> bool:
    return get_smallest_matching_network_ip_address(
        ip_addr, int_ip_networks, ip_network_prefixlens, version
    ) is not None


def get_int_ip_networks_and_prefixlens(ip_categories: ipcategories.IPCategories):
    int_ipv4_networks = {
        int(network.network_address): int(network.broadcast_address)
        for network in ip_categories.ipv4_networks}
    ipv4_network_prefixlens = sorted(
        set([network.prefixlen for network in ip_categories.ipv4_networks]), reverse=True)

    int_ipv6_networks = {
        int(network.network_address): int(network.broadcast_address)
        for network in ip_categories.ipv6_networks}
    ipv6_network_prefixlens = sorted(
        set([network.prefixlen for network in ip_categories.ipv6_networks]), reverse=True)

    return int_ipv4_networks, ipv4_network_prefixlens, int_ipv6_networks, ipv6_network_prefixlens


def get_smallest_matching_network_ip_address(
        ip_addr: Union[ipaddress.IPv4Address, ipaddress.IPv6Address],
        int_ip_networks: dict[int, int],
        ip_network_prefixlens: set[int],
        version: int,
) -> Union[tuple[int, int], None]:
    if version == 4:
        max_prefixlen = 32
        max_number = 0xffffffff
    else:
        max_prefixlen = 128
        max_number = 0xffffffffffffffffffffffffffffffff

    for prefixlen in ip_network_prefixlens:
        int_ip = int(ip_addr)
        int_network_ip = int_ip & (max_number ^ (2 ** (max_prefixlen - prefixlen) - 1))
        int_broadcast_ip = int_network_ip + (2 ** (max_prefixlen - prefixlen) - 1)

        if int_network_ip in int_ip_networks and int_ip_networks[int_network_ip] == int_broadcast_ip:
            return int_network_ip, prefixlen

    return None
