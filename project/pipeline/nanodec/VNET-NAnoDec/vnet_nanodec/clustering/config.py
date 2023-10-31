"""
Common configuration for clustering scripts.
"""

import collections


ISP_OWNED_IP_ADDR_COLUMN = 'IP_SRC_ADDR'
HYPERLOGLOG_ERROR_RATE = 0.05

PORT_CATEGORIES = {
    # No TCP/UDP port used
    'noport': {0},
    # Web traffic
    'noenc_web': {80, 8080},
    # TLS traffic, probably web
    'tls': {443},
    # Separate category due to traffic volume
    'dns': {53},
    # Email traffic
    'email': {25, 26, 109, 110, 143, 209, 218, 220, 465, 587, 993, 995, 2095, 2096},
    # VPNs like L2TP, IPSec, OpenVPN...
    'vpn': {500, 1194, 1701, 1723, 4500},
    # Data transfer FTP, SFTP, TFTP
    'data': {20, 21, 22, 69, 115, 989, 990, 2077, 2078},
    # Shell services - SSH, RSH,...
    'shell': {22, 23, 513, 514},
    # Playstation Network
    'playstation': {3478, 3479, 3480},
    # IRC chat
    'chat': {194, 517, 518, 2351, 6667, 6697},
    # Query-response for amplification
    'query': {
        17, 19, 123, 1900, 3283, 4462, 4463, 5683,
        6881, 6882, 6883, 6883, 6884, 6885, 6886, 6887, 6888, 6889,
        11211, 26000},
}

LOWEST_DYNAMIC_PORT = 49152
PORT_CATEGORIES_LIST = list(PORT_CATEGORIES) + ['dynamic', 'other']
PORT_CATEGORIES_AND_IDS = {category: i for i, category in enumerate(PORT_CATEGORIES_LIST)}


# Taken from: https://stackoverflow.com/a/2912596
class _KeyDefaultDict(collections.defaultdict):

    def __missing__(self, key):
        if self.default_factory is not None:
            # noinspection PyArgumentList
            value = self.default_factory(key)
            self[key] = value
            return value
        else:
            super().__missing__(key)


def _get_port_searchdict() -> collections.defaultdict:
    port_searchdict = _KeyDefaultDict(
        lambda p: PORT_CATEGORIES_AND_IDS['other'] if p < LOWEST_DYNAMIC_PORT
        else PORT_CATEGORIES_AND_IDS['dynamic'])

    for port_category, ports in PORT_CATEGORIES.items():
        for port in ports:
            port_searchdict[port] = PORT_CATEGORIES_AND_IDS[port_category]

    return port_searchdict


PORT_SEARCHDICT = _get_port_searchdict()
