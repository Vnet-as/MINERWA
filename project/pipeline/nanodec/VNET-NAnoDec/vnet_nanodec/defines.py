"""File specifying defines for various scripts within the ML pipeline."""

from collections import defaultdict


### Generic defines used across multiple files ###
# Flow end timestamp to sort the data by
COLNAME_FLOW_END_TSTAMP = 'FLOW_END_MILLISECONDS'


# Dictionary for dataset datatypes casting
FEATURES_CASTDICT = {
    'IN_BYTES'                    : 'uint32',
    'IN_PKTS'                     : 'uint32',
    'PROTOCOL'                    : 'uint8',
    'TCP_FLAGS'                   : 'uint8',
    'L4_SRC_PORT'                 : 'uint16',
    'IPV4_SRC_ADDR'               : 'string',
    'IPV6_SRC_ADDR'               : 'string',
    'L4_DST_PORT'                 : 'uint16',
    'IPV4_DST_ADDR'               : 'string',
    'IPV6_DST_ADDR'               : 'string',
    'OUT_BYTES'                   : 'uint32',
    'OUT_PKTS'                    : 'uint32',
    'MIN_IP_PKT_LEN'              : 'uint16',
    'MAX_IP_PKT_LEN'              : 'uint16',
    'ICMP_TYPE'                   : 'uint16',
    'MIN_TTL'                     : 'uint8',
    'MAX_TTL'                     : 'uint8',
    'DIRECTION'                   : 'uint8',
    'SRC_FRAGMENTS'               : 'uint16',
    'DST_FRAGMENTS'               : 'uint16',
    'CLIENT_TCP_FLAGS'            : 'uint8',
    'SERVER_TCP_FLAGS'            : 'uint8',
    'SRC_TO_DST_AVG_THROUGHPUT'   : 'uint32',
    'DST_TO_SRC_AVG_THROUGHPUT'   : 'uint32',
    'NUM_PKTS_UP_TO_128_BYTES'    : 'uint32',
    'NUM_PKTS_128_TO_256_BYTES'   : 'uint32',
    'NUM_PKTS_256_TO_512_BYTES'   : 'uint32',
    'NUM_PKTS_512_TO_1024_BYTES'  : 'uint32',
    'NUM_PKTS_1024_TO_1514_BYTES' : 'uint32',
    'NUM_PKTS_OVER_1514_BYTES'    : 'uint32',
    'LONGEST_FLOW_PKT'            : 'uint32',
    'SHORTEST_FLOW_PKT'           : 'uint32',
    'RETRANSMITTED_IN_PKTS'       : 'uint32',
    'RETRANSMITTED_OUT_PKTS'      : 'uint32',
    'OOORDER_IN_PKTS'             : 'uint32',
    'OOORDER_OUT_PKTS'            : 'uint32',
    'DURATION_IN'                 : 'uint32',
    'DURATION_OUT'                : 'uint32',
    'TCP_WIN_MIN_IN'              : 'uint16',
    'TCP_WIN_MAX_IN'              : 'uint16',
    'TCP_WIN_MSS_IN'              : 'uint16',
    'TCP_WIN_SCALE_IN'            : 'uint8',
    'TCP_WIN_MIN_OUT'             : 'uint16',
    'TCP_WIN_MAX_OUT'             : 'uint16',
    'TCP_WIN_MSS_OUT'             : 'uint16',
    'TCP_WIN_SCALE_OUT'           : 'uint8',
    'FLOW_VERDICT'                : 'uint16',
    'SRC_TO_DST_IAT_MIN'          : 'uint16',
    'SRC_TO_DST_IAT_MAX'          : 'uint16',
    'SRC_TO_DST_IAT_AVG'          : 'uint16',
    'SRC_TO_DST_IAT_STDDEV'       : 'uint16',
    'DST_TO_SRC_IAT_MIN'          : 'uint16',
    'DST_TO_SRC_IAT_MAX'          : 'uint16',
    'DST_TO_SRC_IAT_AVG'          : 'uint16',
    'DST_TO_SRC_IAT_STDDEV'       : 'uint16',
    'APPLICATION_ID'              : 'int32'
}

# Port categorization
# Sources:
#    - https://opensource.com/article/18/10/common-network-ports
#    - https://hostpapasupport.com/commonly-used-ports/
#    - https://en.wikipedia.org/wiki/List_of_TCP_and_UDP_port_numbers
#    - https://www.networkinghowtos.com/howto/common-vpn-ports-and-protocols
#    - https://www.rfc-editor.org/rfc/rfc7194.html
#    - https://en.wikipedia.org/wiki/Denial-of-service_attack#Amplification
#    - https://manuals.playstation.net/document/en/psvita/psn/firewall.html
#    - https://www.speedguide.net/ports.php

PCAT_NOPORT      = {0}                                                           # No TCP/UDP port used
PCAT_NOENC_WEB   = {80, 8080}                                                    # Web traffic
PCAT_TLS         = {443}                                                         # TLS traffic, probably web
PCAT_DNS         = {53}                                                          # Separate category due to traffic volume
PCAT_EMAIL       = {25, 26, 109, 110, 143, 209, 218, 220, 465, 587, 993, 995,    # Email traffic
                    2095, 2096}
PCAT_VPN         = {500, 1194, 1701, 1723, 4500}                                 # VPNs like L2TP, IPSec, OpenVPN...
PCAT_DATA        = {20, 21, 22, 69, 115, 989, 990, 2077, 2078}                   # Data transfer FTP, SFTP, TFTP
PCAT_SHELL       = {22, 23, 513, 514}                                            # Shell services - SSH, RSH,...
PCAT_PLAYSTATION = {3478, 3479, 3480}                                            # Playstation Network
PCAT_CHAT        = {194, 517, 518, 2351, 6667, 6697}                             # IRC chat
PCAT_QUERY       = {17, 19, 123, 1900, 3283, 4462, 4463, 5683, 6881, 6882,       # Query-response for amplification
                    6883, 6883, 6884, 6885, 6886, 6887, 6888, 6889, 11211, 26000}

# In order to make port categorization computationally easier, create a search dict from the above sets
PCAT_SEARCHDICT = defaultdict(lambda: 'OTHER', dict.fromkeys(PCAT_NOPORT, 'NOPORT') |
    dict.fromkeys(PCAT_NOENC_WEB, 'WEB_NOENC') | dict.fromkeys(PCAT_TLS, 'TLS') | \
    dict.fromkeys(PCAT_DNS, 'DNS') | dict.fromkeys(PCAT_EMAIL, 'EMAIL') | dict.fromkeys(PCAT_VPN, 'VPN') | \
    dict.fromkeys(PCAT_DATA, 'DATA') | dict.fromkeys(PCAT_SHELL, 'SHELL') | \
    dict.fromkeys(PCAT_PLAYSTATION, 'PLAYSTATION') | dict.fromkeys(PCAT_CHAT, 'CHAT') | dict.fromkeys(PCAT_QUERY, 'QUERY'))
