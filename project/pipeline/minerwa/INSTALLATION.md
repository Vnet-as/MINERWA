# Installation guide

For quickstart, the application is dockerized, so the most of the installation
process consists of running [Docker Compose](#quick-startup-using-docker-compose).
That is, however, mostly good for quick demo purposes not for
[real-world deployment](#advanced-deployment).

## Quick startup using Docker Compose

Since nProbe in this setup is configured using command-line arguments, you
should quickly check it. Important part is switch `-i wlan0` that defines
network interface on which the traffic will be monitored. Unless `wlan0` is
available on you computer, you should configure one of the available interfaces.
To get the list of available network interfaces on you system, you can use
for examples command `ip link`.
  * [ ] 
Running docker-compose:

```
$ docker compose up
```

Running services are isolated on bridged docker network, except nprobe service,
which runs in host network mode to be able to capture traffic on physical interface.

Running using Docker Compose is also good for development, therefore in
`docker-compose.yml` file you can find volume mounts of application folders.

## Advanced deployment

### nProbe

In the basic setup nProbe runs without licence in combined mode (exporter and collector).
In case of bigger deployment, running demo version of nProbe on single interface
might be not enough.

It is possible to mount a licence file into container to run commercial,
licensed version of nProbe in Docker, yet it might not be optimal for production.
Using RSPAN and other types of TAP to mirror network traffic might bring
a need to capture traffic on multiple network interfaces (e.g. separate interface
for each direction of traffic or multiple points in network mirrored to one machine).

In such case, nProbe can be ran separately as exporter (possibly multiple instances,
one for each interface) and collector (single instance for all exporters).

Example of running multiple nProbes:

```
nprobe -i=ens6f0 -n=udp://127.0.0.1:2055 -V10 -u=0
nprobe -i=ens6f1 -n=udp://127.0.0.1:2055 -V10 -u=1
nprobe -i=ens10f0np0 -n=udp://127.0.0.1:2055 -V10 -u=0
nprobe -i=ens10f1np1 -n=udp://127.0.0.1:2055 -V10 -u=1
```

In above case, we are capturing mirrored traffic from 2 interfaces of backbone router
(each mirrored interface on router is mirrored to a interfaces of ports, one for
each direction, on our collector machine). All of these exporters are sending IPFIX
flows to a single collector instance.

```
nprobe -i=none --collector-port=2055 --ntopng=tcp://0.0.0.0:5556 --zmq-format=j -n=none -T="%IN_BYTES %IN_PKTS %PROTOCOL %TCP_FLAGS %L4_SRC_PORT %IPV4_SRC_ADDR %IPV6_SRC_ADDR %L4_DST_PORT %IPV4_DST_ADDR %IPV6_DST_ADDR %OUT_BYTES %OUT_PKTS %MIN_IP_PKT_LEN %MAX_IP_PKT_LEN %ICMP_TYPE %MIN_TTL %MAX_TTL %DIRECTION %FLOW_START_MILLISECONDS %FLOW_END_MILLISECONDS %SRC_FRAGMENTS %DST_FRAGMENTS %CLIENT_TCP_FLAGS %SERVER_TCP_FLAGS %SRC_TO_DST_AVG_THROUGHPUT %DST_TO_SRC_AVG_THROUGHPUT %NUM_PKTS_UP_TO_128_BYTES %NUM_PKTS_128_TO_256_BYTES %NUM_PKTS_256_TO_512_BYTES %NUM_PKTS_512_TO_1024_BYTES %NUM_PKTS_1024_TO_1514_BYTES %NUM_PKTS_OVER_1514_BYTES %SRC_IP_COUNTRY %DST_IP_COUNTRY %SRC_IP_LONG %SRC_IP_LAT %DST_IP_LONG %DST_IP_LAT %LONGEST_FLOW_PKT %SHORTEST_FLOW_PKT %RETRANSMITTED_IN_PKTS %RETRANSMITTED_OUT_PKTS %OOORDER_IN_PKTS %OOORDER_OUT_PKTS %DURATION_IN %DURATION_OUT %TCP_WIN_MIN_IN %TCP_WIN_MAX_IN %TCP_WIN_MSS_IN %TCP_WIN_SCALE_IN %TCP_WIN_MIN_OUT %TCP_WIN_MAX_OUT %TCP_WIN_MSS_OUT %TCP_WIN_SCALE_OUT %FLOW_VERDICT %SRC_TO_DST_IAT_MIN %SRC_TO_DST_IAT_MAX %SRC_TO_DST_IAT_AVG %SRC_TO_DST_IAT_STDDEV %DST_TO_SRC_IAT_MIN %DST_TO_SRC_IAT_MAX %DST_TO_SRC_IAT_AVG %DST_TO_SRC_IAT_STDDEV %APPLICATION_ID"
```

Collector instance is collecting exported flows and serving them in specified format
(enumerated fields according to [nProbe documentation](https://www.ntop.org/guides/nprobe/flow_information_elements.html))
via ZeroMQ (nProbe is listening for connection and broadcasts all flows to all connected
subscribers, so there is no way of load-balancing or horizontal scaling usin multiple
subscribers).

### NATS

The are no special requirements for running NATS server.

JetStream feature is currently not used, so just running `./nats-server` binary
distribution executable is enough. By default, it runs on port *4222*.
Optionally `-m 8222` option can be used to open monitoring interface used
by *nats-top* utility.

### IPFIX fields definition
Homogenous and predictable data are always better for handling and performance.
Therefore we rely on using fixed (though user-definable) list of IPFIX fields.

Three major parts of system are set-up based of this list:

1. tables and views in ClickHouse
2. flow-representing dataclass in code
3. Cap'n Proto schema used for effective serialization of data flowing between services

Example definition, also used for demo setup is in [flow_definition.yaml](flow_definition.yaml) file.
Definition is basically a YAML list of objects with 3 keys:
- *id* - string ID of IPFIX field (string because non-standard fields uses dot and even float would be impractical in that case)
- *name* - name of field (usually defined by author of exporter)
- *type* - precise datatype of field (int, uint8, uint16, uint32, uint64, ipv4, ipv6, str)

Once you have flow-definition file prepared, place it in the root folder of this project (default) or
set path to it in config (section `DEFAULT`, option `flow_definition`).

Configured file is used as a source of data for generating Clickhouse DDL
and Cap'n Proto schema and runtime creation of flow-representing datascructure.

#### ClickHouse DDL
Now you can generate ClickHouse table DDL based on it using prepared command:

```
$ docker compose run ingestor gen_clickhouse_ddl --sql-output -
```

This DDL can be either saved to `clickhouse_init.sql` for automatic DB initialization
when using Docker or be manually applied to any ClickHouse instance.

#### Cap'n Proto schema
Cap'n Proto schema file is used in all services that are processing flows.
Not only in Python application, but also in ClickHouse, which is consuming
and storing flows from NATS. Also in case of writing or rewriting parts
of project in different language, Cap'n Proto is unified schema for exchanging
homogenous date among heterogenous applications.

This schema could be easily generated from flow definition file using command:

```
$ docker compose run ingestor gen_capnproto --schema-output -
```

This schema could be used wherever needed (every service that inspects internal flows
serialized using Cap'n Proto), therefore must be also stored in file along other
configuration files (and path to file properly configured in config file
- section `DEFAULT`, option `capnp_schema`).
To be used by ClickHouse, schema file should by placed to ClickHouse schema
directory (default `/var/lib/clickhouse/format_schemas`) and filename correctly
configured in DDL before applying.
