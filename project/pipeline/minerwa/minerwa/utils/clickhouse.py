import sys
import textwrap

from minerwa.helpers import flow_definition
from minerwa.model import camel_case


TYPE_MAP = {
    'int': 'UInt32',
    'uint8': 'UInt8',
    'uint16': 'UInt16',
    'uint32': 'UInt32',
    'uint64': 'UInt64',
    'ipv4': 'IPv4',
    'ipv6': 'IPv6',
    'str': 'String',
}


def _generate_flow_columns(
        fields: list[flow_definition.FieldDefinition]
) -> str:
    columns = ['id UUID']
    for f in fields:
        name = camel_case(f.name)
        type_ = TYPE_MAP.get(f.type, f.type)
        col = f'{name} {type_}'
        columns.append(col)
    return ',\n'.join(columns)


def _generate_ingestion_table(
        fields: list[flow_definition.FieldDefinition]
) -> str:
    columns = _generate_flow_columns(fields)
    return textwrap.dedent(f'''
    CREATE TABLE IF NOT EXISTS minerwa_ingestion (
        \n{textwrap.indent(columns, " " * 8).replace('id UUID', 'id String')}
    ) ENGINE = NATS SETTINGS
        nats_url = '',
        nats_subjects = '',
        nats_format = 'CapnProto',
        nats_schema = 'schema.capnp:Flow';
    ''')


def _generate_flow_table(
        fields: list[flow_definition.FieldDefinition]
) -> str:
    columns = _generate_flow_columns(fields)
    return textwrap.dedent(f'''
    CREATE TABLE IF NOT EXISTS minerwa_flows (
        \n{textwrap.indent(columns, " " * 8)},
        dtc DateTime MATERIALIZED now()
    ) ENGINE = MergeTree()
      ORDER BY id
      TTL dtc + INTERVAL 1 HOUR;
    ''')


def _generate_materialized_view() -> str:
    return ('CREATE MATERIALIZED VIEW minerwa_flows_mv TO minerwa_flows AS '
            'SELECT * FROM minerwa_ingestion;')


def _generate_detections_table() -> str:
    return (textwrap.dedent('''
    SET allow_experimental_object_type=1;

    CREATE TABLE IF NOT EXISTS minerwa_detections_nats (
        flow_id UUID NOT NULL,
        detector String NOT NULL,
        event_name String NOT NULL,
        metric Float32,
    ) ENGINE = NATS SETTINGS
        nats_url = 'nats://nats:4222',
        nats_subjects = 'minerwa.detection',
        nats_format = 'JSONEachRow';

    CREATE TABLE minerwa_detections (
        flow_id UUID NOT NULL,
        detector String NOT NULL,
        event_name String NOT NULL,
        metric Float32,
        data JSON,
        event_time NOT NULL DEFAULT now())
    ENGINE = MergeTree()
    ORDER BY event_time;

    CREATE MATERIALIZED VIEW IF NOT EXISTS minerwa_detections_nats_mv
    TO minerwa_detections AS SELECT * FROM minerwa_detections_nats;
    '''))


def _generate_detections_materialized_view() -> str:
    return (textwrap.dedent('''
    CREATE TABLE IF NOT EXISTS minerwa_detections_flows (
        flow_id UUID NOT NULL,
        ipv4SrcAddr IPv4,
        ipv4DstAddr IPv4,
        protocol UInt8,
        event_name String NOT NULL,
        detector String NOT NULL,
        metric Float32,
        event_time DateTime NOT NULL DEFAULT now())
    ENGINE = MergeTree
    ORDER BY flow_id;

    CREATE MATERIALIZED VIEW minerwa_detections_mv TO minerwa_detection_flows
    AS SELECT
        minerwa_flows.id as flow_id,
        minerwa_flows.ipv4SrcAddr,
        minerwa_flows.ipv4DstAddr,
        minerwa_flows.protocol,
        minerwa_detections.event_name,
        minerwa_detections.detector,
        minerwa_detections.metric,
        minerwa_detections.event_time
    FROM minerwa_detections
    JOIN minerwa_flows ON minerwa_flows.id = minerwa_detections.flow_id;
    '''))


def generate_tables_ddl(definiton_path: str, sql_path: str) -> None:
    definition = flow_definition.load(definiton_path)
    ingestion_table = _generate_ingestion_table(definition)
    flow_table = _generate_flow_table(definition)
    materialized_view = _generate_materialized_view()
    detections_table = _generate_detections_table()
    detections_materialized_view = _generate_detections_materialized_view()
    ddl = '\n'.join((ingestion_table, flow_table, materialized_view,
                     detections_table, detections_materialized_view))
    if sql_path == '-':
        sys.stdout.write(ddl)
        return
    with open(sql_path, 'w') as f:
        f.write(ddl)
