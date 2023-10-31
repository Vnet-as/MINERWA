import random
import sys

from minerwa.helpers import flow_definition
from minerwa.model import camel_case


TYPE_MAP = {
    'int': 'UInt32',
    'uint8': 'UInt8',
    'uint16': 'UInt16',
    'uint32': 'UInt32',
    'uint64': 'UInt64',
    'ipv4': 'UInt32',
    'ipv6': 'Data',
    'str': 'Text',
}


def generate_capnproto_schema(definition_path: str, schema_path: str) -> None:
    definition = flow_definition.load(definition_path)
    fields = (('id', 'Data'),)
    fields += tuple(
        (camel_case(field.name), TYPE_MAP[field.type])
        for field in definition
    )
    schema_id = hex(random.getrandbits(64) | (1 << 64))
    schema_start = f'@{schema_id};\n\nstruct Flow {{\n\t'
    schema_body = '\n\t'.join(f'{name} @{i} :{type};'
                              for i, (name, type) in enumerate(fields))
    schema_end = "\n}"
    schema = schema_start + schema_body + schema_end

    if schema_path == '-':
        sys.stdout.write(schema)
        return
    with open(schema_path, 'w') as f:
        f.write(schema)
