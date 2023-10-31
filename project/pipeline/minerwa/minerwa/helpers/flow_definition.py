from collections import namedtuple

import yaml


FieldDefinition = namedtuple('FieldDefinition', ['id', 'name', 'type'])


def load(path: str) -> list[FieldDefinition]:
    with open(path, 'r') as f:
        definition = yaml.safe_load(f)
        definition_list = [FieldDefinition(**field) for field in definition]
    return definition_list
