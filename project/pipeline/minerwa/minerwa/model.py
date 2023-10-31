from dataclasses import dataclass, field, fields, make_dataclass
from datetime import datetime
from enum import Enum
from functools import cache
from ipaddress import IPv4Address, IPv6Address
import logging
import re
from typing import Any, Self, get_args
from uuid import UUID, uuid4

import capnp
import yaml

from minerwa.config import CONF


@dataclass(kw_only=True, slots=True)
class FlowBase:
    """Internal representation of IPFIX flow
    according to RFC 7012 (IPFIX)
    """
    id: UUID = field(default_factory=uuid4)

    def set(self, field: str, value: Any) -> None:
        setattr(self, field, self._get_field_type(field)(value))

    def to_capnproto(self, cnp):
        map_ = {
            UUID: lambda v: v.bytes,
            datetime: lambda v: v.timestamp(),
            IPv4Address: int,
            IPv6Address: lambda v: v.packed
        }
        msg = cnp.new_message()
        for f in fields(self):
            val = getattr(self, f.name)
            conv_fn = map_.get(type(val), lambda v: v)
            try:
                setattr(msg, f.name, conv_fn(val))
            except Exception:
                logging.error(
                    'Failed to convert to CapNProto - field: %s, value: %s, type: %s, conv_fn: %s, conv_val: %s',
                    f.name, val, type(val), conv_fn, conv_fn(val))

        return msg

    @classmethod
    def from_capnproto(cls, cnp) -> Self:
        map_ = {
            UUID: lambda v: UUID(bytes=v),
            datetime: datetime.fromtimestamp,
            IPv4Address: IPv4Address,
            IPv6Address: lambda v: IPv6Address(v),
        }

        try:
            return cls(**{
                f.name: map_.get(cls._get_field_type(f.name),
                                 lambda v: v)(getattr(cnp, f.name))
                for f in fields(cls)
                if getattr(cnp, f.name)})
        except AttributeError:
            print(cnp)

    @classmethod
    @cache
    def _get_field_type(cls, field: str):
        f = cls.__dataclass_fields__[field]
        return (get_args(f.type) + (f.type,))[0]


def camel_case(s: str) -> str:
    s = re.sub('[^A-Za-z0-9]', ' ', s)
    s = ''.join(w.title() if i > 0 else w.lower()
                for i, w in enumerate(s.split()))
    return s


@cache
def get_flow_model() -> (FlowBase, Enum):
    TYPE_MAP = {
        'int': (int, 0),
        'uint8': (int, 0),
        'uint16': (int, 0),
        'uint32': (int, 0),
        'uint64': (int, 0),
        'ipv4': (IPv4Address, IPv4Address(0)),
        'ipv6': (IPv6Address, IPv6Address(0)),
        'str': (str, '')
    }

    with open(CONF.flow_definition, 'r') as f:
        definition = yaml.safe_load(f)

    fields = []
    for d in definition:
        type_ = TYPE_MAP[d['type']]
        f = (camel_case(d['name']), type_[0], field(default=type_[1]))
        fields.append(f)

    Flow = make_dataclass('Flow', fields, bases=(FlowBase,),
                          kw_only=True, slots=True)

    FlowEnum = Enum('FlowEnum',
                    tuple((camel_case(d['name']), d['id'])
                          for d in definition))
    return Flow, FlowEnum


@cache
def get_capnp_schema():
    capnp.remove_import_hook()
    return capnp.load(CONF.capnp_schema).Flow
