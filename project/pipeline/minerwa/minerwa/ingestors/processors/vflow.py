from datetime import datetime
from functools import partial
from ipaddress import IPv4Address, IPv6Address
import logging
from typing import Callable
from uuid import uuid4

import orjson as json

from minerwa.model import Flow, NF9FieldMapping


class VFlowProcessor:
    CONVERSION_FUNCTIONS = {
        1: partial(int, base=16),
        2: partial(int, base=16),
        6: partial(int, base=16),
        8: IPv4Address,
        12: IPv4Address,
        18: IPv4Address,
        21: datetime.fromtimestamp,
        22: datetime.fromtimestamp,
        27: IPv6Address,
        28: IPv6Address,
        63: IPv6Address,
        89: partial(int, base=16),
    }

    def __init__(self, handler: Callable[[Flow], None]) -> None:
        self._handler = handler

    async def __call__(self, msg: str) -> None:
        try:
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                data = json.loads(msg.replace(b'\x00', b''))
            bundle_time = data['Header']['UNIXSecs']
            uptime = data['Header']['SysUpTime'] / 1000
            ref_timestamp = bundle_time - uptime
            for d in data['DataSets']:
                preprocessed = {k['I']: k['V'] for k in d}
                flow = Flow(id=uuid4())
                for k, v in preprocessed.items():
                    if k in (21, 22):
                        v = ref_timestamp + v / 1000
                    if k in self.CONVERSION_FUNCTIONS:
                        v = self.CONVERSION_FUNCTIONS[k](v)
                    try:
                        setattr(flow, NF9FieldMapping(k).name.lower(), v)
                    except ValueError:
                        # field not defined in mapping, skip
                        continue
                    except Exception as e:
                        logging.exception(e)
                await self._handler(flow)
        except Exception as e:
            logging.exception(e)
            raise
