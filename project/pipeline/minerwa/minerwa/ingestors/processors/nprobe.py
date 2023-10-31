import logging
from typing import Callable
from uuid import uuid4
import zlib

import orjson as json

from minerwa.model import FlowBase, get_flow_model


class NProbeProcessor:

    def __init__(self, handler: Callable[[FlowBase], None]) -> None:
        self._handler = handler

    async def __call__(self, msg: bytes) -> None:
        try:
            if msg[0] == 0:
                msg = zlib.decompress(msg[1:])
            dataset = json.loads(msg)

            Flow, FlowEnum = get_flow_model()
            for d in dataset:
                flow = Flow(id=uuid4())
                for k, v in d.items():
                    try:
                        flow.set(FlowEnum(k).name, v)
                    except ValueError:
                        ... #logging.info('IPFIX field %s not defined', k)
                await self._handler(flow)
        except Exception as e:
            logging.exception(e)
            raise
