import logging
from typing import Optional
from uuid import UUID

import nats
import orjson as json

from minerwa.config import CONF
from minerwa.model import FlowBase, get_capnp_schema, get_flow_model
from minerwa.plugin import PluginBase

LOG = logging.getLogger(__name__)


class DetectorBase(PluginBase):

    @classmethod
    def setup_config(cls, conf, conf_group_name: str = None):
        ...

    async def _process_msg(self, msg):
        schema = get_capnp_schema()
        with schema.from_bytes(msg.data) as f:
            await self.process_flow(get_flow_model()[0].from_capnproto(f))

    async def _setup_broker(self):
        return await nats.connect(CONF.nats_uri)

    async def _init_consumer(self):
        self._broker = broker = await self._setup_broker()
        cb = None
        if hasattr(self, 'process_flow') and callable(self.process_flow):
            cb = self._process_msg
        return await broker.subscribe('minerwa.ingestion', self.name, cb=cb)

    async def _save_result(
        self,
        flow_id: str | UUID,
        type_: str,
        coef: float,
    ):
        data = {
            'flow_id': flow_id,
            'detector': self.name,
            'event_name': type_,
            'metric': coef,
        }
        await self._broker.publish('minerwa.detection', json.dumps(data))

    async def run(self):
        await self._init_consumer()
