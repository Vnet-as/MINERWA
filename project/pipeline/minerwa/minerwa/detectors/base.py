import logging

import nats

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

    async def _init_consumer(self):
        self._broker = broker = await nats.connect(CONF.nats_uri)
        cb = None
        if hasattr(self, 'process_flow') and callable(self.process_flow):
            cb = self._process_msg
        return await broker.subscribe('minerwa.ingestion', self.name, cb=cb)

    async def run(self):
        await self._init_consumer()
