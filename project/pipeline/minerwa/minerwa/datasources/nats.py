import functools

import nats
from oslo_config import cfg
from oslo_config import types as cfg_types

from .main import Datasource


CFG_OPTS = (
    cfg.ListOpt('servers',
                cfg_types.URI(schemes=('nats')),
                bounds=True),
    cfg.StrOpt('publish_subject'),
)


class NATSDatasource(Datasource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._connection = None
        self._message_handler = kwargs.pop('message_handler', None)

    def _setup_config(self, conf, group_name):
        conf.register_opts(CFG_OPTS, group=group_name)

    async def _handle_message(self, msg: bytes, handler=None) -> bool:
        (handler or self._message_handler)(msg.body)
        return True

    async def _get_connection(self):
        if self._connection is None:
            conn = await nats.connect(self.conf.servers)
            self._connection = conn
        return self._connection

    async def receive(self, handler=None, subject: str = None) -> None:
        handle_fn = functools.partial(self._handle_message, handler=handler)
        conn = await self._get_connection()
        await conn.subscribe(subject or self.conf.subscribe_subject, cb=handle_fn)

    async def send(self, data, subject: str = None) -> None:
        conn = await self._get_connection()
        await conn.publish(subject or self.conf.publish_subject, data)
