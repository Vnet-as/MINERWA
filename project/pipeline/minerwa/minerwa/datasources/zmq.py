import asyncio

from oslo_config import cfg
import zmq
import zmq.asyncio

from .main import Datasource


CFG_OPTS = (
    cfg.URIOpt('publisher_uri', schemes=('tcp', 'ipc')),
    cfg.StrOpt('topic', default='flow'),
    cfg.IntOpt('part', default=1)
)


class ZMQDatasource(Datasource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._message_handler = kwargs.pop('message_handler', None)

    @classmethod
    def setup_config(cls, conf, group_name: str):
        conf.register_opts(CFG_OPTS, group_name)

    async def _handle_message(self, msg, handler=None):
        part = msg if self.conf.part == 0 else msg[self.conf.part]
        await (handler or self._message_handler)(part)

    async def send(self, data) -> None:
        ...

    async def receive(self, handler=None):
        await asyncio.sleep(1)
        context = zmq.asyncio.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(self.conf.publisher_uri)
        if self.conf.topic:
            socket.setsockopt_string(zmq.SUBSCRIBE, self.conf.topic)
        while True:
            msg = await socket.recv_multipart()
            await self._handle_message(msg, handler)
        socket.close()
