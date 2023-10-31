import functools
import orjson as json

import nsq
from oslo_config import cfg
from oslo_config import types as cfg_types

from .main import Datasource


CFG_OPTS = (
    cfg.ListOpt('lookupd_http_addresses',
                cfg_types.URI(schemes=('http', 'https')),
                bounds=True),
    cfg.ListOpt('nsqd_tcp_addresses',
                cfg_types.String(),
                bounds=True),
    cfg.StrOpt('topic'),
    cfg.StrOpt('channel'),
    cfg.IntOpt('max_in_flight', default=500),
)


class NSQDatasource(Datasource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._message_handler = kwargs.pop('message_handler', None)
        self._send_buffer = []

    @classmethod
    def setup_config(cls, conf, group_name: str):
        conf.register_opts(CFG_OPTS, group_name)

    async def _handle_message(self, msg: nsq.Message, handler=None) -> bool:
        await (handler or self._message_handler)(msg.body)
        return True

    def _get_writer(self):
        if self._writer is None:
            writer = nsq.Writer(self.conf.nsqd_tcp_addresses)
            self._writer = writer
        return self._writer

    async def receive(self, handler=None):
        handle_fn = functools.partial(self._handle_message, handler=handler)
        nsq.Reader(message_handler=handle_fn,
                   lookupd_http_addresses=self.conf.lookupd_http_addresses,
                   topic=self.conf.topic, channel=self.conf.channel,
                   max_in_flight=self.conf.max_in_flight)

    def send(self, data, topic: str = None) -> None:
        self._send_buffer.append(data)
        if len(self._send_buffer) > 100:
            writer = self._get_writer()
            writer.mpub(topic or self.conf.topic,
                        [json.dumps(d.to_dict()) for d in self._send_buffer])
            self._send_buffer.clear()
