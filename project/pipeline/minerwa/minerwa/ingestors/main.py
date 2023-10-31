import asyncio
from importlib.metadata import entry_points
import logging
from multiprocessing import Process
import os

import nats
from oslo_config import cfg

from minerwa.model import FlowBase, get_capnp_schema
from minerwa.plugin import PluginFactory


CFG_OPTS = (
    cfg.StrOpt('datasource'),
    cfg.StrOpt('processor'),
    cfg.IntOpt('processes'),
)


class IngestionManager:
    def __init__(self, conf: cfg.ConfigOpts) -> None:
        self._setup_config(conf)
        eps = entry_points(group='minerwa.datasources')
        self.datasource_plugins = {ep.name: ep for ep in eps}

        eps = entry_points(group='minerwa.ingestor_processors')
        self.processor_plugins = {ep.name: ep for ep in eps}

    def _setup_config(self, conf: cfg.ConfigOpts) -> None:
        self.conf = conf
        conf.register_group(cfg.OptGroup(name='ingestor'))
        conf.register_opts(CFG_OPTS, group='ingestor')

    def run(self) -> None:
        process_count = self.conf.ingestor.processes or self.conf.processes
        if process_count > 1:
            pass
        elif process_count == -1:
            process_count = os.cpu_count()
        else:
            self.process()
            return

        self.test = []
        processes = (
            Process(target=self.process) for _ in range(process_count)
        )
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def process(self):
        logging.warning('starting process')
        ingestor = self._get_datasource_plugin()
        processor = self._get_processor_plugin()
        loop = asyncio.new_event_loop()
        loop.create_task(self._init_broker())
        loop.create_task(ingestor.receive(processor))
        loop.run_forever()

    async def _init_broker(self):
        self._broker = await nats.connect(self.conf.nats_uri)

    async def _process_flow(self, flow: FlowBase):
        try:
            await self._broker.publish('minerwa.ingestion',
                                       flow.to_capnproto(get_capnp_schema()).to_bytes())
        except Exception as e:
            logging.error(flow)
            logging.exception(e)

    def _get_datasource_plugin(self):
        plugin_name, config_group = self.conf.ingestor.datasource.split(':')
        ingestor_cls = self.datasource_plugins[plugin_name].load()
        factory = PluginFactory(ingestor_cls, self.conf, config_group)
        return factory.get_instance()

    def _get_processor_plugin(self):
        processor_name = self.conf.ingestor.processor
        processor_cls = self.processor_plugins[processor_name].load()
        return processor_cls(self._process_flow)
