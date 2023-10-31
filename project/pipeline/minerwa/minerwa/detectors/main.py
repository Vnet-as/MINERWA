import asyncio
from importlib.metadata import entry_points
import logging
from multiprocessing import Process

from oslo_config import cfg
from oslo_config import types as cfg_types

from minerwa.plugin import PluginFactory
from .base import DetectorBase


CFG_OPTS = (
    cfg.ListOpt('detectors', cfg_types.String(), bounds=True),
)


class DetectionManager:
    def __init__(self, conf: cfg.ConfigOpts) -> None:
        self._setup_config(conf)
        eps = entry_points(group='minerwa.detectors')
        self.detector_plugins = {ep.name: ep for ep in eps}

    def _setup_config(self, conf: cfg.ConfigOpts) -> None:
        self.conf = conf
        conf.register_group(cfg.OptGroup(name='detector'))
        conf.register_opts(CFG_OPTS, group='detector')

    def _get_detector_factories(self):
        for detector in self.conf.detector.detectors:
            plugin_name, config_group = detector.split(':')
            detector_cls = self.detector_plugins[plugin_name].load()
            factory = PluginFactory(detector_cls, self.conf, config_group)
            yield factory

    def run(self) -> None:
        processes = []
        for df in self._get_detector_factories():
            p_list = []
            for _ in range(df.process_count):
                p = Process(target=self._process, args=(df.get_instance(),))
                p_list.append(p)
            processes.extend(p_list)
        for p in processes:
            p.start()
        for p in processes:
            p.join

    @staticmethod
    def _process(detector: DetectorBase):
        loop = asyncio.get_event_loop()
        loop.create_task(detector.run())
        loop.run_forever()
