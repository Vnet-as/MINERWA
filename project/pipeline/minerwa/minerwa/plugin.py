import abc
from oslo_config import cfg


class PluginBase(abc.ABC):

    @abc.abstractclassmethod
    def setup_config(cls, conf):
        """Setup fields for option group, always call parent class' setup_config
        if not inheriting directly from this class"""


class PluginFactory:

    def __init__(
            self,
            plugin_cls: PluginBase,
            conf,
            conf_group_name: str = None
    ):
        self.name = conf_group_name or plugin_cls.__class__.__name__.lower()
        self.conf = conf
        self.plugin_cls = plugin_cls
        if conf_group_name:
            conf.register_group(cfg.OptGroup(name=conf_group_name))
            conf.register_opt(cfg.IntOpt('processes'), group=conf_group_name)
            plugin_cls.setup_config(self.conf, conf_group_name)
            self.conf = getattr(conf, conf_group_name)

    @property
    def process_count(self):
        return getattr(self.conf, 'processes', 1)

    def get_instance(self) -> PluginBase:
        inst = self.plugin_cls()
        inst.conf = self.conf
        inst.name = self.name
        return inst
