from abc import abstractmethod

from minerwa.plugin import PluginBase


class Datasource(PluginBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    async def send(self, data) -> None:
        ...

    @abstractmethod
    async def receive(self, handler=None) -> None:
        ...
