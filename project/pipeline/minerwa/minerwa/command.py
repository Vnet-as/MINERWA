import argparse

from oslo_config import cfg

from minerwa import config


class CommandBase():
    def _setup_config(self, conf: cfg.ConfigOpts) -> None:
        ...

    @classmethod
    def setup_cli_subparser(cls, subparser: argparse.ArgumentParser) -> None:
        ...

    def run(self) -> None:
        ...


class IngestorCommand(CommandBase):
    def run(self) -> None:
        from minerwa.ingestors import IngestionManager
        IngestionManager(config.CONF).run()


class DetectorCommand(CommandBase):
    def run(self) -> None:
        from minerwa.detectors import DetectionManager
        DetectionManager(config.CONF).run()


class GenerateCapnprotoCommand(CommandBase):
    @classmethod
    def setup_cli_subparser(cls, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument('--schema-output', metavar='FILE',
                               required=True,
                               help='output file for generated schema')

    def run(self) -> None:
        from minerwa.utils.schema import generate_capnproto_schema
        generate_capnproto_schema(
            config.CONF.flow_definition,
            config.CONF.command.schema_output
        )


class GenerateClickhouseDDL(CommandBase):
    @classmethod
    def setup_cli_subparser(cls, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument('--sql-output', metavar='FILE',
                               required=True,
                               help='output DDL SQL file ')

    def run(self) -> None:
        from minerwa.utils.clickhouse import generate_tables_ddl
        generate_tables_ddl(
            config.CONF.flow_definition,
            config.CONF.command.sql_output
        )


commands = {
    'ingestor': {
        'class': IngestorCommand,
        'help': 'Run ingestor'
    },
    'detector': {
        'class': DetectorCommand,
        'help': 'Run detector'
    },
    'gen_capnproto': {
        'class': GenerateCapnprotoCommand,
        'help': 'Generate CapnProto schema'
    },
    'gen_clickhouse_ddl': {
        'class': GenerateClickhouseDDL,
        'help': 'Generate DDL SQL for Clickhouse'
    }
}
