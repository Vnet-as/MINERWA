from oslo_config import cfg

from minerwa.command import commands

CONF = cfg.ConfigOpts()


def _setup_global_opts():
    CONF.register_opt(cfg.BoolOpt('debug'))
    CONF.register_opt(cfg.StrOpt('flow_definition'))
    CONF.register_opt(cfg.URIOpt('nats_uri', schemes=('nats',)))
    CONF.register_opt(cfg.StrOpt('capnp_schema'))
    CONF.register_opt(cfg.IntOpt('processes', default=1))
    CONF.register_opt(cfg.StrOpt('ingestor'))
    CONF.register_opt(cfg.StrOpt('ingestor_processor'))


def _setup_cli_opts():
    CONF.register_cli_opt(cfg.BoolOpt('debug'))
    CONF.register_cli_opt(cfg.SubCommandOpt('command',
                                            title='commands',
                                            handler=_add_command_parsers))


def _add_command_parsers(subparsers):
    for cmd_name, cmd_details in commands.items():
        parser = subparsers.add_parser(cmd_name, help=cmd_details['help'])
        cmd_class = cmd_details['class']
        cmd_class.setup_cli_subparser(parser)


def setup_config():
    _setup_global_opts()
    _setup_cli_opts()
