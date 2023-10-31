import logging
import sys

from minerwa import config
from minerwa.command import commands


logging.basicConfig(level=logging.INFO)


def main():
    cmd = commands[config.CONF.command.name]['class']
    cmd_inst = cmd()
    cmd_inst.run()


if __name__ == '__main__':
    config.setup_config()
    config.CONF(sys.argv[1:], 'minerwa')
    main()
