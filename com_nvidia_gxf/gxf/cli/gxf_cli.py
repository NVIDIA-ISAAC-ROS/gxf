"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
import argparse
from collections import namedtuple
import sys
from typing import Dict
from result import Result, Ok, Err
from gxf.cli.service_handler import *
from gxf.cli.version import *
from gxf.cli.transport import Transport
from gxf.cli.http_transport import HttpTransport
from gxf.cli.grpc_transport import GrpcTransport
from gxf.cli.renderer import StringTableDataSource, CompoundStringTableDataSource, UrwidRenderer

DEBUG = True
VERSION = 0.1

class ExceptionParsing(Exception):
    pass

class ExceptionHandling(Exception):
    pass

class BaseModule:
    """base class to execute commands"""
    NAME=''
    HELP=''
    PARAMS={}
    FLAGS=[]
 
    Flag = namedtuple("Flag", "name full_name help")

    @classmethod
    def add_parser(cls, sub_parsers):
        """add a sub parser for this module"""
        parser = sub_parsers.add_parser(cls.NAME, help=cls.HELP)
        parser.set_defaults(parser_name=cls.NAME)
        for k, v in cls.PARAMS.items():
            parser.add_argument(k, help=v)
        for flag in cls.FLAGS:
            parser.add_argument(flag.name, flag.full_name, help=flag.help)

    @classmethod
    def wrap_params(cls, args) -> List:
        """ wrap all the params to a string list"""
        params = []
        for key in cls.PARAMS:
            if not key in args:
                return Err("{key} is required")
            params.append(args[key])
        return params

class DumpModule(BaseModule):
    """ class to handle 'dump' command"""
    NAME = 'dump'
    HELP = "dump the information from a running graph"
    PARAMS = {}
    FLAGS = [
        BaseModule.Flag("-u", "--uid", "filter the graph data by uid")
    ]

    def execute(self, args) -> Result:
        transport = create_transport(args["transport"], args["server"])
        if not transport:
            return Err("Unable to dump graph without a proper transport")
        # create the service handler used to request data from the server and
        # parse the response
        params = [args['uid']] if args['uid'] else ['*']
        res = DumpServiceHandler(transport).request(params)
        if res.is_ok():
            print(res.value)
        return res

class StatisticModule(BaseModule):
    """
    class to handle 'stat' command
    """
    NAME = 'stat'
    HELP = "actions on statistics"
    PARAMS = {
        "target": "the target on which statistic is collected",
    }
    FLAGS = [
        BaseModule.Flag("-u", "--uid", "filter the statistic data by uid")
    ]
    FORMAT = {
        "entity":  [
            # 'key', 'display text', 'max length'
            ('uid', 'ID', 5),
            ('name', 'NAME', 15),
            ('status', 'STATUS', 15),
            ('count', 'TICKS', 10),
            ('load_percentage', 'LOAD(%)', 15),
            ('execution_time_total_ms', 'TOTAL EXECUTION(ms)', 25),
            ('execution_time_median_ms', 'MIDIAN EXECUTION(ms)', 25),
            ('execution_time_90_ms', '90% EXECUTION(ms)', 25),
            ('execution_time_max_ms', 'MAX EXECUTION(ms)', 24),
            ('variation_median_ms', 'MEDIAN DELAY(ms)', 20),
            ('variation_90_ms', '90% DELAY(ms)', 20),
            ('variation_max_ms','MAX DELAY(ms)', 20)
        ],
        "codelet": [
            # 'key', 'display text', 'max length'
            ('uid', 'ID', 5),
            ('name', 'NAME', 20),
            ('entity', 'ENTITY', 15),
            ('ticks', 'TICKS', 10),
            ('type', 'TYPE', 30),
            ('execution_time_mean_ms', 'AVERAGE EXECUTION(ms)', 25),
            ('execution_time_90_ms', '90% EXECUTION(ms)', 25),
            ('execution_time_max_ms', 'MAX EXECUTION(ms)', 24),
            ('tick_frequency_per_ms', 'FREQ(per ms)', 10)
        ],
        "event": [
            # 'key', 'display text', 'max length'
            ('timestamp', "TIME", 30),
            ('state', "STATE", 25)
        ],
        "term": [
            # 'key', 'display text', 'max length'
            ('state', "STATE", 35),
            ('count', "COUNT", 10),
            ('time_median_ms', "MEDIAN TIME(ms)", 16),
            ('time_90_ms', "90% TIME(ms)", 15)
        ]
    }

    def __init__(self):
        self._service_handler = None
        self._renderer = None
        self._service_data = [
            StringTableDataSource(
                name="entity",
                desc="List of active entities",
                header=self.FORMAT['entity'],
                update_fn=lambda filter: self._get_data('entity', filter),
                filterable=['uid', 'name', 'status']
            ),
            StringTableDataSource(
                name="codelet",
                desc="List of codelets",
                header=self.FORMAT['codelet'],
                update_fn=lambda filter: self._get_data('codelet', filter),
                filterable=['uid', 'name', 'entity']
            ),
            CompoundStringTableDataSource(
                name="event",
                desc="History of scheduling term event",
                header=self.FORMAT['event'],
                update_fn=lambda filter: self._get_compound_data('event', filter),
                filterable=['uid']
            ),
            CompoundStringTableDataSource(
                name="term",
                desc="Statistics of scheduling term event",
                header=self.FORMAT['term'],
                update_fn=lambda filter: self._get_compound_data('term', filter),
                filterable=['uid', 'name']
            )
        ]
        self._extra_flags = []

    def _get_data(self, name: str, filter=None):
        stat_data = []
        params = [name]
        if self._extra_flags:
            params += self._extra_flags
        res = self._service_handler.request(params)
        if res.is_ok():
            for item in res.value:
                if filter and filter[0] in item and filter[1] != '*' and str(item[filter[0]]) != filter[1]:
                    continue
                data = []
                for key, _, maxlen in self.FORMAT[name]:
                    if key not in item:
                        data.append(('', maxlen))
                    else:
                        data.append((item[key], maxlen))
                stat_data.append(data)
        return stat_data

    def _get_compound_data(self, name: str, filter=None):
        compound_data = dict()
        params = [name]
        if self._extra_flags:
            params += self._extra_flags
        res = self._service_handler.request(params)
        if res.is_ok():
            for item in res.value:
                if filter and filter[0] in item and filter[1] != '*' and str(item[filter[0]]) != filter[1]:
                    continue
                terms_data = dict()
                for term in item['data']:
                    stat_data = []
                    for term_item in term['data']:
                        data = []
                        for k, _, maxlen in self.FORMAT[name]:
                            if k not in term_item:
                                data.append(('', maxlen))
                            else:
                                data.append((term_item[k], maxlen))
                        stat_data.append(data)
                    key = f"{term['type']}[{term['uid']}]"
                    terms_data[key] = stat_data
                key = f"ENTITY NAME: {item['name']} [ID: {item['uid']}]"
                compound_data[key] = terms_data
        return compound_data

    def execute(self, args) -> Result:
        transport = create_transport(args["transport"], args["server"])
        if not transport:
            return Err("Statistic not available without a proper transport")
        # create the service handler used to request data from the server and
        # parse the response
        self._service_handler = StatServiceHandler(transport)
        # find the data source for rendering
        data_name = args["target"]
        data_source = next((i for i in self._service_data if i.name == data_name), None)
        if data_source is None:
            return Err(f"Statistic data for {data_name} not supported")
        if args["uid"]:
            self._extra_flags.append(args["uid"])
        # create the renderer to render the formatted data
        self._renderer = UrwidRenderer(data_source)
        self._renderer.start()
        return Ok('')

class ConfigModule(BaseModule):
    """class to handle 'config' command"""
    NAME = 'config'
    PARAMS = {
        "cid": "id of the component on which the configuration is performed",
        "key": "name of the configuration",
        "value": "new value to update the configuration"
    }

    def execute(self, args) -> Result:
        transport = create_transport(args["transport"], args["server"])
        if not transport:
            return Err("Unable to perform remote configuration without a proper transport")
        # create the service handler used to request data from the server and
        # parse the response
        res = ConfigServiceHandler(transport).request(self.wrap_params(args))
        if res.is_ok():
            return Ok("Configuration is successful")
        return res

class CLI:
    """
    Main class to process the commands, which parse the command line parameters
    and route the command to the corresponding module
    """
    MODULES = [StatisticModule, ConfigModule, DumpModule]

    def handle_version(self) -> Result:
        version = str(f"CLI version: v{str(CLI_VERSION)}\n")
        return Ok(version)

    def execute(self, args: Dict):
        if args["version"]:
            res = self.handle_version()
            if res.is_ok():
                print(res.value)
            return

        # extract the sub-parser name
        parser_name = args["parser_name"]
        if not parser_name:
            raise ExceptionParsing("Missing a command")
        module = next((m for m in self.MODULES if m.NAME == parser_name), None)
        if module is None:
            raise ExceptionParsing(f"Invalid command: {module}")

        res = module().execute(args)
        if res.is_err():
            raise ExceptionHandling(res.value)

def create_transport(implementation:str, server:str) -> Transport:
    transport = None
    if implementation == "http":
        transport = HttpTransport(server)
    elif implementation == "grpc":
        transport = GrpcTransport(server)
    return transport

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--version",
        dest="version",
        action="store_true",
        help="print cli tool version"
    )
    parser.add_argument(
        "-s",
        "--server",
        action="store",
        default="localhost:8000",
        help="specify the server name or address"
    )
    parser.add_argument(
        "-t",
        "--transport",
        action="store",
        default="http",
        help="specify the transport between clients and the server"
    )
    # subparser for supported modules
    parser.set_defaults(parser_name='')
    sub_parsers = parser.add_subparsers(metavar="")
    for module in CLI.MODULES:
        module.add_parser(sub_parsers)

    return parser

if __name__ == "__main__":
    parser = make_parser()
    try:
        CLI().execute(vars(parser.parse_args(sys.argv[1:])))
    except ExceptionHandling as e:
        print(f"Failed to execute the command: {e}")
        sys.exit(1)
    except ExceptionParsing as e:
        print(f"Parsing failed: {e}\n")
        parser.print_help()
        sys.exit(1)

