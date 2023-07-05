import argparse
import hashlib
import json
import os
from os.path import abspath
import re
import sys
from datetime import datetime
from multiprocessing import cpu_count
from subprocess import run
import numpy as np


def parse_arguments():
    '''
    Parse arguments from command line.
    '''
    parser = argparse.ArgumentParser()

    # DUT
    parser.add_argument(
        '--design',
        type=str,
        metavar='<gcd,jpeg,ibex,aes,...>',
        required=True,
        help='Name of the design for Autotuning.')
    parser.add_argument(
        '--platform',
        type=str,
        metavar='<sky130hd,sky130hs,asap7,...>',
        required=True,
        help='Name of the platform for Autotuning.')

    # Experiment Setup
    parser.add_argument(
        '--config',
        type=str,
        metavar='<path>',
        required=True,
        help='Configuration file that sets which knobs to use for Autotuning.')
    parser.add_argument(
        '--experiment',
        type=str,
        metavar='<str>',
        default='test',
        help='Experiment name. This parameter is used to prefix the'
        ' FLOW_VARIANT and to set the Ray log destination.')

    # Setup
    parser.add_argument(
        '--git-clean',
        action='store_true',
        help='Clean binaries and build files.'
             ' WARNING: may lose previous data.'
             ' Use carefully.')
    parser.add_argument(
        '--git-clone',
        action='store_true',
        help='Force new git clone.'
             ' WARNING: may lose previous data.'
             ' Use carefully.')
    parser.add_argument(
        '--git-clone-args',
        type=str,
        metavar='<str>',
        default='',
        help='Additional git clone arguments.')
    parser.add_argument(
        '--git-latest',
        action='store_true',
        help='Use latest version of OpenROAD app.')
    parser.add_argument(
        '--git-or-branch',
        type=str,
        metavar='<str>',
        default='',
        help='OpenROAD app branch to use.')
    parser.add_argument(
        '--git-orfs-branch',
        type=str,
        metavar='<str>',
        default='master',
        help='OpenROAD-flow-scripts branch to use.')
    parser.add_argument(
        '--build-args',
        type=str,
        metavar='<str>',
        default='',
        help='Additional arguments given to ./build_openroad.sh.')

    # Workload
    parser.add_argument(
        '--timeout',
        type=int,
        metavar='<int>',
        default=3600,
        help='Timeout for openroad.')
    parser.add_argument(
        '--jobs',
        type=int,
        metavar='<int>',
        default=int(np.floor(cpu_count() / 2)),
        help='Max number of concurrent jobs.')
    parser.add_argument(
        '--openroad-threads',
        type=int,
        metavar='<int>',
        default=16,
        help='Max number of threads openroad can use.')
    parser.add_argument(
        '--server',
        type=str,
        metavar='<ip|servername>',
        default=None,
        help='The address of Ray server to connect.')
    parser.add_argument(
        '--port',
        type=int,
        metavar='<int>',
        default=10001,
        help='The port of Ray server to connect.')

    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Verbosity level.\n\t0: only print Ray status\n\t1: also print'
        ' training stderr\n\t2: also print training stdout.')

    arguments = parser.parse_args()

    arguments.experiment += f'-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    return arguments