import os
import sys
import argparse
import hashlib
import json
from os.path import abspath
import re
import math
import random
from datetime import datetime
from multiprocessing import cpu_count
from subprocess import run
import numpy as np

OPENROAD_FLOW_PATH = "/home/szheng22/tools/OpenROAD-flow-scripts"

def read_config(file_name):
    def read(path):
        with open(abspath(path), 'r') as file:
            ret = file.read()
        return ret

    with open(file_name) as file:
        data = json.load(file)
    sdc_file = ''
    fr_file = ''
    config = dict()
    for key, value in data.items():
        if key == '_SDC_FILE_PATH':
            if value != '': 
                if sdc_file != '':
                    print('[WARNING TUN-0004] Overwriting SDC base file.')
                sdc_file = read(f'{os.path.dirname(file_name)}/{value}')
            continue
        if key == '_FR_FILE_PATH':
            if value != '': 
                if fr_file != '':
                    print('[WARNING TUN-0005] Overwriting FastRoute base file.')
                fr_file = read(f'{os.path.dirname(file_name)}/{value}')
            continue
        config[key] = value
    
    return config, sdc_file, fr_file


def write_sdc(variables, path, SDC_ORIGINAL, CONSTRAINTS_SDC):
    new_file = SDC_ORIGINAL
    for key, value in variables.items():
        if key == 'CLK_PERIOD':
            if new_file.find('set clk_period') != -1:
                new_file = re.sub(r'set clk_period .*\n(.*)',
                                  f'set clk_period {value}\n\\1',
                                  new_file)
            else:
                new_file = re.sub(r'-period [0-9\.]+ (.*)',
                              f'-period {value} \\1',
                              new_file)
                new_file = re.sub(r'-waveform [{}\s0-9\.]+[\s|\n]',
                              '',
                              new_file)
        elif key == 'UNCERTAINTY':
            if new_file.find('set uncertainty') != -1:
                new_file = re.sub(r'set uncertainty .*\n(.*)',
                                  f'set uncertainty {value}\n\\1',
                                  new_file)
            else:
                new_file += f'\nset uncertainty {value}\n'
        elif key == "IO_DELAY":
            if new_file.find('set io_delay') != -1:
                new_file = re.sub(r'set io_delay .*\n(.*)',
                                  f'set io_delay {value}\n\\1',
                                  new_file)
            else:
                new_file += f'\nset io_delay {value}\n'
    file_name = path + f'/{CONSTRAINTS_SDC}'
    with open(file_name, 'w') as file:
        file.write(new_file)
    return file_name


def write_fast_route(variables, path, FR_ORIGINAL, FASTROUTE_TCL):
    '''
    Create a FastRoute Tcl file with parameters for current tuning iteration.
    '''
    # TODO: handle case where the reference file does not exist
    layer_cmd = 'set_global_routing_layer_adjustment'
    new_file = FR_ORIGINAL
    for key, value in variables.items():
        if key.startswith('LAYER_ADJUST'):
            layer = key.lstrip('LAYER_ADJUST')
            # If there is no suffix (i.e., layer name) apply adjust to all
            # layers.
            if layer == '':
                new_file += '\nset_global_routing_layer_adjustment'
                new_file += ' $::env(MIN_ROUTING_LAYER)'
                new_file += '-$::env(MAX_ROUTING_LAYER)'
                new_file += f' {value}'
            elif re.search(f'{layer_cmd}.*{layer}', new_file):
                new_file = re.sub(f'({layer_cmd}.*{layer}).*\n(.*)',
                                  f'\\1 {value}\n\\2',
                                  new_file)
            else:
                new_file += f'\n{layer_cmd} {layer} {value}\n'
        elif key == 'GR_SEED':
            new_file += f'\nset_global_routing_random -seed {value}\n'
    file_name = path + f'/{FASTROUTE_TCL}'
    with open(file_name, 'w') as file:
        file.write(new_file)
    return file_name


def parse_config(config, path, SDC_ORIGINAL, CONSTRAINTS_SDC, FR_ORIGINAL, FASTROUTE_TCL):
    options = ''
    sdc = {}
    fast_route = {}
    for key, value in config.items():
        # Keys that begin with underscore need special handling.
        if key.startswith('_'):
            # Variables to be injected into fastroute.tcl
            if key.startswith('_FR_'):
                fast_route[key.replace('_FR_', '', 1)] = value
            # Variables to be injected into constraints.sdc
            elif key.startswith('_SDC_'):
                sdc[key.replace('_SDC_', '', 1)] = value
            # Special substitution cases
            elif key == "_PINS_DISTANCE":
                options += f' PLACE_PINS_ARGS="-min_distance {value}"'
            elif key == "_SYNTH_FLATTEN":
                print('[WARNING TUN-0013] Non-flatten the designs are not '
                      'fully supported, ignoring _SYNTH_FLATTEN parameter.')
        # Default case is VAR=VALUE
        else:
            options += f' {key}={value}'
    if bool(sdc):
        write_sdc(sdc, path, SDC_ORIGINAL, CONSTRAINTS_SDC)
        options += f' SDC_FILE={os.path.abspath(f"{path}/{CONSTRAINTS_SDC}")}'
    if bool(fast_route):
        write_fast_route(fast_route, path, FR_ORIGINAL, FASTROUTE_TCL)
        options += f' FASTROUTE_TCL={os.path.abspath(f"{path}/{FASTROUTE_TCL}")}'
    return options


def run_command(cmd, stderr_file=None, stdout_file=None, fail_fast=False, verbose=0):
    '''
    Wrapper for subprocess.run
    Allows to run shell command, control print and exceptions.
    '''
    process = run(cmd, capture_output=True, text=True, check=False, shell=True)
    if stderr_file is not None:
        with open(stderr_file, 'a') as file:
            file.write(f'\n\n{cmd}\n{process.stderr}')
    if stdout_file is not None:
        with open(stdout_file, 'a') as file:
            file.write(f'\n\n{cmd}\n{process.stdout}')
    if verbose >= 1:
        print(process.stderr)
    if verbose >= 2:
        print(process.stdout)

    if fail_fast and process.returncode != 0:
        raise RuntimeError


def get_flow_variant(param, args):
    variant_hash = hashlib.md5(f"{param}".encode('utf-8')).hexdigest()
    with open(os.path.join(os.getcwd(), 'variant_hash.txt'), 'w') as file:
        file.write(variant_hash)
    return f'variant-{variant_hash}'


def baseline(base_dir, path, args):
    # Make sure path ends in a slash, i.e., is a folder
    flow_variant = 'baseline'
    os.system(f'mkdir -p {path}')
    report_path = f"{base_dir}/flow/results/{args.platform}/{args.design}/{flow_variant}/"
    log_path = path

    print(f'[RUN]: {flow_variant}')

    make_command = ""
    make_command += f'make -C {base_dir}/flow DESIGN_CONFIG=designs/'
    make_command += f'{args.platform}/{args.design}/config.mk'
    make_command += f' FLOW_VARIANT={flow_variant}'
    make_command += f' NPROC=4 SHELL=bash'
    run_command(make_command,
                stderr_file=f'{log_path}/error-make-finish.log',
                stdout_file=f'{log_path}/make-finish-stdout.log')

    metrics_file = os.path.join(report_path, 'metrics.json')
    metrics_command = ""
    metrics_command += f'{base_dir}/flow/util/genMetrics.py -x'
    metrics_command += f' -v {flow_variant}'
    metrics_command += f' -d {args.design}'
    metrics_command += f' -p {args.platform}'
    metrics_command += f' -o {metrics_file}'
    run_command(metrics_command,
                stderr_file=f'{log_path}/error-metrics.log',
                stdout_file=f'{log_path}/metrics-stdout.log')

    return metrics_file


def openroad(base_dir, config, path, args, SDC_ORIGINAL, CONSTRAINTS_SDC, FR_ORIGINAL, FASTROUTE_TCL):
    # Make sure path ends in a slash, i.e., is a folder
    flow_variant = get_flow_variant(config, args)
    os.system(f'mkdir -p {path}')
    report_path = f"{base_dir}/flow/results/{args.platform}/{args.design}/{flow_variant}/"
    log_path = path

    print(f'[RUN]: {flow_variant}')

    parameters = parse_config(config, log_path, \
                              SDC_ORIGINAL, CONSTRAINTS_SDC, FR_ORIGINAL, FASTROUTE_TCL)

    make_command = ""
    make_command += f'make -C {base_dir}/flow DESIGN_CONFIG=designs/'
    make_command += f'{args.platform}/{args.design}/config.mk'
    make_command += f' FLOW_VARIANT={flow_variant} {parameters}'
    make_command += f' NPROC=4 SHELL=bash'
    run_command(make_command,
                stderr_file=f'{log_path}/error-make-finish.log',
                stdout_file=f'{log_path}/make-finish-stdout.log')

    metrics_file = os.path.join(report_path, 'metrics.json')
    metrics_command = ""
    metrics_command += f'{base_dir}/flow/util/genMetrics.py -x'
    metrics_command += f' -v {flow_variant}'
    metrics_command += f' -d {args.design}'
    metrics_command += f' -p {args.platform}'
    metrics_command += f' -o {metrics_file}'
    run_command(metrics_command,
                stderr_file=f'{log_path}/error-metrics.log',
                stdout_file=f'{log_path}/metrics-stdout.log')

    return metrics_file


def set_best_params(platform, design, AUTOTUNER_BEST):
    params = []
    best_param_file = f'designs/{platform}/{design}/{AUTOTUNER_BEST}'
    if os.path.isfile(best_param_file):
        with open(best_param_file) as file:
            params = json.load(file)
    return params


def read_metrics(file_name):
    with open(file_name) as file:
        data = json.load(file)
    clk_period = 9999999
    worst_slack = 'ERR'
    wirelength = 'ERR'
    num_drc = 'ERR'
    total_power = 'ERR'
    core_util = 'ERR'
    place_util = 'ERR'
    final_util = 'ERR'
    final_area = 'ERR'
    for stage, value in data.items():
        if stage == 'constraints' and len(value['clocks__details']) > 0:
            clk_period = float(value['clocks__details'][0].split()[1])
        if stage == 'floorplan' \
                and 'design__instance__utilization' in value:
            core_util = value['design__instance__utilization']
        if stage == 'placeopt' \
                and 'design__instance__utilization' in value:
            place_util = value['design__instance__utilization']
        if stage == 'detailedroute' and 'route__drc_errors' in value:
            num_drc = value['route__drc_errors']
        if stage == 'detailedroute' and 'route__wirelength' in value:
            wirelength = value['route__wirelength']
        if stage == 'finish' and 'timing__setup__ws' in value:
            worst_slack = value['timing__setup__ws']
        if stage == 'finish' and 'power__total' in value:
            total_power = value['power__total']
        if stage == 'finish' and 'design__instance__utilization' in value:
            final_util = value['design__instance__utilization']
        if stage == 'finish' and 'design__instance__area' in value:
            final_area = value['design__instance__area']
    ret = {
        "clk_period": clk_period,
        "worst_slack": worst_slack,
        "wirelength": wirelength,
        "num_drc": num_drc,
        "total_power": total_power,
        "core_util": core_util,
        "place_util": place_util,
        "final_util": final_util, 
        "final_area": final_area
    }
    return ret

# from parsearg import *
FASTROUTE_TCL = 'fastroute.tcl'
CONSTRAINTS_SDC = 'constraint.sdc'
if __name__ == "__main__": 
    args = parse_arguments()
    config, SDC_ORIGINAL, FR_ORIGINAL = read_config(abspath(args.config))

    variables = {}
    for key in config.keys(): 
        assert config[key]['type'] in ['int', 'float'], "Unsupported type: " + config[key]['type']
        minval = config[key]['minmax'][0]
        maxval = config[key]['minmax'][1]
        if config[key]['type'] == 'int': 
            step = config[key]['step']
            candidates = list(range(minval, maxval, step)) if maxval > minval else [minval, ]
            selected = candidates[random.randint(0, len(candidates)-1)]
        elif config[key]['type'] == 'float':  
            selected = minval + (maxval - minval) * random.random()
        variables[key] = selected
    
    print(f"Variables: {variables}")
    # metrics_file = baseline(OPENROAD_FLOW_PATH, "dev/baseline", args)
    metrics_file = openroad(OPENROAD_FLOW_PATH, variables, f"dev/test", args, SDC_ORIGINAL, CONSTRAINTS_SDC, FR_ORIGINAL, FASTROUTE_TCL)
    metrics = read_metrics(metrics_file)
    print(f"Results: {metrics}")
