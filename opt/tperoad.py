import os
import sys 
sys.path.append(".")
import time

from os.path import abspath
from datetime import datetime
from multiprocessing import cpu_count
from subprocess import run

import optuna

from utils.utils import *

class MOTPE: 

    def __init__(self, nDims, nObjs, nameVars, typeVars, rangeVars, funcEval, nInit=2**4): 
        self._nDims      = nDims
        self._nObjs      = nObjs
        self._nameVars   = nameVars
        self._typeVars   = typeVars
        self._rangeVars  = rangeVars
        self._funcEval   = funcEval
        self._name2index = {}
        for idx, name in enumerate(self._nameVars): 
            self._name2index[name] = idx
        self._model = optuna.create_study(directions=["minimize" for _ in range(self._nObjs)], \
                                          sampler=optuna.samplers.TPESampler(n_startup_trials=nInit, multivariate=False, group=False))
        self._visited = {}
        
    def optimize(self, steps=2**4, timeout=None): 
        def objective(trial): 
            variables = []
            values = []
            for idx, name in enumerate(self._nameVars): 
                typename = self._typeVars[idx]
                variables.append(trial.suggest_float(name, 0.0, 1.0))
                value = None
                if typename == "int": 
                    value = round(variables[idx] * (self._rangeVars[idx][1] - self._rangeVars[idx][0])) + self._rangeVars[idx][0]
                elif typename == "float": 
                    value = variables[idx] * (self._rangeVars[idx][1] - self._rangeVars[idx][0]) + self._rangeVars[idx][0]
                elif typename == "enum": 
                    value = self._rangeVars[idx][round(variables[idx] * (len(self._rangeVars[idx]) - 1))] if len(self._rangeVars[idx]) > 1 else self._rangeVars[idx][0]
                else: 
                    assert typename in ["int", "float", "enum"]
                assert not value is None
                values.append(value)
            name = str(variables)
            if name in self._visited: 
                return self._visited[name]
            config = dict(zip(self._nameVars, values))
            score = self._funcEval(config)
            self._visited[name] = score
            print("[Eval] Point:", config)
            print(" -> Result:", score)
            with open("historyMOTPE.txt", "a+") as fout: 
                fout.write(str(config) + "\n")
                fout.write(str(score) + "\n")
            return score
        
        self._model.optimize(objective, n_trials=steps, timeout=timeout)
        trials = self._model.best_trials
        results = []
        for trial in trials: 
            params = trial.params
            values = trial.values
            results.append((params, values))
        return results

import argparse
def parseArgs(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--flowdir", required=True, action="store")
    parser.add_argument("-p", "--platform", default="sky130hd", required=False, action="store")
    parser.add_argument("-d", "--design", default="gcd", required=False, action="store")
    parser.add_argument("--refclk", required=False, action="store", default=1.5)
    parser.add_argument("--refarea", required=False, action="store", default=3200)

    parser.add_argument("-r", "--refpoint", required=False, action="store", default=150)
    parser.add_argument("-n", "--nobjs", required=False, action="store", default=2)
    parser.add_argument("-i", "--ninit", required=False, action="store", default=4)
    parser.add_argument("-s", "--steps", required=False, action="store", default=16)
    parser.add_argument("-t", "--timeout", required=False, action="store", default=None)
    parser.add_argument("-o", "--output", required=False, action="store", default="tmp")
    return parser.parse_args()


from dev.openroad import *
if __name__ == "__main__": 
    args = parseArgs()
    cfgfile = f"{args.flowdir}/flow/designs/{args.platform}/{args.design}/autotuner.json"
    config, SDC_ORIGINAL, FR_ORIGINAL = read_config(os.path.abspath(cfgfile))

    names, types, ranges = [], [], []
    for key in config.keys(): 
        assert config[key]['type'] in ['int', 'float'], "Unsupported type: " + config[key]['type']
        minval = config[key]['minmax'][0]
        maxval = config[key]['minmax'][1]
        names.append(key)
        types.append(config[key]['type'])
        ranges.append([minval, maxval])

    refpoint = float(args.refpoint)
    nobjs = int(args.nobjs)
    ninit = int(args.ninit)
    steps = int(args.steps)
    timeout = None if args.timeout is None else float(args.timeout)
    folder = args.output
    if not os.path.exists(folder): 
        os.mkdir(folder)
    
    def funcEval(config): 
        iter = str(time.time()).replace(".", "")
        metrics_file = openroad(args.flowdir, config, f"run/run{iter}", args, SDC_ORIGINAL, CONSTRAINTS_SDC, FR_ORIGINAL, FASTROUTE_TCL)
        metrics = read_metrics(metrics_file)
        clk_period = (metrics["clk_period"] / args.refclk * 100) if not isinstance(metrics["clk_period"], str) else refpoint*2
        final_area = (metrics["final_area"] / args.refarea * 100) if not isinstance(metrics["final_area"], str) else refpoint*2
        results = [clk_period, final_area]
        return results
    
    ndims = len(names)
    model = MOTPE(ndims, nobjs, names, types, ranges, funcEval, nInit=ninit)
    results = model.optimize(steps=steps, timeout=None)
    list(map(lambda x: print("Parameter:", x[0], "\n -> Value:", x[1]), results))
    print("[Hypervolume]:", calcHypervolume([refpoint] * nobjs, list(map(lambda x: x[1], results))))
    
