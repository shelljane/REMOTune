import os
import sys 
sys.path.append(".")

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
    parser.add_argument("-p", "--paramfile", required=True, action="store")
    parser.add_argument("-c", "--command", required=True, action="store")
    parser.add_argument("-r", "--refpoint", required=True, action="store")
    parser.add_argument("-n", "--nobjs", required=True, action="store")
    parser.add_argument("-i", "--ninit", required=False, action="store", default=16)
    parser.add_argument("-s", "--steps", required=False, action="store", default=64)
    parser.add_argument("-t", "--timeout", required=False, action="store", default=None)
    parser.add_argument("-o", "--output", required=False, action="store", default="tmp")
    return parser.parse_args()

if __name__ == "__main__": 
    args = parseArgs()
    configfile = args.paramfile
    command = args.command
    refpoint = float(args.refpoint)
    nobjs = int(args.nobjs)
    ninit = int(args.ninit)
    steps = int(args.steps)
    timeout = None if args.timeout is None else float(args.timeout)
    folder = args.output
    if not os.path.exists(folder): 
        os.mkdir(folder)
    run = runCommand
    if command[-3:] == ".py": 
        run = runPythonCommand
    
    iter = 0
    def funcEval(config): 
        global iter
        filename = folder + f"/run{iter}.log"
        ret = run(command, config, timeout, filename)
        results = [refpoint, ] * nobjs
        try: 
            with open(filename, "r") as fin: 
                lines = fin.readlines()
                splited = lines[0].split()
                for idx, elem in enumerate(splited): 
                    if len(elem) > 0 and idx < len(results): 
                        results[idx] = float(elem)  
        except Exception: 
            pass
        iter += 1
        
        return results
    
    names, types, ranges = readConfig(configfile)
    ndims = len(names)
    model = MOTPE(ndims, nobjs, names, types, ranges, funcEval, nInit=ninit)
    results = model.optimize(steps=steps, timeout=None)
    list(map(lambda x: print("Parameter:", x[0], "\n -> Value:", x[1]), results))
    print("[Hypervolume]:", calcHypervolume([refpoint] * nobjs, list(map(lambda x: x[1], results))))
    
