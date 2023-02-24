import os
import sys 
sys.path.append(".")
import time

from os.path import abspath
from datetime import datetime
from multiprocessing import cpu_count
from subprocess import run

import torch
import botorch
from botorch.models import SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning

from utils.utils import *

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DTYPE = torch.float if torch.cuda.is_available() else torch.double
DEVICE = torch.device("cpu")
DTYPE = torch.double

class VanillaBO: 

    def __init__(self, nDims, nObjs, nameVars, typeVars, rangeVars, funcEval, weights, refpoint=0.0, nInit=2**4, batchSize=2, numRestarts=10, rawSamples=512, mcSamples=256): 
        self._nDims      = nDims
        self._nObjs      = nObjs
        self._nameVars   = nameVars
        self._typeVars   = typeVars
        self._rangeVars  = rangeVars
        self._funcEval   = funcEval
        self._name2index = {}
        for idx, name in enumerate(self._nameVars): 
            self._name2index[name] = idx
        self._refpoint = [refpoint] * self._nObjs
        self._nInit = nInit
        self._batchSize = batchSize
        self._weights = weights
        self._numRestarts = numRestarts
        self._rawSamples = rawSamples
        self._mcSamples = mcSamples
        self._bounds = torch.tensor([[0.0] * self._nDims, [1.0] * self._nDims], \
                                    device=DEVICE, dtype=DTYPE)
        self._visited = {}

    def _evalPoint(self, point): 
        values = []
        for idx, name in enumerate(self._nameVars): 
            typename = self._typeVars[idx]
            value = None
            if typename == "int": 
                value = int(round(point[idx] * (self._rangeVars[idx][1] - self._rangeVars[idx][0])) + self._rangeVars[idx][0])
            elif typename == "float": 
                value = point[idx] * (self._rangeVars[idx][1] - self._rangeVars[idx][0]) + self._rangeVars[idx][0]
            elif typename == "enum": 
                value = self._rangeVars[idx][int(round(point[idx] * (len(self._rangeVars[idx]) - 1)))] if len(self._rangeVars[idx]) > 1 else self._rangeVars[idx][0]
            else: 
                assert typename in ["int", "float", "enum"]
            assert not value is None
            values.append(value)
        name = str(point)
        if name in self._visited: 
            return self._visited[name]
        config = dict(zip(self._nameVars, values))
        score = self._funcEval(config)
        self._visited[name] = score
        print("[Eval] Point:", config)
        print(" -> Result:", score)
        with open("historyBO.txt", "a+") as fout: 
            fout.write(str(config) + "\n")
            fout.write(str(score) + "\n")

        return score

    def evalPoint(self, point): 
        score = self._evalPoint(point)
        assert len(score) <= len(self._weights)
        cost = 0.0
        for idx, elem in enumerate(score): 
            cost += self._weights[idx] * elem

        return -cost

    def evalBatch(self, batch): 
        if isinstance(batch, list): 
            batch = torch.tensor(batch, device=DEVICE)
        results = []
        batch = batch.cpu().numpy()
        for idx in range(batch.shape[0]): 
            param = batch[idx]
            cost = self.evalPoint(param)
            results.append(cost)
        
        return torch.tensor(results, device=DEVICE).unsqueeze(-1)
    

    def initSamples(self): 
        initX = torch.rand(self._nInit, self._nDims, device=DEVICE, dtype=DTYPE)
        initY = self.evalBatch(initX)
        return initX, initY
    

    def initModel(self, trainX, trainY, stateDict=None): 
        model = SingleTaskGP(trainX, trainY).to(trainX)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        if stateDict is not None:
            model.load_state_dict(stateDict)
        return mll, model
    

    def getObservations(self, acqFunc):
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acqFunc,
            bounds=self._bounds,
            q=self._batchSize,
            num_restarts=self._numRestarts,
            raw_samples=self._rawSamples,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )
        # observe new values 
        newX = candidates.detach()
        newY = self.evalBatch(newX)
        return newX, newY

        
    def optimize(self, steps=2**4, verbose=True): 
        bestX = None
        bestY = None

        trainX, trainY = self.initSamples()
        mll, model = self.initModel(trainX, trainY)
        initX = trainX.cpu().numpy()
        initY = trainY.cpu().numpy()
        index = initY.argmax()
        bestX = initX[index]
        bestY = initY[index]

        params = []
        values = []
        for param in list(initX): 
            params.append(list(param))
            values.append(self._evalPoint(param))
        paretoParams, paretoValues = pareto(params, values)
        print(f'[Initial PARETO]: {paretoParams}, {paretoValues}')
        print(f'[Hypervolume]:', calcHypervolume(self._refpoint, paretoValues))
        
        for iter in range(steps):  
            t0 = time.time()
            
            fit_gpytorch_model(mll)
            qmcSampler = SobolQMCNormalSampler(num_samples=self._mcSamples)
            qEI = qExpectedImprovement(
                model=model, 
                best_f=bestY,
                sampler=qmcSampler
            )
            newX, newY = self.getObservations(qEI)
                    
            trainX = torch.cat([trainX, newX])
            trainY = torch.cat([trainY, newY])

            tmpX = trainX.cpu().numpy()
            tmpY = trainY.cpu().numpy()
            index = tmpY.argmax()
            bestX = tmpX[index]
            bestY = tmpY[index]

            tmpBatchX = newX.cpu().numpy()
            tmpBatchY = newY.cpu().numpy()
            index = tmpBatchY.argmax()
            bestBatchX = tmpBatchX[index]
            bestBatchY = tmpBatchY[index]

            for param in list(tmpBatchX): 
                paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, list(param), self._evalPoint(param))

            mll, model = self.initModel(trainX, trainY)
            
            t1 = time.time()
            if verbose:
                print(f"Batch {iter:>2}: best params = {bestX}; best result = {-bestY}")
                print(f" -> Batch best = {bestBatchX}, {-bestBatchY}")
                print(f" -> Pareto-front:", paretoValues)
                print(f'[Hypervolume]:', calcHypervolume(self._refpoint, paretoValues))
            else:
                print(".", end="")

        # return [[bestX, -bestY], ]
        return list(zip(paretoParams, paretoValues))


import argparse
def parseArgs(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paramfile", required=True, action="store")
    parser.add_argument("-c", "--command", required=True, action="store")
    parser.add_argument("-r", "--refpoint", required=True, action="store")
    parser.add_argument("-n", "--nobjs", required=True, action="store")
    parser.add_argument("-b", "--batchsize", required=False, action="store", default=4)
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
    model = VanillaBO(ndims, nobjs, names, types, ranges, funcEval, refpoint=refpoint, weights=[1.0/nobjs] * nobjs, nInit=ninit, batchSize=int(args.batchsize))
    results = model.optimize(steps=steps, verbose=True)
    print(results)
    list(map(lambda x: print("Parameter:", x[0], "\n -> Value:", x[1]), results))
    print("[Hypervolume]:", calcHypervolume([refpoint] * nobjs, list(map(lambda x: x[1], results))))
