import os
import sys 
sys.path.append(".")
import math
import time
import random
import multiprocessing as mp
from dataclasses import dataclass

from sklearn.cluster import KMeans

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP, ModelList
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
import gpytorch.settings as gpts
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, RFFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from utils.utils import *

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DTYPE = torch.float if torch.cuda.is_available() else torch.double
DEVICE = torch.device("cpu")
DTYPE = torch.double

def samplesSobol(ndims, samples, seed=None):
    sobol = SobolEngine(dimension=ndims, scramble=True, seed=seed)
    return sobol.draw(n=samples).to(dtype=DTYPE, device=DEVICE)

def calcHV(refpoint, initParetoParams, initParetoValues, tmp1, tmp2): 
    newParetoParams, newParetoValues = newParetoSet(initParetoParams, initParetoValues, tmp1, tmp2)
    newHV = calcHypervolume(refpoint, newParetoValues)
    return newHV

def calcHV2(refpoint, newParams, newValues): 
    newParetoParams, newParetoValues = pareto(newParams, newValues)
    newHV = calcHypervolume(refpoint, newParetoValues)
    return newHV

@dataclass
class MorboState:
    dim: int
    batch_size: int
    length: float = 0.25
    length_min: float = 0.0025
    length_max: float = 0.50
    failure_counter: int = 0
    failure_tolerance: int = 2
    success_counter: int = 0
    success_tolerance: int = 5
    restart_triggered: bool = False

    def update(self, success):
        if success:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length /= 2.0
            self.failure_counter = 0

        if self.length < self.length_min:
            self.restart_triggered = True


class Morbo: 

    def __init__(self, nEmbs, nDims, nObjs, nameVars, typeVars, rangeVars, funcEval, refpoint, weights, scale=1.0, nInit=2**4, batchSize=2, mcSamples=1024, nJobs=4): 
        self._nEmbs      = nEmbs
        self._nDims      = nDims
        self._nObjs      = nObjs
        self._nameVars   = nameVars
        self._typeVars   = typeVars
        self._rangeVars  = rangeVars
        self._funcEval   = funcEval
        self._name2index = {}
        for idx, name in enumerate(self._nameVars): 
            self._name2index[name] = idx
        self._refpoint = refpoint
        self._weights = weights
        self._nInit = nInit
        self._batchSize = batchSize
        self._mcSamples = mcSamples
        self._nJobs = nJobs
        self._bounds = torch.tensor([[-1.0] * self._nDims, [1.0] * self._nDims], \
                                    device=DEVICE, dtype=DTYPE)
        self._scale = scale
        self._boundsEmbs = torch.tensor([[-self._scale] * self._nEmbs, [self._scale] * self._nEmbs], \
                                        device=DEVICE, dtype=DTYPE)
        # self._A = np.random.randn(self._nDims, self._nEmbs)
        # self._A = self._A / np.linalg.norm(self._A, axis=1, keepdims=True)
        self._A = self._getA()

        self._visited = {}
    
    def _getA(self): 
        def objective(mat, show=False): 
            variables = list(mat.flatten())
            values = []
            gauss = []
            step = 1.0 / 16.0
            curr = -3.0
            while curr < 3.0: 
                values.append(curr)
                gauss.append(0.5 * (1.0 + math.erf((curr + step)/math.sqrt(2.0))) - 0.5 * (1.0 + math.erf(curr/math.sqrt(2.0))))
                curr += step
            count = [0 for _ in range(len(values))]
            for value in variables: 
                index = 0
                for idx in range(len(values)): 
                    if value > values[idx]: 
                        index = idx
                count[index] += 1
            prob1 = np.array(count) / len(variables)
            prob2 = np.array(gauss) / np.sum(gauss)
            epsilon = 1e-8
            # print(prob1)
            # print(prob2)
            kl = np.sum(prob1 * np.log((prob1 + epsilon) / (prob2 + epsilon)))
            
            count = 0
            samples = list((samplesSobol(self._nEmbs, 1024*4).cpu().numpy() * 2.0 - 1.0) * self._scale)
            for sample in samples: 
                if self._inRange(sample, mat): 
                    count += 1
            prob = 1.0 - count / len(samples)

            if show:
                print("[_getA]", kl, ";", prob)
            
            return (1.0) * kl + (10.0) * prob

        print("[_getA] Optimizing the embedding")
        length = self._nDims * self._nEmbs
        matLast = np.random.randn(self._nDims, self._nEmbs)
        costLast = objective(matLast, show=True)
        temperature = 0.1
        for iter in range(1024*4): 
            mod = matLast.copy()
            mod[random.randint(0, mod.shape[0]-1)] = np.random.randn(mod.shape[1])
            costCurr = objective(mod, show=False)
            if costCurr <= costLast or random.random() < math.exp((costLast - costCurr) / temperature): 
                objective(mod, show=True)
                print("Iter", iter, "-> Accept with prob", math.exp((costLast - costCurr) / temperature), ";", costLast, "->", costCurr)
                matLast = mod
                costLast = costCurr
            if (iter + 1) % 128 == 0: 
                temperature *= 0.9
        return matLast

    def _map(self, embs, clip=False): 
        A = self._A
        y = np.array(embs)[:, np.newaxis]
        Ay = np.dot(A, y)
        result = Ay.T[0]
        if clip: 
            box = self._bounds.cpu().numpy()
            result = np.clip(result, box[0], box[1])
            assert np.all(result >= box[0]) and np.all(result <= box[1])
        return list(result)

    def _inRange(self, embs, A=None): 
        if A is None: 
            A = self._A
        y = np.array(embs)[:, np.newaxis]
        Ay = np.dot(A, y)
        result = Ay.T[0]
        box = self._bounds.cpu().numpy()
        if np.all(result >= box[0]) and np.all(result <= box[1]): 
            return True
        return False

    def _evalPoint(self, point, index=0): 
        point = list((np.array(self._map(point, clip=True)) + 1.0) / 2.0)
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
        name = str(list(point))
        if name in self._visited: 
            return self._visited[name]
        config = dict(zip(self._nameVars, values))
        score = self._funcEval(config, index)
        print("[Eval index " + str(index) + "] Point:", config)
        print(" -> Result:", score)
        with open("historyREMORBO2index" + str(index) + ".txt", "a+") as fout: 
            fout.write(str(config) + "\n")
            fout.write(str(score) + "\n")

        return score

    def evalPoint(self, point, index=0): 
        score = self._evalPoint(point, index)

        return list(-np.array(score))

    def evalBatch(self, batch, index=0): 
        if not isinstance(batch, torch.Tensor): 
            batch = torch.tensor(batch, dtype=DTYPE, device=DEVICE)
        results = []
        batch = batch.cpu().numpy()

        processes = []
        pool = mp.Pool(processes=self._nJobs)
        for jdx in range(batch.shape[0]): 
            param = batch[jdx]
            process = pool.apply_async(self.evalPoint, (param, index))
            processes.append(process)
            time.sleep(0.01)
        pool.close()
        pool.join()

        for idx in range(batch.shape[0]): 
            cost = processes[idx].get()
            results.append(cost)
        results = np.array(results)

        for idx in range(batch.shape[0]): 
            name = str(list(batch[idx]))
            if not name in self._visited: 
                self._visited[name] = list(-results[idx])
        
        return torch.tensor(results, device=DEVICE, dtype=DTYPE)
    

    def initSamples(self, ninit=None): 
        if ninit is None: 
            ninit = self._nInit
        initX = []
        used = set()
        samples = list((samplesSobol(self._nEmbs, 4*ninit).cpu().numpy() * 2.0 - 1.0) * self._scale)
        while len(initX) < ninit: 
            index = random.randint(0, len(samples) - 1)
            if index in used: 
                continue
            used.add(index)
            if self._inRange(samples[index]): 
                initX.append(samples[index])
            if len(samples) <= len(used): 
                print("[initSamples]: not enough samples, add more")
                samples.extend(list((samplesSobol(self._nEmbs, 4*ninit).cpu().numpy() * 2.0 - 1.0) * self._scale))
        print("[initSamples]: OK")
        initX = np.array(initX)
        initX = torch.tensor(initX, dtype=DTYPE, device=DEVICE)
        initY = self.evalBatch(initX, 0)
        initY = self.evalBatch(initX, 1)
        return initX, initY
    

    # def initSamples(self, ninit=None): 
    #     if ninit is None: 
    #         ninit = self._nInit
    #     initX = (samplesSobol(self._nEmbs, ninit) * 2.0 - 1.0) * self._scale
    #     initY = self.evalBatch(initX, 0)
    #     initY = self.evalBatch(initX, 1)
    #     return initX, initY
    

    def initModel(self, trainX, trainY): 
        models = []
        for idx in range(self._nObjs):
            tmpY = trainY[..., idx:idx+1]
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=trainX.shape[1])
            )
            model = SingleTaskGP(trainX, tmpY, covar_module=covar_module, likelihood=likelihood)
            models.append(model)
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model


    def getRegionHV(self, region):
        print("[getRegionPareto]", region)
        state = self._states[region]
        trainX = self._dataXY[region][0]
        trainY = self._dataXY[region][1]

        initX = trainX.cpu().numpy()
        initY = trainY.cpu().numpy()
        initParams = []
        initValues = []
        for idx, param in enumerate(list(initX)): 
            initParams.append(list(param))
            initValues.append(list(-initY[idx]))
        initParetoParams, initParetoValues = pareto(initParams, initValues)
        initHV = calcHypervolume(self._refpoint, initParetoValues)

        return initHV


    def getObservations(self, region):
        print("[GetObservations]", region)
        state = self._states[region]
        trainX = self._dataXY[region][0]
        trainY = self._dataXY[region][1]

        # Select the center
        hvc = []

        initX = trainX.cpu().numpy()
        initY = trainY.cpu().numpy()
        initParams = []
        initValues = []
        for idx, param in enumerate(list(initX)): 
            initParams.append(list(param))
            initValues.append(list(-initY[idx]))
        initParetoParams, initParetoValues = pareto(initParams, initValues)
        print(initParetoValues)
        initHV = calcHypervolume(self._refpoint, initParetoValues)
        
        procs = []
        pool = mp.Pool(processes=64)
        for idx in range(initX.shape[0]): 
            newParams = []
            newValues = []
            for jdx, param in enumerate(list(initX)): 
                if idx == jdx: 
                    continue
                newParams.append(list(param))
                newValues.append(list(-initY[jdx]))
            proc = pool.apply_async(calcHV2, (self._refpoint, newParams, newValues))
            procs.append(proc)
        pool.close()
        pool.join()
        hvc = []
        for idx in range(initX.shape[0]): 
            newHV = procs[idx].get()
            newHVC = initHV - newHV
            hvc.append(newHVC)
        hvc = np.array(hvc)
        
        # for idx in range(initX.shape[0]): 
        #     newParams = []
        #     newValues = []
        #     for jdx, param in enumerate(list(initX)): 
        #         if idx == jdx: 
        #             continue
        #         newParams.append(list(param))
        #         newValues.append(list(-initY[jdx]))
        #     newParetoParams, newParetoValues = pareto(newParams, newValues)
        #     newHV = calcHypervolume(self._refpoint, newParetoValues)
        #     newHVC = initHV - newHV
        #     hvc.append(newHVC)
        # hvc = np.array(hvc)

        center = trainX[hvc.argmax(), :].clone()
        tr_lb = torch.clamp(center - self._scale * state.length / 2.0, -self._scale, self._scale)
        tr_ub = torch.clamp(center + self._scale * state.length / 2.0, -self._scale, self._scale)
        x_lb = torch.clamp(center - self._scale * state.length, -self._scale, self._scale)
        x_ub = torch.clamp(center + self._scale * state.length, -self._scale, self._scale)

        # Fit the model
        print("[Center]:", center)
        dataX = trainX
        dataY = trainY
        used = set()
        for idx in range(dataX.shape[0]): 
            used.add(str(dataX[idx]))
        indices = []
        for idx in range(self._dataAllX.shape[0]): 
            if torch.all(self._dataAllX[idx] <= x_ub) and torch.all(self._dataAllX[idx] >= x_lb) and not str(self._dataAllX[idx]) in used: 
                indices.append(idx)
                used.add(str(self._dataAllX[idx]))
        dataX = torch.cat([dataX, self._dataAllX[indices]])
        dataY = torch.cat([dataY, self._dataAllY[indices]])

        mll, model = self.initModel(dataX, dataY)
        fit_gpytorch_model(mll)

        # Sample on the candidate points
        predX = None
        for submodel in model.models: 
            dim = trainX.shape[-1]
            sobol = SobolEngine(dim, scramble=True)
            candX = sobol.draw(1024 * 4).to(dtype=DTYPE, device=DEVICE)
            candX = tr_lb + (tr_ub - tr_lb) * candX
            with gpts.fast_computations(covar_root_decomposition=True): 
                with torch.no_grad():  # We don't need gradients when using TS
                    thompson_sampling = MaxPosteriorSampling(model=submodel, replacement=False)
                    tmp = thompson_sampling(candX, num_samples=1024)
                if predX is None: 
                    predX = tmp
                else: 
                    predX = torch.cat([predX, tmp],)
        predY = torch.zeros([predX.shape[0], self._nObjs], dtype=DTYPE, device=DEVICE)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for idx, submodel in enumerate(model.models): 
                submodel.eval()
                mll.eval()
                predY[:, idx] = mll.likelihood(submodel(predX))[0].sample()
        predX = predX.cpu().numpy()
        predY = predY.cpu().numpy()
        # hvi = []
        # for idx in range(predX.shape[0]): 
        #     tmp1 = list(predX[idx])
        #     tmp2 = list(-predY[idx])
        #     newParetoParams, newParetoValues = newParetoSet(initParetoParams, initParetoValues, tmp1, tmp2)
        #     newHV = calcHypervolume(self._refpoint, newParetoValues)
        #     hvi.append(newHV - initHV)
        procs = []
        pool = mp.Pool(processes=64)
        for idx in range(predX.shape[0]): 
            tmp1 = list(predX[idx])
            tmp2 = list(-predY[idx])
            proc = pool.apply_async(calcHV, (self._refpoint, initParetoParams, initParetoValues, tmp1, tmp2))
            procs.append(proc)
        pool.close()
        pool.join()
        hvi = []
        for idx in range(predX.shape[0]): 
            res = procs[idx].get()
            hvi.append(res - initHV)
        
        indices = list(range(len(hvi)))
        indices = sorted(indices, key=lambda x: (0 if self._inRange(predX[x]) else 1024.0, -hvi[x], -np.sum(np.abs(predY[x] - np.array(self._refpoint)))))
        hvi = np.array(hvi)
        indices = indices[:2*self._batchSize]
        print(" -> [Sampling original]:", indices, "HVI:", hvi[indices])

        newX = torch.tensor(predX[indices], dtype=DTYPE, device=DEVICE)
        newY = self.evalBatch(newX, 0)
        indices = list(range(len(indices)))
        
        # NSGA-II sorting -> self._batchSize remained
        selected = []
        while len(selected) < self._batchSize: 
            paretoParams, paretoValues = pareto(indices, list(newY[indices]))
            index2value = dict(zip(paretoParams, paretoValues))
            distances = {}
            for idx, index in enumerate(paretoParams): 
                distances[index] = 0.0
            for idx in range(self._nObjs): 
                sortedIndices = sorted(paretoParams, key=lambda x: index2value[x][idx])
                for jdx, index in enumerate(sortedIndices): 
                    if jdx == 0 or jdx == len(sortedIndices) - 1: 
                        distances[index] += 1024.0
                    else: 
                        index1 = sortedIndices[jdx]
                        index2 = sortedIndices[jdx + 1]
                        distances[index] += (index2value[index2][idx] - index2value[index1][idx]) / (index2value[sortedIndices[-1]][idx] - index2value[sortedIndices[0]][idx] + 1e-8)
            sortedIndices = sorted(paretoParams, key=lambda x: distances[x], reverse=True)
            selected.extend(sortedIndices)
            tmp = []
            for index in indices: 
                if not index in selected: 
                    tmp.append(index)
            indices = tmp
        if len(selected) > self._batchSize: 
            selected = selected[0:self._batchSize]
        print(" -> [Sampling sorted]:",selected)

        newX = torch.tensor(newX[selected], dtype=DTYPE, device=DEVICE)
        newY = self.evalBatch(newX, 1)

        return newX, newY

        
    def optimize(self, steps=2**4, regions=4, verbose=True): 
        self._states = []
        self._mlls   = []
        self._models = []
        self._dataXY = []
        self._dataAllX = None
        self._dataAllY = None

        # Initialize all regions
        trainAllX, trainAllY = self.initSamples(regions * self._nInit)
        trainAllX = trainAllX.cpu().numpy()
        trainAllY = trainAllY.cpu().numpy()
        kmeans = KMeans(n_clusters=regions)
        indices = kmeans.fit_predict(trainAllX)
        trainXs = [[] for _ in range(regions)]
        trainYs = [[] for _ in range(regions)]
        for idx in range(regions): 
            for jdx, region in enumerate(indices): 
                if idx == region: 
                    trainXs[idx].append(trainAllX[jdx])
                    trainYs[idx].append(trainAllY[jdx])
            assert len(trainXs[idx]) > 0
            assert len(trainYs[idx]) > 0
            trainXs[idx] = np.array(trainXs[idx])
            trainYs[idx] = np.array(trainYs[idx])
            trainXs[idx] = torch.tensor(trainXs[idx], dtype=DTYPE, device=DEVICE)
            trainYs[idx] = torch.tensor(trainYs[idx], dtype=DTYPE, device=DEVICE)

        initParams = []
        initValues = []
        for region in range(regions): 
            trainX, trainY = trainXs[region], trainYs[region]
            state = MorboState(dim=self._nEmbs, batch_size=self._batchSize)
            initX = trainX.cpu().numpy()
            initY = trainY.cpu().numpy()
            for idx in range(initX.shape[0]): 
                initParams.append(list(initX[idx]))
                initValues.append(list(-initY[idx]))
            self._states.append(state)
            self._dataXY.append([trainX, trainY])
            if self._dataAllX is None: 
                self._dataAllX = trainX
            else: 
                self._dataAllX = torch.cat([self._dataAllX, trainX])
            if self._dataAllY is None: 
                self._dataAllY = trainY
            else: 
                self._dataAllY = torch.cat([self._dataAllY, trainY])
        paretoParams, paretoValues = pareto(initParams, initValues)
        print(f'[Initial PARETO]: {paretoParams}, {paretoValues}')
        print(f'[Hypervolume]:', calcHypervolume(self._refpoint, paretoValues))
        
        for iter in range(steps): 
            t0 = time.time()
            for region in range(regions): 
                with gpytorch.settings.max_cholesky_size(float("inf")):
                    newX, newY = self.getObservations(region)
                self._dataXY[region][0] = torch.cat([self._dataXY[region][0], newX])
                self._dataXY[region][1] = torch.cat([self._dataXY[region][1], newY])
                self._dataAllX = torch.cat([self._dataAllX, newX])
                self._dataAllY = torch.cat([self._dataAllY, newY])

                paretoPrev = str(paretoValues)
                tmpX = newX.cpu().numpy()
                tmpY = newY.cpu().numpy()
                for idx in range(tmpX.shape[0]): 
                    paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, list(tmpX[idx]), list(-tmpY[idx]))
                paretoPost = str(paretoValues)

                self._states[region].update(paretoPrev != paretoPost)
                regionHV = self.getRegionHV(region)
                print("[RegionHV]", regionHV)
                if self._states[region].restart_triggered or regionHV == 0.0: 
                    print(f"[Restart] region {region} restart")
                    trainX, trainY = self.initSamples()
                    self._states[region] = MorboState(dim=self._nEmbs, batch_size=self._batchSize)
                    self._dataXY[region][0] = trainX
                    self._dataXY[region][1] = trainY
                    self._dataAllX = torch.cat([self._dataAllX, trainX])
                    self._dataAllY = torch.cat([self._dataAllY, trainY])
                    initX = trainX.cpu().numpy()
                    initY = trainY.cpu().numpy()
                    for idx in range(initX.shape[0]): 
                        paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, list(initX[idx]), list(-initY[idx]))
                    
            t1 = time.time()
            if verbose:
                print(f"Batch {iter}: Pareto-front ->", paretoValues)
                print(f"[Hypervolume]:", f"(Batch {iter})", calcHypervolume(self._refpoint, paretoValues), ", [Time]:", t1 - t0, "s") 
            else:
                print(".", end="")

        return list(zip(paretoParams, paretoValues))


import argparse
def parseArgs(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paramfile", required=True, action="store")
    parser.add_argument("-c", "--command1", required=True, action="store")
    parser.add_argument("-d", "--command2", required=True, action="store")
    parser.add_argument("-r", "--refpoint", required=True, action="store")
    parser.add_argument("-e", "--nembs", required=False, action="store", default=16)
    parser.add_argument("-n", "--nobjs", required=True, action="store")
    parser.add_argument("-b", "--batchsize", required=False, action="store", default=4)
    parser.add_argument("-i", "--ninit", required=False, action="store", default=16)
    parser.add_argument("-s", "--steps", required=False, action="store", default=64)
    parser.add_argument("-t", "--timeout", required=False, action="store", default=None)
    parser.add_argument("-o", "--output", required=False, action="store", default="tmp")
    parser.add_argument("-j", "--njobs", required=False, action="store", default=4)
    parser.add_argument("-m", "--morbos", required=False, action="store", default=4)
    parser.add_argument("--scale", required=False, action="store", default=1.0)
    return parser.parse_args()


if __name__ == "__main__": 
    args = parseArgs()
    configfile = args.paramfile
    command1 = args.command1
    command2 = args.command2
    refpoint = float(args.refpoint)
    nembs = int(args.nembs)
    nobjs = int(args.nobjs)
    ninit = int(args.ninit)
    steps = int(args.steps)
    timeout = None if args.timeout is None else float(args.timeout)
    folder = args.output
    if not os.path.exists(folder): 
        os.mkdir(folder)
    run = runCommand
    if command1[-3:] == ".py": 
        run = runPythonCommand
    
    def funcEval(config, index=0): 
        command = command2 if index == 1 else command1
        iter = str(time.time()).replace(".", "")
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
        
        return results
    
    names, types, ranges = readConfig(configfile)
    ndims = len(names)
    model = Morbo(nembs, ndims, nobjs, names, types, ranges, funcEval, refpoint=[refpoint, ] * nobjs, weights=[1.0/nobjs] * nobjs, scale=float(args.scale), nInit=ninit, batchSize=int(args.batchsize), nJobs=int(args.njobs))
    results = model.optimize(steps=steps, regions=int(args.morbos), verbose=True)
    print(results)
    list(map(lambda x: print("Parameter:", x[0], "\n -> Value:", x[1]), results))
    print("[Hypervolume]:", calcHypervolume([refpoint] * nobjs, list(map(lambda x: x[1], results))))

