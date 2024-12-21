import os
import sys
sys.path.append(".")

from os.path import abspath
from datetime import datetime
from multiprocessing import cpu_count
from subprocess import run
import numpy as np

from botorch.utils.sampling import draw_sobol_samples
from scipy.optimize import minimize as scipyminimize
from platypus import NSGAII, Problem, Real

from utils.utils import *
from utils.mesmo import GaussianProcess, MaxvalueEntropySearch

DEVICE = torch.device("cpu")
DTYPE = torch.double

class MESMO:

    def __init__(self, nDims, nObjs, nameVars, typeVars, rangeVars, funcEval, refpoint=0.0, nInit=2**4, \
                 samplesNSGA=1, itersNSGA=1024, iterNewton=1024):
        self._nDims     = nDims
        self._nObjs     = nObjs
        self._nameVars  = nameVars
        self._typeVars  = typeVars
        self._rangeVars = rangeVars
        self._funcEval  = funcEval
        self._refpoint  = [refpoint] * self._nObjs
        self._nInit     = nInit
        self._name2index = {}
        for idx, name in enumerate(self._nameVars):
            self._name2index[name] = idx

        self._samplesNSGA = samplesNSGA
        self._itersNSGA = itersNSGA
        self._iterNewton = iterNewton

        self._bound = [0.0, 1.0]
        self._bounds = [self._bound, ] * self._nDims
        # self._grid = sobol_seq.i4_sobol_generate(self._nDims, 1024, np.random.randint(0, 128))
        bounds = torch.tensor([[0.0] * self._nDims, [1.0] * self._nDims], device=DEVICE, dtype=DTYPE)
        self._grid = draw_sobol_samples(bounds=bounds, n=1024*4, q=1).squeeze(1).cpu().numpy()

        self._GPs=[]
        self._multiplemes=[]
        for idx in range(self._nObjs):
            self._GPs.append(GaussianProcess(self._nDims))

        self._visited = {}

    def optimize(self, steps=2**8, timeout=None):
        def objective(variables):
            values = []
            for idx, name in enumerate(self._nameVars):
                typename = self._typeVars[idx]
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
            with open("historyMESMO.txt", "a+") as fout:
                fout.write(str(config) + "\n")
                fout.write(str(score) + "\n")
            return score

        # Init
        index = np.random.randint(0, self._grid.shape[0])
        for idx in range(self._nInit):
            exist = True
            while exist:
                index = np.random.randint(0, self._grid.shape[0])
                x_rand = list(self._grid[index : (index + 1), :][0])
                if (any((x_rand == x).all() for x in self._GPs[0].xValues)) == False:
                    exist = False
            results = objective(x_rand)
            print("[Init]", x_rand)
            print(" -> ", results)
            for jdx in range(self._nObjs):
                self._GPs[jdx].addSample(np.asarray(x_rand), results[jdx])
        for idx in range(self._nObjs):
            self._GPs[idx].fitModel()
            self._multiplemes.append(MaxvalueEntropySearch(self._GPs[idx]))

        # Optimize
        initParams = []
        initValues = []
        for xValue in self._GPs[0].xValues:
            initParams.append(xValue)
        for idx in range(len(self._GPs[0].xValues)):
            tmp = []
            for jdx in range(len(self._GPs)):
                tmp.append(self._GPs[jdx].yValues[idx])
            initValues.append(tmp)
        paretoParams, paretoValues = pareto(initParams, initValues)
        print(f'[Initial PARETO]: {paretoParams}, {paretoValues}')
        print(f'[Hypervolume]:', calcHypervolume(self._refpoint, paretoValues))

        print('[GO!]')
        for iter in range(steps):
            for idx in range(self._nObjs):
                self._multiplemes[idx] = MaxvalueEntropySearch(self._GPs[idx])
                self._multiplemes[idx].Sampling_RFM()

            max_samples = []
            for idx in range(self._samplesNSGA):
                for jdx in range(self._nObjs):
                    self._multiplemes[jdx].weigh_sampling()
                cheap_pareto_front=[]
                def CMO(xi):
                    xi=np.asarray(xi)
                    y=[self._multiplemes[jdx].f_regression(xi)[0][0] for jdx in range(len(self._GPs))]
                    return y

                problem = Problem(self._nDims, self._nObjs)
                problem.types[:] = Real(self._bound[0], self._bound[1])
                problem.function = CMO
                algorithm = NSGAII(problem)
                algorithm.run(self._itersNSGA)
                cheap_pareto_front=[list(solution.objectives) for solution in algorithm.result]
                # picking the max over the pareto: best case
                maxoffunctions=[-1*min(f) for f in list(zip(*cheap_pareto_front))]
                max_samples.append(maxoffunctions)

            def mesmo_acq(x):
                multi_obj_acq_total = 0
                for j in range(self._samplesNSGA):
                    multi_obj_acq_sample = 0
                    for i in range(self._nObjs):
                        multi_obj_acq_sample = multi_obj_acq_sample + self._multiplemes[i].single_acq(x, max_samples[j][i])
                    multi_obj_acq_total = multi_obj_acq_total + multi_obj_acq_sample
                return (multi_obj_acq_total / self._samplesNSGA)


            # l-bfgs-b acquisation optimization
            x_tries = np.random.uniform(self._bound[0], self._bound[1], size=(self._iterNewton, self._nDims))
            y_tries = [mesmo_acq(x) for x in x_tries]
            sorted_indecies = np.argsort(y_tries)
            index = 0
            x_best = x_tries[sorted_indecies[index]]
            while (any((x_best == x).all() for x in self._GPs[0].xValues)):
                index += 1
                x_best = x_tries[sorted_indecies[index]]
            y_best = y_tries[sorted_indecies[index]]
            x_seed = list(np.random.uniform(low=self._bound[0], high=self._bound[1], size=(self._iterNewton, self._nDims)))
            for x_try in x_seed:
                result = scipyminimize(mesmo_acq, x0=np.asarray(x_try).reshape(1, -1), method='L-BFGS-B', bounds=self._bounds)
                if not result.success:
                    continue
                if ((result.fun <= y_best) and (not (result.x in np.asarray(self._GPs[0].xValues)))):
                    x_best = result.x
                    y_best = result.fun

            # Updating and fitting the GPs
            results = objective(list(x_best))
            for idx in range(self._nObjs):
                self._GPs[idx].addSample(x_best, results[idx])
                self._GPs[idx].fitModel()
            paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, list(x_best), results)

            print("[ITERATION]", iter, "; Result:", results, "; Pareto-optimal:", len(paretoValues))
            print(f'[Hypervolume]:', calcHypervolume(self._refpoint, paretoValues))

        results = []
        for idx, params in enumerate(paretoParams):
            values = paretoValues[idx]
            results.append((params, values))
        return results



from analytic_funcs import *
def test0():
    nDims = 16
    nObjs = 2
    nameVars = ["x" + str(idx) for idx in range(nDims)]
    typeVars = ["float" for idx in range(nDims)]
    rangeVars = [[0.0, 1.0] for idx in range(nDims)]
    funcEval = lambda x: [Powell(list(x.values()), nDims), Perm(list(x.values()), nDims)]
    model = MESMO(nDims, nObjs, nameVars, typeVars, rangeVars, funcEval, samplesNSGA=1, itersNSGA=1024, iterNewton=1024)
    results = model.optimize(steps=2**4, timeout=None)
    list(map(lambda x: print(x[0], "\t", x[1]), results))

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
    model = MESMO(ndims, nobjs, names, types, ranges, funcEval, refpoint=refpoint, nInit=ninit, \
                  samplesNSGA=1, itersNSGA=1024, iterNewton=256)
    results = model.optimize(steps=steps, timeout=None)
    list(map(lambda x: print("Parameter:", x[0], "\n -> Value:", x[1]), results))
    print("[Hypervolume]:", calcHypervolume([refpoint] * nobjs, list(map(lambda x: x[1], results))))
(base) linux3:~/data/workspace/REMOTune> cat patue/utils/mesmo.py
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_approximation import RBFSampler


class GaussianProcess:

    def __init__(self, dim, kernel="RBF"):
        self.dim = dim
        self.kernel = None
        self.kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e2))
        self.beta = 1e6
        self.xValues = []
        self.yValues = []
        self.yValuesNorm = []
        self.model = GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=5)


    def fitNormal(self):
        y_mean = np.mean(self.yValues)
        y_std = self.getstd()
        self.yValuesNorm = (self.yValues - y_mean)/y_std
        self.model.fit(self.xValues, self.yValuesNorm)


    def fitModel(self):
        self.model.fit(self.xValues, self.yValues)


    def addSample(self, x, y):
        self.xValues.append(x)
        self.yValues.append(y)


    def getPrediction(self, x):
        mean, std = self.model.predict(x.reshape(1,-1), return_std=True)
        if std[0] == 0:
            std[0] = np.sqrt(1e-5) * self.getstd()
        return mean, std


    def getmean(self):
        return np.mean(self.yValues)


    def getstd(self):
        y_std = np.std(self.yValues)
        if y_std == 0:
            y_std = 1
        return y_std



class MaxvalueEntropySearch(object):

    def __init__(self, GPmodel):
        self.GPmodel = GPmodel
        self.y_max = max(GPmodel.yValues)
        self.d = GPmodel.dim


    def Sampling_RFM(self):
        self.rbf_features = RBFSampler(gamma=1/(2*self.GPmodel.kernel.length_scale**2), n_components=1000, random_state=1)
        X_train_features = self.rbf_features.fit_transform(np.asarray(self.GPmodel.xValues))

        A_inv = np.linalg.inv((X_train_features.T).dot(X_train_features) + np.eye(self.rbf_features.n_components)/self.GPmodel.beta)
        self.weights_mu = A_inv.dot(X_train_features.T).dot(self.GPmodel.yValues)
        weights_gamma = A_inv / self.GPmodel.beta
        self.L = np.linalg.cholesky(weights_gamma)


    def weigh_sampling(self):
        random_normal_sample = np.random.normal(0, 1, np.size(self.weights_mu))
        self.sampled_weights = np.c_[self.weights_mu] + self.L.dot(np.c_[random_normal_sample])


    def f_regression(self,x):

        X_features = self.rbf_features.fit_transform(x.reshape(1,len(x)))
        return -(X_features.dot(self.sampled_weights))


    def single_acq(self, x,maximum):
        mean, std = self.GPmodel.getPrediction(x)
        mean=mean[0]
        std=std[0]
        if maximum < max(self.GPmodel.yValues)+5/self.GPmodel.beta:
            maximum=max(self.GPmodel.yValues)+5/self.GPmodel.beta

        normalized_max = (maximum - mean) / std
        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        if (cdf==0):
            cdf=1e-30
        return   -(normalized_max * pdf) / (2*cdf) + np.log(cdf)