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