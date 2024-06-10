from sklearn.base import BaseEstimator
from AG_marcartal1_natolmvil import AG
from sklearn.metrics import r2_score

class AGEvaluator(BaseEstimator):
    def __init__(self, seed=123, nInd=50, maxIter=100, pc=0.7, pm=0.1, k=4, porcentaje_elitismo=0.2):
        self.seed = seed
        self.nInd = nInd
        self.maxIter = maxIter
        self.pc = pc
        self.pm = pm
        self.k = k
        self.porcentaje_elitismo = porcentaje_elitismo

    def fit(self, X, y):
        self.ag = AG(datos_train=X, datos_test=None, seed=self.seed, nInd=self.nInd, maxIter=self.maxIter)
        self.X_train = X
        self.y_train = y
        return self

    def score(self, X, y):
        self.ag.X_train = self.X_train
        self.ag.y_train = self.y_train
        self.ag.X_test = X

        mejor_individuo, y_pred = self.ag.run()

        return r2_score(y, y_pred)

    def get_params(self, deep=True):
        return {
            'seed': self.seed,
            'nInd': self.nInd,
            'maxIter': self.maxIter,
            'pc': self.pc,
            'pm': self.pm,
            'k': self.k,
            'porcentaje_elitismo': self.porcentaje_elitismo
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
