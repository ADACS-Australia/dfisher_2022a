# interface between cube and model
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from ..io.cube import FitReadyCube
from ..models import Lm_Const_1GaussModel

class FitCube():
    def __init__(self, fitreadycube, model):
        self.x = fitreadycube.cube.wave.coord()
        self.data = fitreadycube.cube.data
        self.var = fitreadycube.cube.var
        self._get_weight()
        self.model = model

    def _get_weight(self):
        if self.var:
            weights = 1 / np.sqrt(np.abs(self.var))
        else:
            weights = None
        self.weights = weights

    def fit_single_spaxel(self, i, j):
        spaxel = self.data[:, i, j]
        m = self.model()
        params = m.guess(spaxel, self.x)

        result = self.model.fit(spaxel, params, x=self.x, weights=self.weights)

        # type: lmfit ModelResult 
        return result

    def _fit_single_index(self,i):
        axis_0 = self.data.shape[0]
        axis_x = self.data.shape[1]
        axis_y = self.data.shape[2]
        flat_data = self.data.reshape(axis_0,axis_x*axis_y)
        single = flat_data[:,i]
        m = self.model()
        params = m.guess(single, self.x)

        result = self.model.fit(single, params, x=self.x, weights=self.weights)


    def fit_all(self, nprocess):
        p = Pool(processes=nprocess)
        axis_x = self.data.shape[1]
        axis_y = self.data.shape[2]
        p.map(self._fit_single_index, list(range(axis_x*axis_y)))



class FitResult():
    def __init__(self, fitresult):
        self.results = fitresult

    def _get_parameter(self):
        pass

    def to_csv():
        pass

    def to_hdf5():
        pass




