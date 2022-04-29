# interface between cube and model
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from ..io.cube import FitReadyCube
from ..models import Lm_Const_1GaussModel

class FitCube():
    def __init__(self, cube, model):
        self.x = cube.wave.coord()
        self.data = cube.data
        self.var = cube.var
        self._get_weight()
        self.model = model

    def _get_weight(self):
        if self.var is not None:
            weights = 1 / np.sqrt(np.abs(self.var))
        else:
            weights = None
        self.weights = weights

    def fit_single_spaxel(self, i, j):
        spaxel = self.data[:, i, j]
        if self.weights is not None:
            sp_weight = self.weights[:,i,j]
        else:
            sp_weight = None
        m = self.model()
        params = m.guess(spaxel, self.x)

        result = m.fit(spaxel, params, x=self.x, weights=sp_weight)

        # type: lmfit ModelResult 
        return result

    def _fit_single_index(self,i):
        axis_0 = self.data.shape[0]
        axis_x = self.data.shape[1]
        axis_y = self.data.shape[2]
        flat_data = self.data.reshape(axis_0, axis_x*axis_y)
        if self.weights is not None:
            flat_weights = self.weights.reshape(axis_0, axis_x*axis_y)
            single_weight = flat_weights[:,i]
        else:
            single_weight = None
        single = flat_data[:,i]

        m = self.model()
        params = m.guess(single, self.x)

        result = m.fit(single, params, x=self.x, weights=single_weight)


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




