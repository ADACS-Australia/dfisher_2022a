# interface between cube and model
import numpy as np
from ..io.cube import FitReadyCube
from ..models import Lm_Const_1GaussModel

class FitCube():
    def __init__(self, fitreadycube, model):
        self.x = fitreadycube.cube.wave.coord()
        self.data = fitreadycube.cube.data
        self.var = fitreadycube.cube.var
        self._get_weight()
        self.model = model


    def _guess_params(self, y, x):
        params = self.model.guess(y, x)
        return params

    def _get_weight(self):
        if self.var:
            weights = 1 / np.sqrt(np.abs(self.var))
        else:
            weights = None
        self.weights = weights

    def fit_single_spaxel(self, loc=[0,0]):
        spaxel = self.data[:, loc[0], loc[1]]

        result = self.model.fit(spaxel, x=self.x)

    def fit_all():


