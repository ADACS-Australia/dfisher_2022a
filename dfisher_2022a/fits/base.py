# interface between cube and model
import numpy as np
import pandas as pd
import os
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
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
        self._create_results_placeholder()

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

        if spaxel.mask.all():
            print("masked all data: ", i, j)
            result = None

        else:
            m = self.model()
            params = m.guess(spaxel, self.x)

            result = m.fit(spaxel, params, x=self.x, weights=sp_weight)
            name = os.getpid()
            print("subprocess: ", name, " pixel: ", i, j)

            # type: lmfit ModelResult 
            
        return result

    def _create_results_placeholder(self):
        axis_x = self.data.shape[1]
        axis_y = self.data.shape[2]
        res = np.zeros((axis_x*axis_y, 3), dtype=np.int32)
        df = pd.DataFrame(res, columns = ["row", "col", "Result"])
        
        for i in range(axis_x):
            df["row"][i*axis_y:(i+1)*axis_y] = i
            df["col"][i*axis_y:(i+1)*axis_y] = list(range(axis_y))
        
        # df = df.astype({"row":np.int32, "col":np.int32})
        self.res = df

    def fit_all(self, nprocess):
        axis_x = self.data.shape[1]
        axis_y = self.data.shape[2]
        res_collect = []
        with ProcessPoolExecutor(max_workers=nprocess) as pool:
            for i in range(axis_x):
                for j in range(axis_y):
                    idx = i*axis_y + j
                    future = pool.submit(self.fit_single_spaxel, i, j)
                    self.res["Result"][idx] = future.result()
                    # res_collect.append(future)
        # return res_collect
                   
        #TODO: fit selected region
            

        # p = Pool(processes=nprocess)
        # axis_x = self.data.shape[1]
        # axis_y = self.data.shape[2]
        # p.map(self._fit_single_index, list(range(axis_x*axis_y)))



class FitResult():
    def __init__(self, fitresult):
        self.results = fitresult

    def _get_parameter(self):
        pass

    def to_csv():
        pass

    def to_hdf5():
        pass




