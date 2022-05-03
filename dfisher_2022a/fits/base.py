# interface between cube and model
import numpy as np
import pandas as pd
import os
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
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

    def fit_single_spaxel(self, i, j, enable_res=False):
        spaxel = self.data[:, i, j]

        if self.weights is not None:
            sp_weight = self.weights[:,i,j]
        else:
            sp_weight = None

        if enable_res == True:
            axis_y = self.data.shape[2]
            idx = i*axis_y + j
            if spaxel.mask.all():
                print("masked all data: ", i, j)
                result = (idx, None)

            else:                
                m = self.model()
                params = m.guess(spaxel, self.x)

                result = (idx, m.fit(spaxel, params, x=self.x, weights=sp_weight))
                name = os.getpid()
                print("subprocess: ", name, " pixel: ", i, j)

                # type: lmfit ModelResult 
                
            return result
        else:
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
        
        self.res = df

    def _fit_all(self, nprocess):
        axis_x = self.data.shape[1]
        axis_y = self.data.shape[2]
        res_collect = []
        with ProcessPoolExecutor(max_workers=nprocess) as pool:
            for i in range(axis_x):
                for j in range(axis_y):
                    future = pool.submit(self.fit_single_spaxel, i, j, enable_res=True)
                    # self.res["Result"][idx] = future.result()
                    res_collect.append(future)
        
        #TODO: fit selected region

        # p = Pool(processes=nprocess)
        # axis_x = self.data.shape[1]
        # axis_y = self.data.shape[2]
        # p.map(self._fit_single_index, list(range(axis_x*axis_y)))
        return res_collect
                   
    def _map_result(self, res_ele):
        name = threading.current_thread().getName()
        print("threading: ", name)
        future_result = res_ele.result()
        idx = future_result[0]
        self.res["Result"][idx] = future_result[1]


    def _map_all(self, res_collect, nthread):
        with ThreadPoolExecutor(max_workers=nthread) as pool:
            for ele in res_collect:
                pool.submit(self._map_result, ele)
                

     
    def fit_all(self, nprocess, nthread):
        res_collect = self._fit_all(nprocess = nprocess)
        self._map_all(res_collect, nthread=nthread)





        



class FitResult():
    def __init__(self, fitresult):
        self.results = fitresult

    def _get_parameter(self):
        pass

    def to_csv():
        pass

    def to_hdf5():
        pass




