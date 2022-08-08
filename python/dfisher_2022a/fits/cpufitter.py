__all__ = ["CubeFitterLM", "ResultLM"]

import itertools
import math
import multiprocessing as mp
import os
import threading
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from multiprocessing import RawArray, sharedctypes
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import pandas as pd
from numpy.ma.core import MaskedArray
from sympy import N
from viztracer import VizTracer, log_sparse

from ..exceptions import InputDimError, InputShapeError
from ..utils import get_custom_attr
from .base import CubeFitter

# to inheritate from parent process (shared object), it looks like we should use fork, instead of spawn



ctx = mp.get_context('fork')
# 


# CPU COUNT
CPU_COUNT = os.cpu_count()
print(f"The number of cpus:{CPU_COUNT}")






#TODO: write new error class for fitting spaxel (reference: fc mpdaf_ext)




class CubeFitterLM(CubeFitter):
    """use lmfit as fitter (CPU fitter)"""

    _lmfit_result_default = [
    'aic', 'bic', 'chisqr',
    'ndata', 'nfev',
    'nfree', 'nvarys', 'redchi',
    'success']

    # TODO: SOME OF THE INIT PARAMETERS ARE PARSED DIRECTLY FROM LMFIT MODEL.FIT
    # CHECK WHETHER ALL OF THEM ARE NEEDED OR ALLOWED IN CUBE FIT
    # SOME MAY SLOW DOWN THE PROCESS, OR PRODUCE INCONPATIBLE RESULTS
    # WITH THE CURRENT RESULT HANDLING SETTING
    def __init__(self, data, weight, x, model, fit_method='leastsq', 
    iter_cb=None, scale_covar=True, verbose=False, fit_kws=None, 
    nan_policy=None, calc_covar=True, max_nfev=None):
        self._data = data
        self._weight = weight
        self.x = x
        self.model = model
        self.fit_method = fit_method
        self.iter_cb = iter_cb
        self.scale_covar = scale_covar
        self.verbose = verbose
        self.fit_kws = fit_kws
        self.nan_policy = nan_policy
        self.calc_covar = calc_covar
        self.max_nfev = max_nfev
        self.result = None
        self._input_data_check()
        self._prepare_data()
        self._create_result_container()

    def _input_data_check(self):
        if self._data.ndim != 3:
            raise InputDimError(self._data.ndim)
        if self._weight is not None and self._weight.shape != self._data.shape:
            raise InputShapeError("Weight must be either None or of the same shape as data.")
        if len(self.x) != self._data.shape[0]:
            raise InputShapeError("The length of x must be equal to the length of the spectrum.")
        
    def _convert_array(self, arr):
        arr = np.transpose(arr, axes=(1,2,0)).copy()
        axis_y, axis_x = arr.shape[0], arr.shape[1]
        axis_d = arr.shape[2]
        pix = axis_y * axis_x
        arr = arr.reshape(pix, axis_d)
        return arr

    def _prepare_data(self):
        """prepare data for parallel fitting"""
        self.data = self._convert_array(self._data)
        if self._weight is not None:
            self.weight = self._convert_array(self._weight)
        else:
            self.weight = self._weight

    def _get_param_names(self):
        """get the param names of the model"""
        m = self.model()
        _pars = m.make_params()
        _pars_name = list(_pars.valuesdict().keys())
        self._pars_name = _pars_name

        return _pars_name

    def _set_result_columns(self):
        """set param columns: [name, err] for each param"""
        _pars_name = self._get_param_names()
        _pars_col = []
        for p in _pars_name:
            _pars_col += [p, p+"_err"]
        self.result_columns = self._lmfit_result_default + _pars_col
        
    def _create_result_container(self):
        """create result array with nan value"""
        self._set_result_columns()
        n_cols = len(self.result_columns)
        result = np.zeros((self.data.shape[0], n_cols))
        result[:] = np.nan
        self.result = result

    def _read_fit_result(self, res):
        """res: ModelResult; read according to result columns"""
        vals = []
        for name in self._lmfit_result_default:
            val = getattr(res, name)
            vals.append(val)

        pars = res.params
        for name in self._pars_name:
            val = pars[name]
            vals += [val.value, val.stderr]

        return vals
    
    # def _fit_single_spaxel(self, pix_id: int):        
    #     # shared memory
    #     rresult = np.ctypeslib.as_array(shared_res_c)
    #     rdata = np.ctypeslib.as_array(shared_data)
    #     sp = rdata[pix_id,:]
    #     if self.weight is not None:
    #         rweight = np.ctypeslib.as_array(shared_weight)
    #         sp_weight = rweight[pix_id,:]
    #     else:
    #         sp_weight = None
        
    #     # start fitting    
    #     m = self.model()
    #     params = m.guess(sp, self.x)
    #     res = m.fit(sp, params, x=self.x, weights=sp_weight, method=self.fit_method,
    #     iter_cb=self.iter_cb, scale_covar=self.scale_covar, verbose=self.verbose, 
    #     fit_kws=self.fit_kws, nan_policy=self.nan_policy, calc_covar=self.calc_covar, 
    #     max_nfev=self.max_nfev)

    #     # read fitting result
    #     out = self._read_fit_result(res)
    #     rresult[pix_id,:] = out

    #     # temp: process information
    #     name = os.getpid()
    #     current, peak = tracemalloc.get_traced_memory()
    #     # print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
    #     print("subprocess: ", name, " pixel: ", pix_id)

    def _fit_single_spaxel(self, data: np.ndarray, weight: np.ndarray or None, 
                            mask: np.ndarray, x: np.ndarray, 
                            resm: shared_memory.SharedMemory, pix_id: int):        
        if not mask[pix_id].all():
            print("enter")
            inx = np.where(~mask[pix_id])
            sp = data[pix_id][inx]
            sp_x = x[inx]
            sp_weight = None
            if weight is not None:
                sp_sweight = weight[pix_id][inx]
    
            # start fitting    
            m = self.model()
            params = m.guess(sp, sp_x)
            res = m.fit(sp, params, x=sp_x, weights=sp_weight, method=self.fit_method,
            iter_cb=self.iter_cb, scale_covar=self.scale_covar, verbose=self.verbose, 
            fit_kws=self.fit_kws, nan_policy=self.nan_policy, calc_covar=self.calc_covar, 
            max_nfev=self.max_nfev)

            # read fitting result
            result = np.ndarray(self.result.shape, self.result.dtype, buffer=resm.buf)
            out = self._read_fit_result(res)
            result[pix_id,:] = out

            # # temp: process information
            name = os.getpid()
            # current, peak = tracemalloc.get_traced_memory()
            # print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
            print("subprocess: ", name, " pixel: ", pix_id)

    def _set_default_chunksize(self, ncpu):
        return math.ceil(self.data.shape[0]/ncpu)

    def fit_cube(self, nprocess=CPU_COUNT, chunksize=None):
        """fit data cube parallelly"""
        if chunksize is None:
            chunksize = self._set_default_chunksize(nprocess)

        with SharedMemoryManager() as smm:
            print("put in shared memory")
            shm_dd = smm.SharedMemory(size=self.data.nbytes)
            shm_dm = smm.SharedMemory(size=self.data.mask.nbytes)
            shm_x = smm.SharedMemory(size=self.x.nbytes)
            shm_r = smm.SharedMemory(size=self.result.nbytes)
            sdd = np.ndarray(self.data.data.shape, dtype=self.data.data.dtype, buffer=shm_dd.buf)
            sdd[:] = self.data.data[:]
            sdm = np.ndarray(self.data.mask.shape, dtype=self.data.mask.dtype, buffer=shm_dm.buf)
            sdm[:] = self.data.mask[:]
            sx = np.ndarray(self.x.shape, dtype=self.x.dtype, buffer=shm_x.buf)
            sx[:] = self.x[:]
            sr = np.ndarray(self.result.shape, dtype=self.result.dtype, buffer=shm_r.buf)
            sr[:] = self.result[:]
            # print("result before fitting: ", sr)
            
            sw = None
            if self.weight is not None:
                shm_wd = smm.SharedMemory(size=self.weight.nbytes)
                sw = np.ndarray(self.weight.shape, dtype=self.weight.dtype, buffer=shm_wd.buf)
                sw[:] = self.weight.data[:]
        
            
            pool = mp.Pool(processes=nprocess)
            npix = self.result.shape[0]
            print("start pooling")
            pool.map(partial(self._fit_single_spaxel, sdd, sw, sdm, sx, shm_r), range(npix), chunksize=chunksize)
            print("finish pooling")
            # print("result after fitting: ", sr)
        self.result[:] = sr[:]
        

    # def fit_cube(self, nprocess=CPU_COUNT, chunksize=None):
    #     """fit data cube parallelly"""
    #     if chunksize is None:
    #         chunksize = self._set_default_chunksize(nprocess)

    #     datac = np.ctypeslib.as_ctypes(self.data)
    #     global shared_data
    #     shared_data = sharedctypes.RawArray(datac._type_, datac)

    #     if self.weight is not None:
    #         weightc = np.ctypeslib.as_ctypes(self.weight)
    #         global shared_weight
    #         shared_weight = sharedctypes.RawArray(weightc._type_, weightc)

    #     resc = np.ctypeslib.as_ctypes(self.result)
    #     global shared_res_c
    #     shared_res_c = mp.sharedctypes.RawArray(resc._type_, resc)

    #     # ctx = get_context('fork')
    #     npix = self.result.shape[0]
    #     p = ctx.Pool(processes=nprocess)
    #     print("start pooling")
    #     p.map(self._fit_single_spaxel, list(range(npix)), chunksize=chunksize)
    #     print("finish pooling")

    #     res = np.ctypeslib.as_array(shared_res_c)
    #     self.result = res

    #     # temp print
    #     current, peak = tracemalloc.get_traced_memory()
    #     print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
    #     tracemalloc.stop()
    
    def fit_serial(self):
        """"fit data cube serially"""
        for i in range(self.data.shape[0]):
            sp = self.data[i,:]
            if self.weight is not None:
                sp_weight = self.weight[i,:]
            else:
                sp_weight = None

            
            if sp.mask.all():
                print("the spectrum is masked")
                continue
            else:
                m = self.model()
                params = m.guess(sp, self.x)
                res = m.fit(sp, params, x=self.x, weights=sp_weight, method=self.fit_method,
                iter_cb=self.iter_cb, scale_covar=self.scale_covar, verbose=self.verbose, 
                fit_kws=self.fit_kws, nan_policy=self.nan_policy, calc_covar=self.calc_covar, 
                max_nfev=self.max_nfev)

                out = self._read_fit_result(res)
                self.result[i,:] = out

class ResultLM():
    _cube_attr = ["z", "line", "snr_threshold", "snrmap"]
    _fit_attr = ["fit_method", "result", "result_column"]
    _default = ["success", "aic", "bic", "chisqr", "redchi"]

    _save = ["data", "weight", "x"]

    def __init__(self, path="./"):
        self.path = path
        self._create_output_dir()

    def _create_output_dir(self):
        """create the output directory; the default dir is the current dir"""
        os.makedirs(self.path + "/out", exist_ok=True)

    @property
    def _flatsnr(self):
        return self.snr.filled(-999).flatten()

    def _create_result_df(self):
        df = pd.DataFrame(self.result, columns=self.result_columns)
        df['snr'] = self._flatsnr
        return df

    def _save_result(self, df):
        store = pd.HDFStore(self.path + "/out/result.h5")
        store.put("result", df)
        store.close()

    def _save_fit_input_data(self):
        data_dir = self.path + "/out/fitdata/"
        os.makedirs(data_dir, exist_ok=True)
        for name in self._save:
            val = getattr(self, name)
            data_name = data_dir + name
            if type(val) is MaskedArray:
                np.save(data_name + "_data", val.data)
                np.save(data_name + "_mask", val.mask)
            else:
                np.save(data_name, val)

    # def _write_fit_summary(self):

    def get_output(self, cls):
        get_custom_attr(self, cls)

    def save(self, save_fitdata=True):
        df = self._create_result_df()
        self._save_result(df)
        if save_fitdata:
            self._save_fit_input_data()
            

        
    


    

        


        
        





    
        
    
        


    

    




