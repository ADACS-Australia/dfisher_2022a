'''
    This is the base class for all fits.
    It contains the basic functions to fit a model to selected spaxels in a data cube.
'''
__all__ = ["FitCube", "CubeFitterLM", "ResultLM"]

import itertools
import multiprocessing as mp
import os
import threading
import time
import tracemalloc
import math
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from multiprocessing import RawArray, sharedctypes
from numpy.ma.core import MaskedArray

import line_profiler
import numpy as np
import pandas as pd
from viztracer import VizTracer, log_sparse

from ..io.cube import ProcessedCube
from ..models import Lm_Const_1GaussModel, GaussianConstModelH
from ..models.base import guess_1gauss
from ..exceptions import InputDimError, InputShapeError
from abc import ABC, abstractclassmethod, abstractmethod
from types import MethodType

# to inheritate from parent process (shared object), it looks like we should use fork, instead of spawn



ctx = mp.get_context('fork')
# 


# CPU COUNT
CPU_COUNT = os.cpu_count()
print(f"The number of cpus:{CPU_COUNT}")


def _get_custom_attr(top, cls):
    attr_names = dir(cls)
    for name in attr_names:
        attr = getattr(cls, name)
        if not (name.startswith("_") or 
        type(attr) is MethodType):
            print(f"{name}: {attr}")
            setattr(top, name, attr)



#TODO: write new error class for fitting spaxel (reference: fc mpdaf_ext)


class CubeFitter(ABC):

    @abstractmethod
    def __init__(self, data, weight, x, model, fit_method):
        self._data = data
        self._weight = weight
        self.x = x
        self.model = model
        self.method = fit_method
        self.result = None
    
    @abstractclassmethod
    def fit_cube(self):
        """method to fit data cube"""

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
    
    def _fit_single_spaxel(self, pix_id: int):        
        # shared memory
        rresult = np.ctypeslib.as_array(shared_res_c)
        rdata = np.ctypeslib.as_array(shared_data)
        sp = rdata[pix_id,:]
        if self.weight is not None:
            rweight = np.ctypeslib.as_array(shared_weight)
            sp_weight = rweight[pix_id,:]
        else:
            sp_weight = None
        
        # start fitting    
        m = self.model()
        params = m.guess(sp, self.x)
        res = m.fit(sp, params, x=self.x, weights=sp_weight, method=self.fit_method,
        iter_cb=self.iter_cb, scale_covar=self.scale_covar, verbose=self.verbose, 
        fit_kws=self.fit_kws, nan_policy=self.nan_policy, calc_covar=self.calc_covar, 
        max_nfev=self.max_nfev)

        # read fitting result
        out = self._read_fit_result(res)
        rresult[pix_id,:] = out

        # temp: process information
        name = os.getpid()
        current, peak = tracemalloc.get_traced_memory()
        # print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
        print("subprocess: ", name, " pixel: ", pix_id)

    def _set_default_chunksize(self, ncpu):
        return math.ceil(self.data.shape[0]/ncpu)

    def fit_cube(self, nprocess=CPU_COUNT, chunksize=None):
        if chunksize is None:
            chunksize = self._set_default_chunksize(nprocess)

        datac = np.ctypeslib.as_ctypes(self.data)
        global shared_data
        shared_data = sharedctypes.RawArray(datac._type_, datac)

        if self.weight is not None:
            weightc = np.ctypeslib.as_ctypes(self.weight)
            global shared_weight
            shared_weight = sharedctypes.RawArray(weightc._type_, weightc)

        resc = np.ctypeslib.as_ctypes(self.result)
        global shared_res_c
        shared_res_c = mp.sharedctypes.RawArray(resc._type_, resc)

        # ctx = get_context('fork')
        npix = self.result.shape[0]
        p = ctx.Pool(processes=nprocess)
        print("start pooling")
        p.map(self._fit_single_spaxel, list(range(npix)), chunksize=chunksize)
        print("finish pooling")

        res = np.ctypeslib.as_array(shared_res_c)
        self.result = res

        # temp print
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
        tracemalloc.stop()

  
    
   
    

   

        
class FitCube():
        
    def __init__(self, cube, model):        
        self.x = cube.wave.coord()
        self.data = np.transpose(cube.data, axes=(1,2,0)).copy()
        print(self.data.data.flags)
        self.var = cube.var
        self._get_weight()
        self.model = model
        self._create_results_placeholder()
        self.name = None

    def _get_weight(self):      
        if self.var is not None:
            self.var = np.transpose(self.var, axes=(1,2,0)).copy()
            weights = 1 / np.sqrt(np.abs(self.var))
        else:
            weights = None
        self.weights = weights

    def fit_single_spaxel(self, i, j, enable_res=False):        
        spaxel = self.data[i,j,:]

        if self.weights is not None:
            sp_weight = self.weights[i,j,:]
        else:
            sp_weight = None

        if enable_res == True:
            axis_y = self.data.shape[1]
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

    #TODO: 1. try to share the final results only 
    #TODO: 2. share input data too
    #TODO: 3. writeout all the necessary information

    def _create_results_placeholder(self):
        axis_spec = self.data.shape[2]
        axis_x = self.data.shape[0]
        axis_y = self.data.shape[1]
        pix = axis_x*axis_y

        stats_col = 5 # TODO: CHANGE HARD CODED TO USER CHOICE

        m = self.model()
        pars = m.make_params()
        pars_col = 2 * len(pars) # number of parameters, including independent and dependent
        
        cols = stats_col + pars_col

        res = np.zeros((pix,cols))
        res[:] = np.nan
        self.res = res
       


    # @profile
    def _fit_single_index(self, i):
        start = time.time()
        # current, peak = tracemalloc.get_traced_memory()
        # print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
        records = np.ctypeslib.as_array(shared_records_c)
        
        # shared memory
        res = np.ctypeslib.as_array(shared_res_c)
        rdata = np.ctypeslib.as_array(shared_data)
        # rid = id(rdata)
        # print(rdata.flags)
        # current, peak = tracemalloc.get_traced_memory()
        # print(f"after loading ctypes, Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")

        # axis_spec = self.data.shape[0]
        # axis_x = self.data.shape[1]
        # axis_y = self.data.shape[2]
        # pix = axis_x*axis_y
        # rdata = self.data.reshape(axis_spec, pix)
        sp = rdata[i,:]
        # print(rdata.flags)
        # print("wave coord: ", type(self.x), self.x)

        if self.weights is not None:
            rweights = np.ctypeslib.as_array(shared_weights)
            # rweights = self.weights.reshape(axis_spec, pix)
            sp_weight = rweights[i,:]
        else:
            sp_weight = None
        
                     
        m = self.model()
        params = m.guess(sp, self.x)
        

        result = m.fit(sp, params, x=self.x, weights=sp_weight)
        g = result.params.get('g1_height')
        res[i,0] = g
        end = time.time()
        records[i] = end -start
        name = os.getpid()
        current, peak = tracemalloc.get_traced_memory()
        # print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
        print("subprocess: ", name, " pixel: ", i)
        
    def fit_all(self, nprocess, chunksize=10):
        tracemalloc.start()
        axis_spec = self.data.shape[2]
        axis_x = self.data.shape[0]
        axis_y = self.data.shape[1]
        pix = axis_x*axis_y
        rdata = self.data.reshape(pix,axis_spec)
        # print(self.data.flags)
        # print(rdata.flags)
        rdatac = np.ctypeslib.as_ctypes(rdata)
        global shared_data
        shared_data = sharedctypes.RawArray(rdatac._type_, rdatac)

        if self.weights is not None:
            rweights = self.weights.reshape(pix,axis_spec)
            rweightsc = np.ctypeslib.as_ctypes(rweights)
            global shared_weights
            shared_weights = sharedctypes.RawArray(rweightsc._type_, rweightsc)

        resc = np.ctypeslib.as_ctypes(self.res)
        global shared_res_c
        shared_res_c = mp.sharedctypes.RawArray(resc._type_, resc)

        # just for timing purpose
        records = np.zeros(pix)
        records[:] = np.nan
        recordsc = np.ctypeslib.as_ctypes(records)
        global shared_records_c
        shared_records_c = sharedctypes.RawArray(recordsc._type_, recordsc)

        # ctx = get_context('fork')
        p = ctx.Pool(processes=nprocess)
        print("start pooling")
        p.map(self._fit_single_index, list(range(pix)), chunksize=chunksize)
        print("finish pooling")

        res = np.ctypeslib.as_array(shared_res_c)
        self.res = res

        records = np.ctypeslib.as_array(shared_records_c)
        self.records = records
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
        tracemalloc.stop()
    
    def fit_all_serial(self):
        axis_spec = self.data.shape[2]
        axis_x = self.data.shape[0]
        axis_y = self.data.shape[1]
        pix = axis_x*axis_y
        rdata = self.data.reshape(pix, axis_spec)

        if self.weights is not None:
            rweights = self.weights.reshape(pix, axis_spec)


        # just for timing purpose
        records = np.zeros(pix)
        records[:] = np.nan
        
        for i in range(pix):
            
            start = time.time()
            sp = rdata[i,:]
            
            if self.weights is not None:            
                sp_weight = rweights[i,:]
            else:
                sp_weight = None
            
            if sp.mask.all():
                print("the spectrum is masked")
                continue
            else:
                m = self.model()
                params = m.guess(sp, self.x)


                result = m.fit(sp, params, x=self.x, weights=sp_weight)
                out = FitOutput(result)
                out = out.get_col_value()

                print("columns value: ", out)
                nfev = result.nfev
                # g = result.params.get('g1_height')
                # self.res[i,0] = g
                # self.res[i] = self._write_fit_output(result)
                self.res[i] = out
                end = time.time()
                records[i] = end -start
                name = os.getpid()
                print("subprocess: ", name, " pixel: ", i, " nfev: ", nfev)
            np.save("res", self.res)
            
    def record_guess(self):
        axis_spec = self.data.shape[2]
        axis_x = self.data.shape[0]
        axis_y = self.data.shape[1]
        pix = axis_x*axis_y
        rdata = self.data.reshape(pix, axis_spec)

        if self.weights is not None:
            rweights = self.weights.reshape(pix, axis_spec)

        guess_res = np.zeros((pix, 4))

        for i in range(pix):
            
            start = time.time()
            sp = rdata[i,:]
            if self.weights is not None:            
                sp_weight = rweights[i,:]
            else:
                sp_weight = None
            guess_res[i] = guess_1gauss(sp, self.x)

        out_file = os.getcwd() + "/guess"

        np.save(out_file, guess_res)


    # @profile
    def single_out_fitting(self):
        tracemalloc.start()
        axis_spec = self.data.shape[0]
        axis_x = self.data.shape[1]
        axis_y = self.data.shape[2]
        pix = axis_x*axis_y
        rdata = self.data.reshape(axis_spec, pix)
        rdatac = np.ctypeslib.as_ctypes(rdata)
        global shared_data
        shared_data = sharedctypes.RawArray(rdatac._type_, rdatac)

        if self.weights is not None:
            rweights = self.weights.reshape(axis_spec, pix)
            rweightsc = np.ctypeslib.as_ctypes(rweights)
            global shared_weights
            shared_weights = sharedctypes.RawArray(rweightsc._type_, rweightsc)

        resc = np.ctypeslib.as_ctypes(self.res)
        global shared_res_c
        shared_res_c = mp.sharedctypes.RawArray(resc._type_, resc)

        # just for timing purpose
        records = np.zeros(pix)
        records[:] = np.nan
        recordsc = np.ctypeslib.as_ctypes(records)
        global shared_records_c
        shared_records_c = sharedctypes.RawArray(recordsc._type_, recordsc)

        for i in range(pix):
            self._fit_single_index(i)

        res = np.ctypeslib.as_array(shared_res_c)
        self.res = res

        
    


    def _fit_all(self, nprocess, batch_size=10):
        axis_x = self.data.shape[1]
        axis_y = self.data.shape[2]
        
        
        
        with ProcessPoolExecutor(max_workers=nprocess) as pool:
            for i in range(axis_x):
                res_collect = []
                for j in range(axis_y):
                    # idx = i*axis_y + j
                    future = pool.submit(self.fit_single_spaxel, i, j, enable_res=True)
                    # self.res["Result"][idx] = future.result()
                    res_collect.append(future)
                self.map_all(res_collect)
                print("mapping for row: ", i)
        
        #TODO: fit selected region


    def fit_map_all(self, nprocess, batch=100):
        p = Pool(processes=nprocess)
        axis_x = self.data.shape[1]
        axis_y = self.data.shape[2]
        sample_size = axis_x*axis_y
        res = []
        for result in p.map(self._fit_single_index, [batch for _ in range(sample_size//batch)]):
            res.append(result)


            
        # p.map(self._fit_single_index, list(range(axis_x*axis_y)))
        
        
                   
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

    def map_all(self, res_collect):
        for ele in res_collect:
            future_result = ele.result()
            idx = future_result[0]
            self.res["Result"][idx] = future_result[1]
                    
    # def fit_all(self, nprocess, nthread):
    #     res_collect = self._fit_all(nprocess = nprocess)
    #     self._map_all(res_collect, nthread=nthread)

    def _extract_info(self, name):
        self.name = name
        self.res[name] = self.res["Result"].parallel_apply(self._extract_info_single)

    def _extract_info_single(self, res):
        fres = FitResult()
        names = fres.get_info(res)
        self.names = names
        if self.name is not None:
            info = getattr(fres, self.name)
            return info

   

    def write_txt(self, filename):
        df = self.res.drop(axis=1, columns="Result")
        out = Output(df)
        out.to_csv(filename)



        
# class ResultBase():

    
#     def __init__(self):
#         self._init_attr()

#     def _init_attr(self):
#         for attr in self._attr:
#             setattr(self, attr, None)

#     def write(self):
#         """write out result to a file"""
#         pass

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
        return self.snr.flatten()

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
        _get_custom_attr(self, cls)

    def save(self, save_fitdata=True):
        df = self._create_result_df()
        self._save_result(df)
        if save_fitdata:
            self._save_fit_input_data()
            

        
    


    

        


        
        



class FitResult():
    # read in lmfit ModelResult
    def __init__(self):
        pass
        # self.result = fitresult
        # self.dict = {}


    def _get_parameter_names(self, fitresult):
        var_name = fitresult.var_names
        return var_name

    def _get_parameter_values(self, fitresult, var_name):
        param = fitresult.params.get(var_name)
        value = param.value
        err_name = var_name + "_err"
        stderr = param.stderr
        setattr(self, var_name, value)
        setattr(self, err_name, stderr)
        # self.dict[var_name] = value
        # self.dict[err_name] = stderr
        # return value, stderr

    def _get_parameters(self, fitresult):
        names = self._get_parameter_names(fitresult)
        for name in names:
            self._get_parameter_values(fitresult, name)

    def _get_stats(self, fitresult):
        setattr(self, "success",fitresult.success)
        setattr(self, "chisqr",fitresult.chisqr)
        setattr(self, "redchi",fitresult.redchi)

    def get_info(self, fitresult):
        self._get_parameters(fitresult)
        self._get_stats(fitresult)
        names = []
        for key in self.__dict__.keys():
            names.append(key)
        return names

    
#NOTE: TO DELETE THE RESULTLM CLASS
# class ResultLM():
#     """fit result (fitter: lmfit)"""
    
#     _default = ["success", "aic", "bic", "chisqr", "redchi"]
#     _pars = []
#     col_dict = {}

#     # TODO: allow user input list of output columns
#     def __init__(self, res, attr_list, param_name):
#         """res: ModelResult"""
#         self._get_default_columns(res)


#         # self._get_out_array()

#     def _get_par_info(self,res, params_name):
#         pars = res.params
#         vals = []

#         for name in params_name:
#             val = pars[name]
#             vals += [val.value, val.stderr]
#         # for key, val in pars.items():
#         #     self.col_dict[key] = val.value
#         #     self.col_dict[key+"_err"] = val.stderr
    
#     def _get_default_info(self, res, attr_list):
#         vals = []
#         for name in attr_list:           
#             val = getattr(res, name)
#             vals.append(val)

#     def _get_default_columns(self, res):
#         self._get_default_info(res)
#         self._get_par_info(res)
#         self.columns = list(self.col_dict.keys())

#     def get_col_value(self):
#         return list(self.col_dict.values())
        

#     def _validate_name(name, res):
#         """valid user input name"""
#         pass

    
        
    
        


    

    




