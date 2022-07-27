'''
    This is the base class for all fits.
    It contains the basic functions to fit a model to selected spaxels in a data cube.
'''
__all__ = ["FitCube", "FitResult", "FitOutput"]

import itertools
import multiprocessing as mp
import os
import threading
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from multiprocessing import RawArray, sharedctypes

import line_profiler
import numpy as np
import pandas as pd
from viztracer import VizTracer, log_sparse

from ..io.cube import FitReadyCube
from ..models import Lm_Const_1GaussModel, GaussianConstModelH
from ..models.base import guess_1gauss

# to inheritate from parent process (shared object), it looks like we should use fork, instead of spawn



ctx = mp.get_context('fork')
# 

#TODO: write new error class for fitting spaxel (reference: fc mpdaf_ext)

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

    
class FitOutput():
    _default = ["success", "aic", "bic", "chisqr", "redchi"]
    _pars = []
    col_dict = {}

    def __init__(self, result):
        self._get_default_columns(result)


        # self._get_out_array()

    def _get_par_info(self,result):
        pars = result.params
        for key, val in pars.items():
            self.col_dict[key] = val.value
            self.col_dict[key+"_err"] = val.stderr
    
    def _get_default_info(self, res):
        for name in self._default:
            val = getattr(res, name)
            self.col_dict[name] = getattr(res, name)

    def _get_default_columns(self, res):
        self._get_default_info(res)
        self._get_par_info(res)
        self.columns = list(self.col_dict.keys())

    def get_col_value(self):
        return list(self.col_dict.values())
        

    def _validate_name(name, res):
        """valid user input name"""
        pass

    def set_columns(self, names):
        """update column names according to user input"""
        pass
        
    
        


    

    




