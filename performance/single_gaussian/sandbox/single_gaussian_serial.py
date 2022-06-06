#!/usr/bin/env python
# coding: utf-8

from mpdaf.obj import Cube
import numpy as np
import os
import time
from viztracer import VizTracer


import dfisher_2022a as dfi

def fit_all():
        
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

        # ctx = get_context('fork')
        p = ctx.Pool(processes=nprocess)
        print("start pooling")
        p.map(self._fit_single_index, list(range(pix)), chunksize=chunksize)
        print("finish pooling")

        res = np.ctypeslib.as_array(shared_res_c)
        self.res = res

        records = np.ctypeslib.as_array(shared_records_c)
        self.records = records
       
   

# tracer = VizTracer()
# tracer.start()
time_start = time.time()
path = "./../../../dfisher_2022a/tests/data"
file = path + "/" + "single_gaussian_muse_size.fits"

cube = Cube(file)


model = dfi.models.Lm_Const_1GaussModel
small = cube[:,100:120,100:120]
# small = cube
axis_x = small.data.shape[1]
axis_y = small.data.shape[2]

da = small.data
xwav = small.wave.coord()
fc = dfi.fits.base.FitCube(small, model)


# if __name__ == "__main__":
tracer = VizTracer(max_stack_depth=5)

# fc = dfi.fits.base.FitCube(small, model)
# tracer.register_exit()
tracer.start()
fc.fit_all_serial()
tracer.stop()
tracer.save()
time_m = time.time()
print("--- %s seconds ---" % (time_m - time_start))
    
    
            


            
    # with ProcessPoolExecutor(max_workers=10) as pool:
    #     for i in range(axis_x):
    #         for j in range(axis_y):
    #             # name = os.getpid()
    #             future = pool.submit(fit_single_spaxel, da, model, xwav, i, j)
    #             res = future.result()
    #             if res.success is True:
    #                 df[i][j] = future.result()
    #                 # print("get result")
    #             else:
    #                 print("failed")

print("--- %s seconds ---" % (time.time() - time_m))
print("--- %s seconds ---" % (time.time() - time_start))
















