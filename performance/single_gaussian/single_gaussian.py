#!/usr/bin/env python
# coding: utf-8

from mpdaf.obj import Cube
import numpy as np
import pandas as pd
import os
import time
from viztracer import VizTracer


from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import dfisher_2022a as dfi

time_start = time.time()
path = "/Users/SuperTiger/ADACS/repo/SS2022A-DFisher/dfisher_2022a/dfisher_2022a/tests/data"

file = path + "/" + "single_gaussian_muse_size.fits"

cube = Cube(file)


model = dfi.models.Lm_Const_1GaussModel
small = cube[:,100:150,100:150]
# small = cube
axis_x = small.data.shape[1]
axis_y = small.data.shape[2]

da = small.data
xwav = small.wave.coord()
fc = dfi.fits.base.FitCube(small, model)

# def fit_single_spaxel(data, model, x, i, j):
#         spaxel = data[:, i, j]
#         m = model()
#         params = m.guess(spaxel, x)
#         result = m.fit(spaxel, params, x=x)
#         name = os.getpid()
#         print("subprocess: ", name, " pixel: ", i, j)

#         # type: lmfit ModelResult 
#         return result

# res = np.zeros((axis_x, axis_y))
# df = pd.DataFrame(res)
a = np.random.rand(10)

def f(x):
    return a[x]

if __name__ == "__main__":
    # tracer = VizTracer()
    
    # fc = dfi.fits.base.FitCube(small, model)
    # tracer.start()
    fc.fit_all(4, 1000)
    # tracer.stop()
    # tracer.save()
    time_m = time.time()
    print("--- %s seconds ---" % (time_m - time_start))
    
    # fc._map_all(res, 10)
    # fc.extract_info()
    # fc.write_txt('fiting_results.txt')
    # print(fc.res[])
    # with ProcessPoolExecutor(max_workers=4) as pool:
    #      for i in range(a.shape[0]):
    #          fu = pool.submit(f, i)
    #          print("queue count: ", pool._queue_count)
    #          print("pending work items: ", pool._pending_work_items)
    #          if pool._queue_count == 5:
    #             fu.result()
    #             print("get results, queue count: ", pool._queue_count)
    #             print("get resluts, pending work items: ", pool._pending_work_items)
            
            


            
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
















