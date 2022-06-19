#!/usr/bin/env python
# coding: utf-8

from mpdaf.obj import Cube
import time
import dfisher_2022a as dfi

time_start = time.time()
path = "./../../dfisher_2022a/tests/data"
file = path + "/" + "single_gaussian_muse_size.fits"

cube = Cube(file)


model = dfi.models.Lm_Const_1GaussModel
small = cube[:,100:120,100:120]
# small = cube
axis_x = small.data.shape[0]
axis_y = small.data.shape[1]

da = small.data
xwav = small.wave.coord()
fc = dfi.fits.base.FitCube(small, model)
time_m = time.time()
# if __name__ == "__main__":

# fit spaxels parallelly using multiprocessing
fc.fit_all(4, 100)

time_e = time.time()
print("--- %s seconds ---" % (time_m - time_start))
print("--- %s seconds ---" % (time_e - time_m))
    
    









