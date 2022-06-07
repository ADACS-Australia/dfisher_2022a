#!/usr/bin/env python
# coding: utf-8

from mpdaf.obj import Cube
import numpy as np
import os
import time
from viztracer import VizTracer


import dfisher_2022a as dfi
   
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
# tracer = VizTracer(max_stack_depth=5)
# tracer.start()

# fit spaxel serially 
fc.fit_all_serial()

# tracer.stop()
# tracer.save()
time_m = time.time()
print("--- %s seconds ---" % (time_m - time_start))
















