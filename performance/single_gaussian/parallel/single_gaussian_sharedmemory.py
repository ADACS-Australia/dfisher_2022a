#!/usr/bin/env python
# coding: utf-8

from mpdaf.obj import Cube
import numpy as np
import pandas as pd
from numpy import ma
import numpy.testing as nt
from types import MethodType
from astropy.io import fits
import time

import dfisher_2022a as dfi
from dfisher_2022a.emission_lines import Halpha
from dfisher_2022a import ProcessedCube, SNRMap, CubeRegion
from dfisher_2022a import Line
from dfisher_2022a.fits import ResultLM, CubeFitterLM
from dfisher_2022a import fit_lm

time_start = time.time()

# 0. initialize output instance
out = ResultLM()

# 1. read in test data cube
path = "./../data"
file = path + "/" + "single_gaussian_muse_size.fits"
cube = Cube(file)

# 2. settings
Z = 0.009482649107040553
SNR_THRESHOLD = 5

# 3. prepare data for fitting
p = ProcessedCube(cube, z=Z)
p.de_redshift(z=Z)
p.select_region("Halpha", left=20, right=20)
p.get_snrmap(SNR_THRESHOLD)

out.get_output(p)
# out.snr = p.snr[100:120, 100:120]
# out.snrmap = p.snrmap[100:120, 100:120]
#
# data = p.data[:,100:120,100:120]
data = p.data
# weight = p.weight[:,100:120,100:120]
weight = p.weight
model = dfi.models.Lm_Const_1GaussModel
x = p.x

# 4. fit the data
if __name__ == "__main__":
    cfl = CubeFitterLM(data=data, weight=weight, 
                        model=model, x=x, 
                        fit_method='leastsq')

    cfl.fit_cube()
    out.get_output(cfl)
    out.save()
    time_m = time.time()

    # write out result

    time_e = time.time()
    print("--- %s seconds ---" % (time_m - time_start))
    print("--- %s seconds ---" % (time_e - time_m))
        
        









