#!/usr/bin/env python
# coding: utf-8

from mpdaf.obj import Cube
import numpy as np
import os
import time

import dfisher_2022a as dfi
from dfisher_2022a.emission_lines import Halpha
from dfisher_2022a.io.cube import FitReadyCube, RestCube, SNRMap
from dfisher_2022a.io.line import Line

time_start = time.time()

# 1. read in test data cube
path = "./../../../dfisher_2022a/tests/data"
file = path + "/" + "single_gaussian_muse_size.fits"
cube = Cube(file)

# 2. settings
Z = 0.009482649107040553
SNR_THRESHOLD = 5

# 3. prepare data for fitting
# 3.1 de-redshift
rcube = RestCube(cube, Z)

# 3.2 select fitting region
fline = Line(Halpha)
frcube = FitReadyCube(rcube, fline)

# 3.3 get snr map
snrmap = SNRMap(rcube.restcube, SNR_THRESHOLD)
snr = snrmap.snr
smap = snrmap.map

# 3.4 get fit ready cube
ffrcube = FitReadyCube(rcube, fline, SNR_THRESHOLD)

# 4. fit the data
model = dfi.models.Lm_Const_1GaussModel
small = cube[:, 100:120,100:120]
# small = cube[:, 100:101,100]
# small = cube
fc = dfi.fits.base.FitCube(small, model)

# fit spaxel serially 
fc.fit_all_serial()

time_m = time.time()

print("--- %s seconds ---" % (time_m - time_start))
















