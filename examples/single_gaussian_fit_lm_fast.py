#!/usr/bin/env python
# coding: utf-8
import time

from dfisher_2022a import Lm_Const_1GaussModel, fit_lm

time_start = time.time()



# 1. files
path = "./data"
file0 = path + "/" + "single_gaussian_muse_size.fits"

# 2. settings
Z = 0.009482649107040553
SNR_THRESHOLD = 5
LINE = "Halpha"
LEFT = 20
RIGHT = 20

# 3. model
model = Lm_Const_1GaussModel

# 4. fit data


# 4. fit the data
if __name__ == "__main__":
    fit_lm(cubefile=file0, line=LINE, model=model, 
        z=Z, left=LEFT, right=RIGHT, snr_threshold=SNR_THRESHOLD, nprocess=4, fit_method="fast_leastsq", fast=True)

    time_e = time.time()
    print("--- %s seconds ---" % (time_e - time_start))
 
        









