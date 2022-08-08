import os

import numpy as np
import numpy.ma as ma
import pytest
from numpy.testing import *

from dfisher_2022a import CubeFitterLM, Lm_Const_1GaussModel
from dfisher_2022a.exceptions import InputDimError, InputShapeError
from sympy import xfield

PATH = os.path.dirname(__file__) +"/fixtures/"
DFILE = PATH + "data_data.npy"
DMFILE = PATH + "data_mask.npy"
WFILE = PATH + "weight_data.npy"
WMFILE = PATH + "weight_mask.npy"
XFILE = PATH + "x.npy"

datad = np.load(DFILE)
datam = np.load(DMFILE)
weightd = np.load(WFILE)
weightm = np.load(WMFILE)
x = np.load(XFILE)

# setting
MODEL = Lm_Const_1GaussModel

def pack_mask_arr(data, mask):
    return ma.MaskedArray(data=data, mask=mask)

@pytest.fixture
def input_data():
    input_data = {}
    input_data["data"] = pack_mask_arr(datad, datam)
    input_data["weight"] = pack_mask_arr(weightd, weightm)
    input_data["x"] = x
    return input_data

# def test_cube_fit(input_data) -> None:
    

