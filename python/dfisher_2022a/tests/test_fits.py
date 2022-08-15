import os
import random

from mpdaf.obj import Cube
import numpy as np
import numpy.ma as ma
import pytest
from numpy.testing import *

from dfisher_2022a import CubeFitterLM, Lm_Const_1GaussModel, ProcessedCube
from dfisher_2022a.exceptions import InputDimError, InputShapeError
from multiprocessing import shared_memory, Process
from multiprocessing.managers import SharedMemoryManager

# prepare data for testing
PATH = os.path.dirname(__file__)
DATA_FILE_1 = PATH + "/fixtures/single_gaussian_muse_size.fits"
cube = Cube(DATA_FILE_1)

Z = 0.009482649107040553
SNR_THRESHOLD = 5
LINE = "Halpha"

p = ProcessedCube(cube, z=Z)
p.de_redshift(z=Z)
p.select_region("Halpha", left=20, right=20)
p.get_snrmap(SNR_THRESHOLD)

data = p.data
weight = p.weight
x = p.x


MODEL = Lm_Const_1GaussModel

@pytest.fixture
def input_data():
    input_data = {}
    input_data["data"] = p.data[:,100:110,100:120]
    input_data["weight"] = p.weight[:,100:110,100:120]
    input_data["x"] = p.x
    return input_data

def test_paralle_vs_serial(input_data) -> None:
    """Test whether fit_cube and fit_serial generate the same results"""
    cfl = CubeFitterLM(input_data["data"], input_data["weight"], input_data["x"], MODEL)
    res1 = cfl.fit_cube()
    res2 = cfl.fit_serial()
    assert_allclose(res1, res2)

########
# comment the following function out when publishing to pypi
# def test_fast_leastsq(input_data) -> None:
#     """Test whether fast leastsq and leastsq generate the same results"""
#     cfl1 = CubeFitterLM(input_data["data"], input_data["weight"], input_data["x"], MODEL, method="leastsq")
#     cfl2 = CubeFitterLM(input_data["data"], input_data["weight"], input_data["x"], MODEL, method="fast_leastsq", fast=True)
#     res1 = cfl2.fit_serial()
#     res2 = cfl2.fit_serial()
#     assert_allclose(res1, res2)





