import os
import random

import numpy as np
import numpy.ma as ma
import pytest
from mpdaf.obj import Cube
from numpy.testing import *

from dfisher_2022a import Line, ProcessedCube, ReadCubeFile, RestCube, SNRMap
from dfisher_2022a.exceptions import EmptyRegionError

# data file for testing single gaussian model
PATH = os.path.dirname(__file__)
DATA_FILE_1 = PATH + "/fixtures/single_gaussian_muse_size.fits"

# setting constants
Z = 0.009482649107040553
SNR_THRESHOLD = 5
LINE = "Halpha"

@pytest.fixture
def mpdaf_cube():
    """read in cube for testing"""
    cube = Cube(DATA_FILE_1)
    return cube

def test_readcubefile_data(mpdaf_cube) -> None:
    rcf = ReadCubeFile(DATA_FILE_1)
    cube = rcf.cube
    assert_array_equal(cube.data, mpdaf_cube.data)

def test_readcubefile_masked_data(mpdaf_cube) -> None:
    rcf = ReadCubeFile(DATA_FILE_1)
    cube = rcf.cube
    ma.allequal(cube.data, mpdaf_cube.data)

def test_readcubefile_var(mpdaf_cube) -> None:
    rcf = ReadCubeFile(DATA_FILE_1)
    assert_array_equal(rcf.cube.var, mpdaf_cube.var)

def test_readcubefile_masked_var(mpdaf_cube) -> None:
    rcf = ReadCubeFile(DATA_FILE_1)
    ma.allequal(rcf.cube.var, mpdaf_cube.var)


def test_readcubefile_2files(mpdaf_cube) -> None:
    rcf = ReadCubeFile(DATA_FILE_1, DATA_FILE_1)
    assert_array_equal(rcf.cube.data, mpdaf_cube.data)

# tests for ProcessedCube

def test_attr_z(mpdaf_cube) -> None:
    p = ProcessedCube(mpdaf_cube, Z, SNR_THRESHOLD)
    p.de_redshift(Z)
    assert p.z == Z

def test_de_redshift_default(mpdaf_cube) -> None:
    p = ProcessedCube(mpdaf_cube, Z, SNR_THRESHOLD)
    p.de_redshift()
    rest_wav_p = p.cube.wave.get_crval()
    rest_wav_c = mpdaf_cube.wave.get_crval()/(1+Z)
    assert rest_wav_p == rest_wav_c

def test_de_redshift(mpdaf_cube) -> None:
    """test updating z after using a new value in the called function"""
    p = ProcessedCube(mpdaf_cube, Z, SNR_THRESHOLD)
    random_z = random.random()
    p.de_redshift(random_z)
    rest_wav_p = p.cube.wave.get_crval()
    rest_wav_c = mpdaf_cube.wave.get_crval()/(1+random_z)
    assert rest_wav_p == rest_wav_c

def test_de_redshift_invariance(mpdaf_cube) -> None:
    """test whether rest wavelength is the same after calling function multiple times"""
    p = ProcessedCube(mpdaf_cube, Z, SNR_THRESHOLD)
    n = random.randint(1,5)
    for i in range(n):
        p.de_redshift(Z)
    rest_wav_p = p.cube.wave.get_crval()
    rest_wav_c = mpdaf_cube.wave.get_crval()/(1+Z)
    assert rest_wav_p == rest_wav_c

def test__cube_invariance_1(mpdaf_cube) -> None:
    """test whether cube.copy() works as expected"""
    p = ProcessedCube(mpdaf_cube, Z, SNR_THRESHOLD)
    n = random.randint(1,5)
    for i in range(n):
        p.de_redshift(Z)
    ma.allequal(p._cube.data, mpdaf_cube.data)

def test_select_region(mpdaf_cube) -> None:
    """test whehter the cube shape is changed accordlingly after selecting region"""
    p = ProcessedCube(mpdaf_cube, Z, SNR_THRESHOLD)
    p.de_redshift()
    p.select_region(LINE, left=0, right=0)
    assert p.cube.shape[0] == 1

def test_select_region_error(mpdaf_cube) -> None:
    """test whehter the cube shape is changed accordlingly after selecting region"""
    p = ProcessedCube(mpdaf_cube, Z, SNR_THRESHOLD)
    p.de_redshift()
    with pytest.raises(EmptyRegionError):
        p.select_region(LINE, left=0, right=-1)

def test__cube_invariance_2(mpdaf_cube) -> None:
    """test whether cube.copy() works as expected"""
    p = ProcessedCube(mpdaf_cube, Z, SNR_THRESHOLD)
    p.de_redshift()
    p.select_region(LINE)
    ma.allequal(p._cube.data, mpdaf_cube.data)

def test_snr_threshold_default(mpdaf_cube) -> None:
    """test whether filtering by snr threshold works"""
    p = ProcessedCube(mpdaf_cube, Z, SNR_THRESHOLD)
    p.get_snrmap()
    assert p.snr.min() >= SNR_THRESHOLD

def test_snr_threshold_default_2(mpdaf_cube) -> None:
    """test whether filtering by snr threshold works"""
    p = ProcessedCube(mpdaf_cube, Z, SNR_THRESHOLD)
    p.get_snrmap()
    assert_array_equal(p.snr.mask, p.cube.sum(axis=0).mask)

def test_snr_threshold_zero(mpdaf_cube) -> None:
    p = ProcessedCube(mpdaf_cube, Z)
    p.get_snrmap(snr_threshold=0)
    ma.allequal(p.cube.data, mpdaf_cube.data)

def test_snr_threshold_none(mpdaf_cube) -> None:
    p = ProcessedCube(mpdaf_cube, Z)
    p.get_snrmap()
    ma.allequal(p.cube.data, mpdaf_cube.data)

def test_snr_threshold_all(mpdaf_cube) -> None:
    p = ProcessedCube(mpdaf_cube, Z, SNR_THRESHOLD)
    p.get_snrmap(np.inf)
    assert p.cube.data.mask.all() == True

def test__cube_invariance_3(mpdaf_cube) -> None:
    """test whether cube.copy() works as expected"""
    p = ProcessedCube(mpdaf_cube, Z, SNR_THRESHOLD)
    p.get_snrmap()
    ma.allequal(p._cube.data, mpdaf_cube.data)

def test_weight(mpdaf_cube) -> None:
    p = ProcessedCube(mpdaf_cube, Z, SNR_THRESHOLD)
    ma.allequal(p.data.mask, p.weight.mask)