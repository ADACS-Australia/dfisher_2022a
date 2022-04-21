'''
Tests for io.cube.
'''
import os
import pytest
import numpy as np
from mpdaf.obj import Cube

from dfisher_2022a.io.cube import FitReadyCube, RestCube, SNRMap
from dfisher_2022a.io.line import Line
from .setting_constants import Z, SNR_THRESHOLD, OIII5007


@pytest.fixture
def mpdaf_cube():
    datafile = os.getcwd() + "/data/testing.fits"
    return Cube(datafile)


class TestRestCube():
    def test_input_shape(self, mpdaf_cube):
        assert mpdaf_cube.shape == (43,23,30)

    def test_de_redshift(self, mpdaf_cube):
        restcube = RestCube(mpdaf_cube, Z)
        restcube_wav = restcube.restcube.wave.get_crval()
        rest_wav = restcube._restwave
        assert rest_wav == restcube_wav

@pytest.fixture
def rest_cube(mpdaf_cube):
    return RestCube(mpdaf_cube, Z)

@pytest.fixture
def emission_line():
    return Line(OIII5007)

class TestFitReadyCube():
    
    def test_init(self, rest_cube, emission_line):
        frcube = FitReadyCube(rest_cube, emission_line, SNR_THRESHOLD)
        assert frcube.snr_threshold == SNR_THRESHOLD

    def test_get_wave_index(self, rest_cube, emission_line):
        # the index of CRVAL should be 0
        frcube = FitReadyCube(rest_cube, emission_line, SNR_THRESHOLD)
        wav = rest_cube.restcube.wave.get_crval()
        pixel = frcube._get_wave_index(rest_cube, wav)
        assert pixel == 0
    
    def test_line_index(self, rest_cube, emission_line):
        frcube = FitReadyCube(rest_cube, emission_line, SNR_THRESHOLD)
        assert frcube.low_index < frcube.high_index

    def test_get_fitreadycube(self, rest_cube, emission_line):
        frcube = FitReadyCube(rest_cube, emission_line)
        frcube._get_fitreadycube(rest_cube)
        assert frcube.cube.shape == (frcube.high_index + 1 - frcube.low_index,23,30)

  


class TestSNRMap():
    def test_snr_shape(self, mpdaf_cube):
        snrmap = SNRMap(mpdaf_cube)
        assert snrmap.snr.shape == (23,30)

    def test_snr_type(self, mpdaf_cube):
        snrmap = SNRMap(mpdaf_cube)
        assert type(snrmap.snr) is np.ma.core.MaskedArray

    def test_get_map(self, mpdaf_cube):
        snrmap = SNRMap(mpdaf_cube, SNR_THRESHOLD)
        assert np.max(snrmap.map.data) > SNR_THRESHOLD

    def test_mask_snr(self, mpdaf_cube):
        snrmap = SNRMap(mpdaf_cube, SNR_THRESHOLD)
        combined_mask = snrmap.snr.mask + mpdaf_cube.mask[0]
        assert np.array_equal(snrmap.snr_masked_cube.mask[0], combined_mask)


    




