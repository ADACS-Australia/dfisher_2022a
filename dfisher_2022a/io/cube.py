__all__ = ["RestCube", "FitReadyCube", "ProcessedCube", "SNRMap"]

import numpy as np
import numpy.ma as ma

from mpdaf.obj import Cube
from .line import Line

class ProcessedCube():
    def __init__(self, cube:Cube, z=0.0, snr_threshold=None):
        self.cube = cube
        self.z = z
        self.snr_threshold = None
        self._de_redshift()

    def _de_redshift(self):
        obs_wav = self.cube.wave.get_crval()
        res_wav = obs_wav / (1 + self.z)
        self._restwave = res_wav
        self.cube.wave.set_crval(res_wav)

    # TODO: CHECK HOW THIS IS DONE IN THREADCOUNT
    def refine_redshift(self):
        """allow user to refine redshift"""
        pass
    def _get_wave_index(self, wavelength: float, nearest=True):
        wave_index = self.cube.wave.pixel(wavelength, nearest=nearest)
        return wave_index

    def set_fit_region(self, line: Line):
        low_idx = self._get_wave_index(line.low)
        high_idx = self._get_wave_index(line.high)
        self.cube = self.cube[low_idx:high_idx,:,:]

    def filter_snr(self, threshold):
        snrmap = SNRMap(self.cube, threshold=threshold) #TODO: SEPARATE THE CREATION OF SNRMAP OBJECT AND USE IT
        snr_mask = snrmap.filtered_snr.mask
        self.cube = self.cube * snr_mask
        

class RestCube():
    def __init__(self, cube, z=0.0):
        self.z = z
        self.restcube = None
        self._restwave = None
        self._de_redshift(cube)

    def _de_redshift(self, cube):
        obs_wav = cube.wave.get_crval()
        res_wav = obs_wav / (1 + self.z)
        # print("some waves: ", obs_wav, res_wav)
        self._restwave = res_wav
        # rcube = cube.clone()
        cube.wave.set_crval(res_wav)
        self.restcube = cube

    def refine_redshift(self):
        pass


class FitReadyCube():
    def __init__(self, restcube, line, snr_threshold=None):
        self.snr_threshold = snr_threshold
        self.line_index = self._get_wave_index(restcube, line.line)
        self.low_index = self._get_wave_index(restcube, line.low)
        self.high_index = self._get_wave_index(restcube, line.high)
        self.cube = None
        self._get_fitreadycube(restcube)

    def _get_wave_index(self, restcube, wave, nearest=True):
        wave_index = restcube.restcube.wave.pixel(wave, nearest=nearest)
        return wave_index

    def _get_fitreadycube(self, restcube):
        subcube = restcube.restcube[self.low_index:self.high_index+1,:,:]

        if self.snr_threshold:
            self.snrmap = SNRMap(subcube, self.snr_threshold)
            subcube = self.snrmap.snr_masked_cube

        self.cube = subcube

           
            
   
        
   
    # TODO: raise warning when the proposed fitting region is shrinked (exceeds the edge of the data range)
    # TODO: raise error if the fitting region is not included in the data range

class SNRMap():
    def __init__(self, cube: Cube, threshold=None):
        self.threshold = threshold
        self.snr = None
        self.map = None
        self._get_snr_map(cube) 
        if threshold:
            self._filter_snr(threshold)
       

    def _get_snr_map(self, cube):
        cube_sum = cube.sum(axis=0)
        snr = cube_sum.data/np.sqrt(cube_sum.var)
        self.snr = snr

        cube_sum.data = self.snr
        self.map = cube_sum
        
        #TODO: raise error when cube_sum.var is zero
    
    # TODO: SET THE FILLED_VALUE, MAKE IT EASY TO MASK CUBE
    def _filter_snr(self, threshold):
        filtered_snr = ma.masked_where(self.snr < threshold, self.snr)
        self.filtered_snr = filtered_snr

        filtered_map = self.map
        filtered_map.data = filtered_snr
        self.filtered_map = filtered_map

    def _count_masked_pixels(self,snr):
        """count the number of masked pixels"""
        return np.sum(snr.mask)


