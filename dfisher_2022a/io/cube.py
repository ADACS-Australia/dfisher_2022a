__all__ = ["RestCube", "FitReadyCube", "ProcessedCube", "SNRMap"]

from turtle import left
import numpy as np
import numpy.ma as ma

from mpdaf.obj import Cube
from .line import Line
from ..exceptions import EmptyRegionError

class ProcessedCube():
    
    def __init__(self, cube:Cube, z=0.0, snr_threshold=None):
        self.cube = cube.copy()
        self.z = z
        self.snr_threshold = None
        self._de_redshift(cube)
        print("reload 5")
    
    @property
    def data(self):
        return self.cube.data

    @property
    def weight(self):      
        if self.cube.var is not None:
            weight = 1 / np.sqrt(np.abs(self.cube.var))
        else:
            weight = None
        return weight

    @property
    def x(self):
        return self.cube.wave.coord()

    def _de_redshift(self, cube):
        obs_wav = cube.wave.get_crval()
        res_wav = obs_wav / (1 + self.z)
        self._restwave = res_wav
        # update cube attribute
        self.cube.wave.set_crval(res_wav)

    # def _init_process(self, cube):
    #     """ initial datacube processing at creation"""
    #     self._de_redshift(cube)
    #     if self.snr_threshold:
    #         self.filter_snr(self.snr_threshold)


    # TODO: CHECK HOW THIS IS DONE IN THREADCOUNT
    def refine_redshift(self):
        """allow user to refine redshift"""
        pass

class CubeRegion():
    def __init__(self, cube: Cube, line: Line, left=15, right=15, snr_threshold=None):
        self.cube = cube.copy()
        self.line = line
        self.left = left
        self.right = right  
        self.snr_threshold = snr_threshold

    @property
    def low(self):
        return self.line.wavelength - self.left
    
    @property
    def high(self):
        return self.line.wavelength + self.right

    def _get_wave_index(self, wavelength: float, nearest=True):
        wave_index = self.cube.wave.pixel(wavelength, nearest=nearest)
        return wave_index

    # TODO: RAISE ERROR IF NO REGION IS SELECTED
    def _get_fit_region(self):
        low_idx = self._get_wave_index(self.low)
        high_idx = self._get_wave_index(self.high)
        
        if low_idx == high_idx:
            raise EmptyRegionError()

        return self.cube[low_idx:high_idx+1,:,:]

    def _filter_snr(self, cube, threshold):
        snrmap = SNRMap(cube, threshold=threshold) #TODO: SEPARATE THE CREATION OF SNRMAP OBJECT AND USE IT
        snr_mask = snrmap.snr.mask
        cube.mask = cube.mask + snr_mask
        return cube

    def get_regional_cube(self):
        fcube = self._get_fit_region()
        if self.snr_threshold:
            fcube = self._filter_snr(fcube, self.snr_threshold)
        return fcube
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
        self.snr = ma.masked_where(self.snr < threshold, self.snr)
        self.map.data = self.snr

    def _count_masked_pixels(self,snr):
        """count the number of masked pixels"""
        return np.sum(snr.mask)


