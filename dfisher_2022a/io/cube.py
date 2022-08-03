__all__ = ["RestCube", "FitReadyCube", "ProcessedCube", "SNRMap", "CubeRegion"]

from turtle import left
import numpy as np
import numpy.ma as ma
from types import MethodType

from mpdaf.obj import Cube
from .line import Line
from ..exceptions import EmptyRegionError

def _get_custom_attr(top, cls):
    attr_names = dir(cls)
    for name in attr_names:
        attr = getattr(cls, name)
        if not (name.startswith("_") or 
        type(attr) is MethodType):
            print(f"{name}: {attr}")
            setattr(top, name, attr)


class ProcessedCube():
    
    def __init__(self, cube: Cube, z=0.0, snr_threshold=None):
        self._cube = cube
        self.cube = cube.copy()
        self.z = z
        self.snr_threshold = snr_threshold
    
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

    def de_redshift(self, z=None):
        if z is None:
            z = self.z
        rc = RestCube(self._cube, z=z) 
        print("type of self: ", type(self))
        _get_custom_attr(self, rc)
        return rc

    def select_region(self, line: str, left=15, right=15):
        cr = CubeRegion(self.cube, line, left=left, right=right)
        cr.get_regional_cube()
        _get_custom_attr(self, cr)

    def get_snrmap(self, snr_threshold=None):
        if snr_threshold is None:
            snr_threshold = self.snr_threshold
        sm = SNRMap(self.cube, snr_threshold=snr_threshold)
        sm.filter_cube()
        _get_custom_attr(self, sm)




    # TODO: CHECK HOW THIS IS DONE IN THREADCOUNT
    # def refine_redshift(self):
    #     """allow user to refine redshift"""
    #     pass

class RestCube():
    """class responds for de-reshifting data cube"""
    def __init__(self, cube: Cube, z=0.):
        self.cube = cube.copy()
        self.z = z
        self._de_redshift(cube)

    def _de_redshift(self, cube):
        obs_wav = cube.wave.get_crval()
        res_wav = obs_wav / (1 + self.z)
        self._restwave = res_wav
        # update cube attribute
        self.cube.wave.set_crval(res_wav)
class CubeRegion():
    def __init__(self, cube: Cube, line: str, left=15, right=15):
        self._cube = cube
        self.cube = cube.copy()
        self.line = Line(line)
        self.left = left
        self.right = right  

    @property
    def low(self):
        return self.line.wavelength - self.left
    
    @property
    def high(self):
        return self.line.wavelength + self.right

    def _get_wave_index(self, wavelength: float, nearest=True):
        wave_index = self.cube.wave.pixel(wavelength, nearest=nearest)
        return wave_index

    def _get_fit_region(self):
        low_idx = self._get_wave_index(self.low)
        high_idx = self._get_wave_index(self.high)
        
        if low_idx == high_idx:
            raise EmptyRegionError()
        self.cube = self._cube[low_idx:high_idx+1,:,:]
        return self.cube
    
    # def _create_snr_map(self, cube, threshold):
    #     snrmap = SNRMap(cube, threshold=threshold)
    #     self.snrmap = snrmap
    #     return snrmap


    def get_regional_cube(self):
        fcube = self._get_fit_region()    
        return fcube

class SNRMap():
    def __init__(self, cube: Cube, snr_threshold=None):
        self._cube = cube
        self.cube = cube.copy()
        self.snr_threshold = snr_threshold
        self.snr = None
        self.snrmap = None
        self._get_snr_map(cube) 
        self._count_masked_pixels(self._snr)
        self._get_snr_stats(self._snr)
        if snr_threshold:
            self._filter_snr(snr_threshold)
            self._count_masked_pixels(self.snr)
            self._get_snr_stats(self.snr)
       

    def _get_snr_map(self, cube):
        cube_sum = cube.sum(axis=0)
        snr = cube_sum.data/np.sqrt(cube_sum.var)
        self._snr = snr
        self.snr = snr.copy()

        cube_sum.data = self.snr
        self._snrmap = cube_sum
        self.snrmap = cube_sum.copy()
        
        #TODO: raise error when cube_sum.var is zero
    
    # TODO: SET THE FILLED_VALUE, MAKE IT EASY TO MASK CUBE
    def _filter_snr(self, threshold):
        self.snr = ma.masked_where(self._snr < threshold, self._snr)
        self.snrmap.data = self.snr

    def _count_masked_pixels(self, snr):
        """count the number of masked pixels and the percentage of the total pixel"""
        n_masked = np.sum(snr.mask)
        masked_percentage = n_masked/snr.size
        print(n_masked, masked_percentage)

    def _get_snr_stats(self, snr):
        """get statistics of snr"""
        median = ma.median(snr)
        mean = ma.mean(snr)
        avg = ma.average(snr)
        min_snr, max_snr = ma.min(snr), ma.max(snr)
        print(median, mean, avg, min_snr, max_snr)

    def filter_cube(self):
        """filter cube by snr threshold"""
        snr_mask = self.snr.mask
        self.cube.mask = self.cube.mask + snr_mask
        return self.cube
class PreFitCube():

    """fine-tune fit settings before fitting the cube"""
    
    def __init__(self, cube: Cube, line: Line, model, fit_method='leastsq'):
        self.cube = cube
        self.line = line
        self.model = model

    def fit_spaxel(self, cube: Cube, pix:tuple or list):
        m = self.model()
        sp = cube[:,pix[0], pix[1]]


# class RestCube():
#     def __init__(self, cube, z=0.0):
#         self.z = z
#         self.restcube = None
#         self._restwave = None
#         self._de_redshift(cube)

#     def _de_redshift(self, cube):
#         obs_wav = cube.wave.get_crval()
#         res_wav = obs_wav / (1 + self.z)
#         # print("some waves: ", obs_wav, res_wav)
#         self._restwave = res_wav
#         # rcube = cube.clone()
#         cube.wave.set_crval(res_wav)
#         self.restcube = cube

#     def refine_redshift(self):
#         pass


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





