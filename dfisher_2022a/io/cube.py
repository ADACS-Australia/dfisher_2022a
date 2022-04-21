# process cube
import numpy as np
import numpy.ma as ma

class RestCube():
    def __init__(self, cube, z):
        self.z = z
        self.restcube = None
        self._restwave = None
        self._de_redshift(cube)

    def _de_redshift(self, cube):
        obs_wav = cube.wave.get_crval()
        res_wav = obs_wav / (1 + self.z)
        self._restwave = res_wav
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
            subcube = SNRMap(subcube, self.snr_threshold).snr_masked_cube

        self.cube = subcube

           
            
   
        
   
    # TODO: raise warning when the proposed fitting region is shrinked (exceeds the edge of the data range)
    # TODO: raise error if the fitting region is not included in the data range

class SNRMap():
    def __init__(self, cube, threshold=None) -> None:
        self.threshold = threshold
        self.snr = None
        self.map = None
        self._get_snrmap(cube)
        if self.threshold:
            self._mask_snr(cube)

    def _get_snrmap(self, cube):
        cube_sum = cube.sum(axis=0)
        snr = cube_sum.data/np.sqrt(cube_sum.var)
        self.snr = snr
        cube_sum.data = snr
       
        if self.threshold:
            cube_sum = (cube_sum > self.threshold)
           
        self.map = cube_sum
        
        #TODO: raise error when cube_sum.var is zero
    
    def _mask_snr(self, cube):
        for i in range(cube.shape[0]):
            cube.mask[i,:,:] = cube.mask[i,:,:] + self.snr.mask
        self.snr_masked_cube = cube
