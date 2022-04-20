# process cube
import numpy as np
import numpy.ma as ma

class RestCube():
    def __init__(self, cube, z):
        self.z = z
        self.restcube = None
        self._de_redshift()

    def _de_redshift(self):
        obs_wav = cube.wave.get_crval()
        res_wav = obs_wav / (1 + z)
        cube.wave.set_crval(res_wav)
        self.restcube = cube

    def refine_redshift(self):
        pass


class FitReadyCube():
    def __init__(self, restcube, line, snr_threshold=None):
        self.snr_threshold = snr_threshold
        self.line_index = self._get_wave_index(line.line)
        self.low_index = self._get_wave_index(line.low)
        self.high_index = self._get_wave_index(line.high)
        self.cube = None
        self._get_fitreadycube()

    def _get_wave_index(self, wave, nearest=True):
        wave_index = restcube.restcube.wave.pixel(wave, nearest=nearest)
        return wave_index

    def _get_fitreadycube(self):
        subcube = restcube[self.low_index:self.high_index+1,:,:]
        if snr_threshold:
            subcube = self._mask_snr(cube=subcube)

        self.cube = subcube

           
            
    def _mask_snr(self, cube):
        snr = SNRMap(subcube, threshold=self.snr_threshold).snr
        restcube.data[:,slice(snr)] = ma.masked
        
        return restcube 
        
   
    # TODO: raise warning when the proposed fitting region is shrinked (exceeds the edge of the data range)
    # TODO: raise error if the fitting region is not included in the data range

class SNRMap():
    def __init__(self, cube, threshold=None) -> None:
        self.snr = None
        self.snr_mask = None
        self.map = None
        self._get_snrmap()

    def _get_snrmap(self):
        cube_sum = cub_sum.sum(axis=0)
        snr = cube_cum.data/np.sqrt(cube_sum.var)
        self.snr = snr
        cube_sum.data = snr
        if threshold:
            snr_mask = (snr < threshold)
            self.snr_mask = snr_mask
            new_mask = cube_sum.mask + snr_mask
            cube_sum.mask = new_mask
            cube_sum.data = ma.array(cube_sum._data, mask=new_mask)

        self.map = cube_sum
        