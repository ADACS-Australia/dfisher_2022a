import logging

from lmfit import Model
from mpdaf.obj import Cube

from .cube import CubeRegion, ProcessedCube, ReadCubeFile, RestCube, SNRMap
from .fits.cpufitter import CubeFitterLM, ResultLM
from .line import Line

logger = logging.getLogger(__name__)
class FitInterface():
    def __init__(self, cubefile, varfile=None):
        self.cubefile = cubefile
        self.varfile = varfile
        self.rawcube = ReadCubeFile(cubefile, varfile).cube
        self.out = ResultLM()

    def prepare_data(self, line: str, z=0., left=15, right=15, snr_threshold=None):
        p = ProcessedCube(self.cube, z=z, snr_threshold=snr_threshold)
        p.de_redshift(z=z)
        p.select_region(line=line, left=15, right=15)
        p.get_snrmap(snr_threshold=snr_threshold)
        self.p = p
        self.out.get_output(p)

def prepare_data(cube: Cube, line: str, z=0., left=15, right=15, snr_threshold=None):
    """"Get data ready for fitting."""
    p = ProcessedCube(cube=cube, z=z, snr_threshold=snr_threshold)
    p.de_redshift(z=z)
    p.select_region(line=line, left=left, right=right)
    p.get_snrmap(snr_threshold=snr_threshold)
    return p

def fit_lm(cubefile, line: str, model: Model, varfile=None, z=0.,
            left=15, right=15, snr_threshold=None, fit_method="leastsq", mode="parallel", **kwargs):
    out = ResultLM()
    setattr(out,"CubeFile", cubefile)
    setattr(out, "VarFile", varfile)
    logger.info("Read in cube")
    rawcube = ReadCubeFile(cubefile, varfile).cube
    
    logger.info("Prepare data")
    p = prepare_data(rawcube, line, z, left, right, snr_threshold)
    out.get_output(p)

    logger.info("Start fitting data")
    fr = CubeFitterLM(p.data, p.weight, p.x, model, method=fit_method, **kwargs)
    if mode == "parallel":
        fr.fit_cube()
    if mode == "serial":
        fr.fit_serial()
    
    out.get_output(fr)
    out.save()
    return out






        