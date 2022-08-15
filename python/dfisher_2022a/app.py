import logging
import os

from lmfit import Model
from mpdaf.obj import Cube

from .cube import CubeRegion, ProcessedCube, ReadCubeFile, RestCube, SNRMap
from .fits.cpufitter import CubeFitterLM, ResultLM
from .line import Line

logger = logging.getLogger(__name__)

# CPU COUNT
CPU_COUNT = os.cpu_count()

# NOTE: once multiple fitters (cpu, gpu) are implemented, a fit interface could be added
# class FitInterface():
#     def __init__(self, cubefile, varfile=None):
#         self.cubefile = cubefile
#         self.varfile = varfile
#         self.rawcube = ReadCubeFile(cubefile, varfile).cube
#         self.out = ResultLM()

#     def prepare_data(self, line: str, z=0., left=15, right=15, snr_threshold=None):
#         p = ProcessedCube(self.cube, z=z, snr_threshold=snr_threshold)
#         p.de_redshift(z=z)
#         p.select_region(line=line, left=15, right=15)
#         p.get_snrmap(snr_threshold=snr_threshold)
#         self.p = p
#         self.out.get_output(p)

def prepare_data(cube: Cube, line: str, z=0., left=15, right=15, snr_threshold=None):
    """
    Get data ready for fitting.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        
    line : str
        name of emission line, e.g. "Hapha"
    z : float, optional
        redshift, by default 0.
    left : int, optional
        relative wavelength cut at the left side of the line, by default 15
    right : int, optional
        relative wavelength cut at the right side of the line, by default 15
    snr_threshold : None or float, optional
        snr threshold, by default None

    Returns
    -------
    dfisher_2022a.ProcessedCube
    """
    p = ProcessedCube(cube=cube, z=z, snr_threshold=snr_threshold)
    p.de_redshift(z=z)
    p.select_region(line=line, left=left, right=right)
    p.get_snrmap(snr_threshold=snr_threshold)
    return p

def fit_lm(cubefile, line: str, model: Model, varfile=None, z=0.,
            left=15, right=15, snr_threshold=None, nprocess=CPU_COUNT, fit_method="leastsq", mode="parallel", **kwargs):
    """
    A wrapper function for cube fitting backed with lmfit.

    Parameters
    ----------
    cubefile : fits file 
        data file
    line : str
        name of emission line, e.g. "Hapha"
    model : lmfit.Model or lmfit.CompositeModel    
    varfile : fits file, optional
        variance file, by default None. 
    z : float, optional
        redshift, by default 0.
    left : int, optional
        relative wavelength cut at the left side of the line, by default 15, by default 15
    right : int, optional
        relative wavelength cut at the right side of the line, by default 15, by default 15
    snr_threshold : None or float, optional
        snr threshold, by default None
    nprocess : int, optional
        number of worker processes used in parallel fitting, by default "os.cpu_count()"
    fit_method : str, optional
        fitting method available in lmfit, by default "leastsq"
    mode : "parallel" or "serial", optional
        fitting mode, by default "parallel"
    kwargs: dict, optional
        other keyword arguments that could be passed to dfisher_2022a.CubeFitterLM
    Returns
    -------
    dfisher_2022a.ResultLM
        result container
    """
    out = ResultLM()
    setattr(out,"CubeFile", cubefile)
    setattr(out, "VarFile", varfile)
    logger.info("Read in cube")
    rawcube = ReadCubeFile(cubefile, varfile).cube
    
    logger.info("Prepare data")
    p = prepare_data(rawcube, line, z, left, right, snr_threshold)
    out.get_output(p)

    logger.info("Start fitting data")
    fr = CubeFitterLM(p.data, p.weight, p.x, model, nprocess=nprocess, method=fit_method, **kwargs)
    if mode == "parallel":
        fr.fit_cube()
    if mode == "serial":
        fr.fit_serial()
    
    out.get_output(fr)
    out.save()
    return out






        