# __version__ = '0.1.0'

import logging

# set up logger for the package
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(message)s')
file_handler = logging.FileHandler('log.log', 'w')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# generate line dictionary at runtime
def _generate_line_dict(emission_lines):
    line_dict = {}
    for attr in dir(emission_lines):
        if not attr.startswith("_"):
            line_dict[attr] = getattr(emission_lines,attr)
    logger.info("EmissionLines Dictionary has been created.")
    return line_dict
    
from . import emission_lines

EmissionLines = _generate_line_dict(emission_lines)

from .app import fit_lm
from .cube import *
from .fits import *
from .line import *
from .models import *
