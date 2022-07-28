# __version__ = '0.1.0'

__all__ = ["fits", "models", "emission_lines", "cube", "line"]

# generate line dictionary at runtime
def _generate_line_dict(emission_lines):
    line_dict = {}
    for attr in dir(emission_lines):
        if not attr.startswith("_"):
            line_dict[attr] = getattr(emission_lines,attr)
    return line_dict
    
from . import emission_lines

EmissionLines = _generate_line_dict(emission_lines)
from . import fits, models
from .io import cube, line
