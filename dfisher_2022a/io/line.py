from dataclasses import dataclass

from dfisher_2022a import EmissionLines

__all__ = ["Line"]
@dataclass(frozen=True)
class Line():
    '''
    object to store line information
    '''
    name: str
    @property
    def wavelength(self):
        return EmissionLines[self.name]

    def __repr__(self):
        return f"{self.name}: {self.wavelength}"
    
