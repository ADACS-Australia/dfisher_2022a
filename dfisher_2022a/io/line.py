from dataclasses import dataclass

from dfisher_2022a import EmissionLines


@dataclass(frozen=True)
class Line():
    '''
    object to store line information
    '''
    name: str
    @property
    def wavelength(self):
        return EmissionLines[self.name]
 
class LineRegion():
    def __init__(self, line:Line, right=15, left=15):
        self.line = line
        self.right = right
        self.left = left

    @property
    def low(self):
        return self.line.wavelength - self.left

    @property
    def high(self):
        return self.line.wavelength + self.right


   # def __init__(self, name):
    #     self.name = name
    #     self.wavelength = EmissionLines[name]
    # def __init__(self, line, right=15, left=15) -> None:
    #     self.line = line
    #     self.right = right
    #     self.left = left
       
    #     # get the fitting range in wavelength
    #     self.low = line - left
    #     self.high = line + right