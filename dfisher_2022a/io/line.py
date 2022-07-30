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

    def __repr__(self):
        return f"{self.name}: {self.wavelength}"
    
# class LineRegion():
#     def __init__(self, central:float, right=15, left=15):
#         self.central = central
#         self.right = right
#         self.left = left

#     @property
#     def low(self):
#         return self.central - self.left

#     @property
#     def high(self):
#         return self.central + self.right


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