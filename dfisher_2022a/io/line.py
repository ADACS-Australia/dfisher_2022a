# emission line related

class Line():
    '''
    determine the fitting region of a given line
    '''
    def __init__(self, line, right=15, left=15) -> None:
        self.line = line
        self.right = right
        self.left = left
       
        # get the fitting range in wavelength
        self.low = line - left
        self.high = line + right

       
    


