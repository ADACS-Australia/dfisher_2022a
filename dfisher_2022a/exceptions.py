class Error(Exception):
    """Base class of other error classes."""
    def __init__(self, message):
        Exception.__init__(self)
        self.message = message

    def __str__(self):
        return f"{self.message}"

class InputDimError(Error):
    def __init__(self, dim, message="The input data must be 3-d."):
        self.dim = dim
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'The array is {self.dim}-d. {self.message}'

class InputShapeError(Error):
    def __init__(self, message="Weight must be either None or of the same shape as data."):
        super().__init__(message)

