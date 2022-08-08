""" composite models
"""
import operator

import lmfit

from ..base import gaussianCH
from .base import ConstantModelH, GaussianModelH, _guess_1gauss

__all__ = ["Const_1GaussModel"]
class Const_1GaussModel(lmfit.model.CompositeModel):
    """Constant + 1 Gaussian Model.

    Essentially created by:

    ``lmfit.models.ConstantModel() + GaussianModelH(prefix="g1_")``

    The param names are ['g1_height',
    'g1_center',
    'g1_sigma',
    'c']
    """
    
    def __init__(
        self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs  # noqa
    ):
        kwargs.update({"nan_policy": nan_policy, "independent_vars": independent_vars})
        if prefix != "":
            print(
                "{}: I don't know how to get prefixes working on composite models yet. "
                "Prefix is ignored.".format(self.__class__.__name__)
            )

        g1 = GaussianModelH(prefix="g1_", **kwargs)
        c = ConstantModelH(prefix="", **kwargs)

        # the below lines gives g1 + c
        super().__init__(g1, c, operator.add)
        self._set_paramhints_prefix()
        self.com_func = gaussianCH

    def _set_paramhints_prefix(self):
        # GaussianModelH paramhints already sets sigma min=0 and height min=0
        pass

    def _reprstring(self, long=False):
        return "constant + 1 gaussian"

    guess = _guess_1gauss

    #NOTE: the following function replace the default eval function defined in lmfit to speed up 
    def eval(self, nvars, **kwargs):
        # print("new eval: ", nvars)
        return self.com_func(kwargs['x'], *nvars)

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC