""" Custom lmfit composite models.
The models are provided by the science team of DFisher_2022A project.
The original code can be found at https://github.com/astrodee/threadcount/blob/master/src/threadcount/models.py
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
        self.com_func = gaussianCH      # model function of the composite model

    def _set_paramhints_prefix(self):
        # GaussianModelH paramhints already sets sigma min=0 and height min=0
        pass

    def _reprstring(self, long=False):
        return "constant + 1 gaussian"

    guess = _guess_1gauss

    #NOTE: the following function is used instead of `Model.eval` when `fast_leastsq` method is called
    def eval_fast(self, nvars, **kwargs):
        return self.com_func(kwargs['x'], *nvars)


    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC