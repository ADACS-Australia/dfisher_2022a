# base models
import lmfit
import numpy as np

from ..base import (constantH, gaussianCH, gaussianH, guess_1gauss,
                    guess_from_peak)


def flux_expr(model):
    """Return constraint expression for line flux."""
    fmt = "{factor:.7f}*{prefix:s}height*{prefix:s}sigma"
    return fmt.format(factor=model.flux_factor, prefix=model.prefix)

def _guess_1gauss(self, data, x, **kwargs):
    """Estimate initial model parameter values from data.

    The data for gaussian g1 will be guessed by 1 gaussian plus constant.

    a and b model parameters are initialized by any model parameter hint and not
    affected by the guess function.

    Parameters
    ----------
    data : array_like
        Array of data (i.e., y-values) to use to guess parameter values.
    x : array_like
        Array of values for the independent variable (i.e., x-values).
    **kws : optional
        Additional keyword arguments, passed to model function.

    Returns
    -------
    params : :class:`~lmfit.parameter.Parameters`
        Initial, guessed values for the parameters of a Model.
    """
    g1_height, g1_center, g1_sigma, constant = guess_1gauss(data, x)
   
    pars = self.make_params(
        g1_height=g1_height, g1_center=g1_center, g1_sigma=g1_sigma, c=constant,
    )

    return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

class ConstantModelH(lmfit.Model):
    """Constant model, with a single Parameter: `c`.
    Note that this is 'constant' in the sense of having no dependence on
    the independent variable `x`, not in the sense of being non-varying.
    To be clear, `c` will be a Parameter that will be varied in the fit
    (by default, of course).
    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        super().__init__(constantH, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = self.make_params()

        pars[f'{self.prefix}c'].set(value=data.mean())
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC
    guess.__doc__ = lmfit.models.COMMON_GUESS_DOC



class GaussianModelH(lmfit.Model):
    r"""A model heavily based on lmfit's :class:`~lmfit.models.GaussianModel`, fitting height instead of amplitude.

    A model based on a Gaussian or normal distribution lineshape.
    The model has three Parameters: `height`, `center`, and `sigma`.
    In addition, parameters `fwhm` and `flux` are included as
    constraints to report full width at half maximum and integrated flux, respectively.

    .. math::

       f(x; A, \mu, \sigma) = A e^{[{-{(x-\mu)^2}/{{2\sigma}^2}}]}

    where the parameter `height` corresponds to :math:`A`, `center` to
    :math:`\mu`, and `sigma` to :math:`\sigma`. The full width at half
    maximum is :math:`2\sigma\sqrt{2\ln{2}}`, approximately
    :math:`2.3548\sigma`.

    For more information, see: https://en.wikipedia.org/wiki/Normal_distribution

    The default model is constrained by default param hints so that height > 0.
    You may adjust this as you would in any lmfit model, either directly adjusting
    the parameters after they have been made ( params['height'].set(min=-np.inf) ),
    or by changing the model param hints ( model.set_param_hint('height',min=-np.inf) ).

    """

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    """float: Factor used to create :func:`lmfit.models.fwhm_expr`."""
    flux_factor = np.sqrt(2 * np.pi)
    """float: Factor used to create :func:`flux_expr`."""

    def __init__(
        self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs  # noqa
    ):
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(gaussianH, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("height", min=0)
        self.set_param_hint("fwhm", expr=lmfit.models.fwhm_expr(self))
        self.set_param_hint("flux", expr=flux_expr(self))

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data, :func:`guess_from_peak`.

        Parameters
        ----------
        data : array_like
            Array of data (i.e., y-values) to use to guess parameter values.
        x : array_like
            Array of values for the independent variable (i.e., x-values).
        negative : bool, default False
            If True, guess height value assuming height < 0.
        **kws : optional
            Additional keyword arguments, passed to model function.

        Returns
        -------
        params : :class:`~lmfit.parameter.Parameters`
            Initial, guessed values for the parameters of a :class:`lmfit.model.Model`.

        """
        height, center, sigma = guess_from_peak(data, x, negative=negative)
        pars = self.make_params(height=height, center=center, sigma=sigma)

        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC

def _tmp():
    pass

class GaussianConstModelH(lmfit.Model):
    """Constant model, with a single Parameter: `c`.
    Note that this is 'constant' in the sense of having no dependence on
    the independent variable `x`, not in the sense of being non-varying.
    To be clear, `c` will be a Parameter that will be varied in the fit
    (by default, of course).
    """
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    """float: Factor used to create :func:`lmfit.models.fwhm_expr`."""
    flux_factor = np.sqrt(2 * np.pi)
    """float: Factor used to create :func:`flux_expr`."""

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        super().__init__(gaussianCH, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("height", min=0)
        self.set_param_hint("fwhm", expr=lmfit.models.fwhm_expr(self))
        self.set_param_hint("flux", expr=flux_expr(self))

    guess = _guess_1gauss

    # def guess(self, data, x=None, **kwargs):
    #     """Estimate initial model parameter values from data."""
    #     pars = self.make_params()

    #     pars[f'{self.prefix}c'].set(value=data.mean())
    #     return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC
    guess.__doc__ = lmfit.models.COMMON_GUESS_DOC

class CompositeModel(lmfit.model.CompositeModel):
    def __init__(self, left, right, op, **kws):
        """
        Parameters
        ----------
        left : Model
            Left-hand model.
        right : Model
            Right-hand model.
        op : callable binary operator
            Operator to combine `left` and `right` models.
        **kws : optional
            Additional keywords are passed to `Model` when creating this
            new model.
        Notes
        -----
        The two models must use the same independent variable.
        """
        if not isinstance(left, lmfit.Model):
            raise ValueError(f'CompositeModel: argument {left} is not a Model')
        if not isinstance(right, lmfit.Model):
            raise ValueError(f'CompositeModel: argument {right} is not a Model')
        if not callable(op):
            raise ValueError(f'CompositeModel: operator {op} is not callable')

        self.left = left
        self.right = right
        self.op = op

        name_collisions = set(left.param_names) & set(right.param_names)
        if len(name_collisions) > 0:
            msg = ''
            for collision in name_collisions:
                msg += (f"\nTwo models have parameters named '{collision}'; "
                        "use distinct names.")
            raise NameError(msg)

        # we assume that all the sub-models have the same independent vars
        if 'independent_vars' not in kws:
            kws['independent_vars'] = self.left.independent_vars
        if 'nan_policy' not in kws:
            kws['nan_policy'] = self.left.nan_policy

        # def _tmp(self, *args, **kws):
        #     pass
        lmfit.Model.__init__(self, _tmp, **kws)

        for side in (left, right):
            prefix = side.prefix
            for basename, hint in side.param_hints.items():
                self.param_hints[f"{prefix}{basename}"] = hint