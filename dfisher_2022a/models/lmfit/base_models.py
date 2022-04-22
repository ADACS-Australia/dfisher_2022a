# base models

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
