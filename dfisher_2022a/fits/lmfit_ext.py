# temporary placeholder for extended functionality for lmfit

import lmfit
import numpy as np

# lmfit.minimizer.MinimizerResult is a placeholder to add more attributes
# and offer statistic analysis if certain attributes are provided

def aic_real(self):
    """Chisqr + 2 * nvarys."""
    try:
        _neg2_log_likel = self.chisqr
        nvarys = getattr(self, "nvarys", len(self.init_vals))
        return _neg2_log_likel + 2 * nvarys
    except (AttributeError, TypeError):
        return None


def bic_real(self):
    """Chisqr + np.log(ndata) * nvarys."""
    try:
        _neg2_log_likel = self.chisqr
        # fallback incase nvarys or ndata not defined.
        nvarys = getattr(self, "nvarys", len(self.init_vals))
        ndata = getattr(self, "ndata", len(self.residual))
        return _neg2_log_likel + np.log(ndata) * nvarys
    except (AttributeError, TypeError):
        return None
