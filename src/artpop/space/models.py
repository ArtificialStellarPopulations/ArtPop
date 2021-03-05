# Third-party
import numpy as np
from astropy.modeling import Fittable2DModel, Parameter


__all__ = ['Plummer2D']

class Plummer2D(Fittable2DModel):
    """
    Two-dimensional Plummer surface brightness profile.

    Parameters
    ----------
    amplitude : float
        Central surface brightness.
    scale_radius : float
        Characteristic scale radius of the mass distribution.
    x_0 : float, optional
        x position of the center.
    y_0 : float, optional
        y position of the center.
    """

    amplitude = Parameter(default=1)
    scale_radius = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)

    @classmethod
    def evaluate(cls, x, y, amplitude, scale_radius, x_0, y_0):
        """Two-dimensional Plummer profile evaluation function."""
        r = np.sqrt((x - x_0)**2 + (y - y_0)**2)
        return amplitude / (1 + (r / scale_radius)**2)**2
