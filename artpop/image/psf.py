# Third-party
import numpy as np
from astropy import units as u
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel, Moffat2DKernel

# Project
from ..util import check_units, check_odd


__all__ = ['gaussian_psf', 'moffat_psf']


def _check_shape(shape):
    """Make sure shape is a tuple."""
    if type(shape) == int:
        shape = (shape, shape)
    check_odd(shape, 'psf shape')
    return shape


def gaussian_psf(fwhm, pixel_scale=0.2, shape=41):
    """
    Gaussian point-spread function.

    Parameters
    ----------
    fwhm : float or `~astropy.units.Quantity`
        Full width at half maximum of the psf. If a float is given, the units
        will be assumed to be arcsec. The units can be angular or in pixels.
    pixel_scale : float or `~astropy.units.Quantity`, optional
        The pixel scale of the psf image. If a float is given, the units are 
        assumed to be arcsec / pixel (why would you want anything different?).
    shape : int or list-like, optional
        Shape of the psf image. Must be odd. If an int is given, the x and y 
        dimensions will be set to this value: (shape, shape).

    Returns
    -------
    psf : `~numpy.ndarray`
        The PSF image normalized such that its sum is equal to one. 
    """

    fwhm = check_units(fwhm, 'arcsec')
    pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel)
    y_size, x_size = _check_shape(shape)

    width = fwhm.to('pixel', u.pixel_scale(pixel_scale)).value
    width *= gaussian_fwhm_to_sigma
    model = Gaussian2DKernel(x_stddev=width,
                             y_stddev=width,
                             x_size=x_size,
                             y_size=y_size)
    model.normalize()
    psf = model.array
    return psf


def moffat_psf(fwhm, pixel_scale=0.2, shape=41, alpha=4.765):
    """
    Moffat point-spread function.

    Parameters
    ----------
    fwhm : float or `~astropy.units.Quantity`
        Full width at half maximum of the psf. If a float is given, the units
        will be assumed to be arcsec. The units can be angular or in pixels.
    pixel_scale : float or `~astropy.units.Quantity`, optional
        The pixel scale of the psf image. If a float is given, the units are 
        assumed to be arcsec / pixel (why would you want anything different?).
    shape : int or list-like, optional
        Shape of the psf image. Must be odd. If an int is given, the x and y 
        dimensions will be set to this value: (shape, shape).
    alpha : float, optional
        Power index of the Moffat model.

    Returns
    -------
    psf : `~numpy.ndarray`
        The PSF image normalized such that its sum is equal to one. 

    Notes
    -----
    The default value `alpha = 4.765` is a fit to the prediction from 
    atmospheric turbulence theory (`Trujillo et al. 2001 
    <https://ui.adsabs.harvard.edu/abs/2001MNRAS.328..977T/abstract>`_).
    """
    fwhm = check_units(fwhm, 'arcsec')
    pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel)
    y_size, x_size = _check_shape(shape)

    width = fwhm.to('pixel', u.pixel_scale(pixel_scale)).value
    gamma = width / (2 * np.sqrt(2**(1 / alpha) - 1))
    model = Moffat2DKernel(gamma=gamma,
                           alpha=alpha,
                           x_size=x_size,
                           y_size=y_size)
    model.normalize()
    psf = model.array
    return psf
