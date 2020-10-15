import numpy as np
from astropy import units as u
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel, Moffat2DKernel
from ..utils import check_units, check_odd


__all__ = ['gaussian_psf', 'moffat_psf']


def _check_shape(shape):
    if type(shape) == int:
        shape = (shape, shape)
    check_odd(shape, 'psf shape')
    return shape


def gaussian_psf(fwhm, pixel_scale=0.2, shape=41):

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


def moffat_psf(fwhm, pixel_scale=0.2, shape=41, moffat_alpha=4.765):

    fwhm = check_units(fwhm, 'arcsec')
    pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel)
    y_size, x_size = _check_shape(shape)

    width = fwhm.to('pixel', u.pixel_scale(pixel_scale)).value
    gamma = width / (2 * np.sqrt(2**(1 / moffat_alpha) - 1))
    model = Moffat2DKernel(gamma=gamma,
                           alpha=moffat_alpha,
                           x_size=x_size,
                           y_size=y_size)
    model.normalize()
    psf = model.array
    return psf
