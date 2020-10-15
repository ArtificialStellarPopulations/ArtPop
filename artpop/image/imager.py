import abc
import numpy as np
from astropy import units as u
from astropy.convolution import convolve_fft
from fast_histogram import histogram2d
from ..utils import check_units, check_xy_dim, check_random_state
from ..source import Source
from ..log import logger


__all__ = ['IdealImager', 'ArtImager']


class Imager(metaclass=abc.ABCMeta):

    def __init__(self, source, xy_dim, pixel_scale=0.2): 
        if Source not in source.__class__.__mro__:
            raise Exception(f'{type(src)} is not a valid Source object')

        self.src = source
        self.pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel)
        self.pixel_scale.to('arcsec / pixel').value
        self.xy_dim = check_xy_dim(xy_dim)
        self.filters = source.mags.colnames

    def inject(self, band, signal, zpt=27):
        if band not in self.filters:
            raise Exception(f'you do not seem to have {band}-band magnitudes')

        bins = tuple(self.xy_dim.astype(int))
        hist_range = [[0, self.xy_dim[0] - 1],
                      [0, self.xy_dim[1] - 1]]

        image = histogram2d(self.src.x, self.src.y, bins=bins,
                            weights=signal, range=hist_range).T

        return image

    def observe(self, band, psf, **kwargs):
        return NotImplementedError()


class IdealImager(Imager):

    def observe(self, band, psf=None, zpt=27, convolve_boundary='wrap'):
        flux = 10**(0.4 * (zpt - self.src.mags[band]))
        image = self.inject(band, flux)

        if psf is not None:
            image = convolve_fft(image, psf,
                                 boundary=convolve_boundary,
                                 normalize_kernel=True)
        return image


class ArtImager(Imager):

    def __init__(self, source, xy_dim, diameter, read_noise=None, 
                 pixel_scale=0.2, random_state=None):

        super(ArtImager, self).__init__(source, xy_dim, pixel_scale)
        self.diameter = check_units(diameter, 'm')
        self.read_noise = read_noise
        self.rng = check_random_state(random_state)
