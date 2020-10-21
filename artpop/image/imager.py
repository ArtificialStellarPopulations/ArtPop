import os
import abc
from collections import namedtuple
import numpy as np
from astropy import units as u
from astropy import constants
from astropy.convolution import convolve_fft
from fast_histogram import histogram2d
from ..utils import check_units, check_xy_dim, check_random_state
from ..filters import phot_system_lookup, FilterSystem
from ..source import Source
from ..log import logger
from .. import data_dir


__all__ = ['IdealImager', 'ArtImager']


mAB_0 = 48.6
def fnu_from_AB_mag(mag):
    fnu = 10.**((mag + mAB_0)/(-2.5))
    return fnu*u.erg/u.s/u.Hz/u.cm**2


class Imager(metaclass=abc.ABCMeta):

    def __init__(self, source, fov, pixel_scale=0.2): 
        if Source not in source.__class__.__mro__:
            raise Exception(f'{type(src)} is not a valid Source object')

        self.src = source
        self.pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel)

        p = u.pixel_scale(self.pixel_scale)
        self.fov = check_units(fov, 'pixel').to('arcmin', p) 
        self.xy_dim = check_xy_dim(self.fov.to('pixel', p).value, False)
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

    def __init__(self, source, diameter, fov, read_noise=None, 
                 pixel_scale=0.2, throughput=1, random_state=None, **kwargs):

        super(ArtImager, self).__init__(source, fov, pixel_scale)
        self.diameter = check_units(diameter, 'm')
        self.read_noise = read_noise
        self.rng = check_random_state(random_state)
        self.phot_system = [phot_system_lookup(f) for f in self.filters]
        self.phot_system = np.unique(self.phot_system)
        self.filter_system = FilterSystem(self.phot_system, **kwargs)
        self.throughput = throughput

    @property
    def area(self):
        r = self.diameter / 2
        return np.pi * r**2

    def mag_to_counts(self, band, exptime):
        exptime = check_units(exptime, 's')
        fnu = fnu_from_AB_mag(self.src.mags[band])
        dlam = self.filter_system.dlam(band)
        lam_eff = self.filter_system.lam_eff(band)
        photon_flux = (dlam / lam_eff) * fnu / constants.h.to('erg s')
        counts = photon_flux * self.area.to('cm2') * exptime.to('s')
        counts = counts.decompose()
        assert counts.unit == u.dimensionless_unscaled
        counts *= self.throughput
        return counts.value

    def mu_to_counts_per_pixel(self, mu, band, exptime):
        exptime = check_units(exptime, 's')
        dlam = self.filter_system.dlam(band)
        lam_eff = self.filter_system.lam_eff(band)
        pixscale = self.pixel_scale.to('arcsec / pixel')

        E_lam = (constants.h * constants.c / lam_eff).decompose().to('erg')
        fnu_per_square_arcsec = fnu_from_AB_mag(mu) / u.arcsec**2
        flam_per_square_arcsec = fnu_per_square_arcsec *\
            constants.c.to('angstrom/s') / lam_eff**2

        flam_per_pixel = flam_per_square_arcsec * pixscale**2
        photon_flux_per_sq_pixel = (flam_per_pixel * dlam / E_lam).\
            decompose().to('1/(cm2*pix2*s)')
        counts_per_pixel = photon_flux_per_sq_pixel * exptime.to('s')
        counts_per_pixel *= self.area.to('cm2') * u.pixel**2
        assert counts_per_pixel.unit == u.dimensionless_unscaled
        counts_per_pixel *= self.throughput
        return counts_per_pixel.value
