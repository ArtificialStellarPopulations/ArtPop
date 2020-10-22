import os
import abc
from collections import namedtuple
import numpy as np
from astropy import units as u
from astropy import constants
from astropy.convolution import convolve_fft
from fast_histogram import histogram2d
from ..utils import check_units, check_xy_dim, check_random_state
from ..filters import phot_system_lookup, FilterSystem, get_filter_names
from ..source import Source
from ..log import logger
from .. import data_dir


__all__ = ['IdealImager', 'ArtImager']


ArtImage = namedtuple(
    'ArtImage',
    'raw_counts src_counts sky_counts '
    'image noise calibration '
    'zpt band exptime'
)


mAB_0 = 48.6
def fnu_from_AB_mag(mag):
    fnu = 10.**((mag + mAB_0)/(-2.5))
    return fnu*u.erg/u.s/u.Hz/u.cm**2


class Imager(metaclass=abc.ABCMeta):

    def __init__(self, phot_system): 
        self.phot_system = phot_system
        self.filters = get_filter_names(phot_system)

    def _check_source(self, source):
        if Source not in source.__class__.__mro__:
            raise Exception(f'{type(src)} is not a valid Source object')

    def _check_band(self, band):
        if band not in self.filters:
            raise Exception(f'you do not seem to have {band}-band magnitudes')

    def inject(self, x, y, signal, xy_dim):
        bins = tuple(np.asarray(xy_dim).astype(int))
        hist_range = [[0, bins[0] - 1], [0, bins[1]- 1]]
        image = histogram2d(x, y, bins=bins, 
                            weights=signal, range=hist_range).T
        return image

    def apply_seeing(self, signal, psf=None, boundary='wrap', **kwargs):
        if psf is not None:
            signal = convolve_fft(signal, psf,
                                 boundary=boundary,
                                 normalize_kernel=True)
        return signal

    def observe(self, band, psf, **kwargs):
        return NotImplementedError()


class IdealImager(Imager):

    def observe(self, source, band, psf=None, zpt=27, **kwargs):
        self._check_band(band)
        self._check_source(source)
        flux = 10**(0.4 * (zpt - source.mags[band]))
        image = self.inject(source.x, source.y, flux, source.xy_dim)
        image = self.apply_seeing(image, psf, **kwargs)
        return image


class ArtImager(Imager):

    def __init__(self, diameter, phot_system, read_noise=0.0, throughput=1, 
                 random_state=None, **kwargs):
        super(ArtImager, self).__init__(phot_system)
        self.read_noise = read_noise
        self.diameter = check_units(diameter, 'm')
        self.rng = check_random_state(random_state)
        self.filter_system = FilterSystem(self.phot_system, **kwargs)
        self.throughput = throughput

    @property
    def area(self):
        r = self.diameter / 2
        return np.pi * r**2

    def mag_to_counts(self, mags, band, exptime):
        exptime = check_units(exptime, 's')
        fnu = fnu_from_AB_mag(mags)
        dlam = self.filter_system.dlam(band)
        lam_eff = self.filter_system.lam_eff(band)
        photon_flux = (dlam / lam_eff) * fnu / constants.h.to('erg s')
        counts = photon_flux * self.area.to('cm2') * exptime.to('s')
        counts = counts.decompose()
        assert counts.unit == u.dimensionless_unscaled
        counts *= self.throughput
        return counts.value

    def mu_to_counts_per_pixel(self, mu, band, exptime, pixscale):
        exptime = check_units(exptime, 's')
        dlam = self.filter_system.dlam(band)
        lam_eff = self.filter_system.lam_eff(band)
        pixscale = pixscale.to('arcsec / pixel')

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

    def calibration(self, band, exptime, zpt=27.0):
        fs = self.filter_system
        dlam_over_lam = fs.dlam(band) / fs.lam_eff(band)
        lam_factor = (dlam_over_lam / constants.h.cgs).value
        cali_factor = 10**(0.4 * zpt) * 10**(0.4 * mAB_0) / lam_factor
        cali_factor /= exptime.to('s').value
        cali_factor /= self.area.to('cm2').value
        return cali_factor

    def observe(self, source, band, exptime, psf=None, zpt=27.0, 
                mu_sky=None, **kwargs):
        self._check_band(band)
        self._check_source(source)
        exptime = check_units(exptime, 's')
        if mu_sky is not None:
            sky_counts = self.mu_to_counts_per_pixel(
                mu_sky, band, exptime, source.pixel_scale)
        else:
            sky_counts = 0
        counts = self.mag_to_counts(source.mags[band], band, exptime)
        src_counts = self.inject(source.x, source.y, counts, source.xy_dim)
        src_counts = self.apply_seeing(src_counts, psf, **kwargs)
        if mu_sky is None:
            src_counts[src_counts < 0] = 0
        raw_counts = self.rng.poisson(src_counts + sky_counts)
        if self.read_noise > 0.0:
            rn = self.rng.normal(scale=self.read_noise, size=src_counts.shape)
            raw_counts = raw_counts + rn

        cali = self.calibration(band, exptime, zpt)
        image_cali = (raw_counts - sky_counts) * cali
        noise = np.sqrt(raw_counts + self.read_noise**2)

        results = ArtImage(raw_counts=raw_counts,
                           src_counts=src_counts,
                           sky_counts=sky_counts,
                           image=image_cali,
                           noise=noise,
                           calibration=cali,
                           zpt=zpt, band=band,
                           exptime=exptime)

        return results
