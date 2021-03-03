# Standard library
import os
import abc
import pickle
from collections import namedtuple

# Third-party
import numpy as np
from astropy import units as u
from astropy import constants
from astropy.convolution import convolve_fft
from fast_histogram import histogram2d

# Project
from ..util import check_units, check_random_state, data_dir
from ..filters import FilterSystem, get_filter_names
from ..source import Source
from ..log import logger


__all__ = ['IdealObservation', 'ArtObservation', 'IdealImager', 'ArtImager']


class BaseObservation(metaclass=abc.ABCMeta):

    def to_pickle(self, file_name):
        pkl_file = open(file_name, 'wb')
        pickle.dump(self, pkl_file)
        pkl_file.close()

    @staticmethod
    def from_pickle(file_name):
        pkl_file = open(file_name, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        return data


# return object for the ideal imager
class IdealObservation(BaseObservation):

    def __init__(self, image, zpt, bandpass):
        self.image = image
        self.zpt = zpt
        self.bandpass = bandpass


# return object for the artificial imager
class ArtObservation(BaseObservation):

    def __init__(self, raw_counts, src_counts, sky_counts, image,
                 var_image, calibration, zpt, bandpass, exptime):

        self.raw_counts = raw_counts
        self.src_counts = src_counts
        self.sky_counts = sky_counts
        self.image = image
        self.var_image = var_image
        self.calibration = calibration
        self.zpt = zpt
        self.bandpass = bandpass
        self.exptime = exptime


mAB_0 = 48.6
def fnu_from_AB_mag(mag):
    """
    Convert AB magnitude into flux density fnu in cgs units.
    """
    fnu = 10.**((mag + mAB_0)/(-2.5))
    return fnu*u.erg/u.s/u.Hz/u.cm**2


class Imager(metaclass=abc.ABCMeta):
    """
    Base class for Imager objects.

    Parameters
    ----------
    phot_system : str or list-like
        Name of the photometric system(s).
    """

    def __init__(self, phot_system):
        self.phot_system = phot_system
        self.filters = get_filter_names(phot_system)

    def _check_source(self, source):
        """Verify that input object is a Source object."""
        if Source not in source.__class__.__mro__:
            raise Exception(f'{type(src)} is not a valid Source object')

    def _check_bandpass(self, bandpass):
        """Verify bandpass is the loaded filter list."""
        if bandpass not in self.filters:
            raise Exception(f'you do not seem to have {bandpass}-band mags')

    def inject_stars(self, x, y, signal, xy_dim):
        """
        Inject sources into image.

        Parameters
        ----------
        x : `~numpy.ndarray`
            Source x coordinates.
        y : `~numpy.ndarray`
            Source y coordinates.
        signal : `~numpy.ndarray`
            The signal to inject into image (e.g., flux, counts, etc...).
        xy_dim : int or list-like
            The dimensions of mock image in xy units. If `int` is given, it is
            assumed to be both the x and y dimensions: (xy_dim, xy_dim).

        Returns
        -------
        image : `~numpy.ndarray`
            The mock image *before* psf convolution.
        """
        bins = tuple(np.asarray(xy_dim).astype(int))
        hist_range = [[0, bins[0] - 1], [0, bins[1]- 1]]
        image = histogram2d(x, y, bins=bins,
                            weights=signal, range=hist_range).T
        return image

    def apply_seeing(self, image, psf=None, boundary='wrap', **kwargs):
        """
        Convolve mock image with psf.

        Parameters
        ----------
        image : `~numpy.ndarray`
            The mock image.
        psf : `~numpy.ndarray` or None, optional
            The point-spread function. If None, will return the input image.
        boundary : {'fill', 'wrap'}, optional
            A flag indicating how to handle boundaries:

                * 'fill': set values outside the array boundary to fill_value
                  (default)
                * 'wrap': periodic boundary

            The `None` and 'extend' parameters are not supported for FFT-based
            convolution.

        Returns
        -------
        image : `~numpy.ndarray`
            The psf-convolved image. If `psf` is not `None`, the original image
            is returned.
        """
        if psf is not None:
            image = convolve_fft(image , psf,
                                 boundary=boundary,
                                 normalize_kernel=True)
        return image

    def inject_smooth_model(self, bimage, source, bandpass, zpt):
        """Inject smooth model if it exists."""
        return NotImplementedError()

    def observe(self, bandpass, psf, **kwargs):
        """Mock observe source."""
        return NotImplementedError()


class IdealImager(Imager):
    """Ideal imager for making noise-free images."""

    def inject_smooth_model(self, image, source, bandpass, zpt):
        if hasattr(source, 'smooth_model'):
            if source.smooth_model is None:
                image = image
            else:
                mag = source.ssp.mag_integrated_component(bandpass)
                amp, amp_image, name = source.mag_to_image_amplitude(mag, zpt)
                setattr(source.smooth_model, name, amp_image)
                yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
                image = image + source.smooth_model(xx, yy)
        else:
            image = image
        return image

    def observe(self, source, bandpass, psf=None, zpt=27, **kwargs):
        """
        Make ideal observation.

        Parameters
        ----------
        source : `~artpop.source.Source`
            Source to mock observer.
        bandpass : str
            Filter of observation. Must be a filter in the given
            photometric system(s).
        psf : `~numpy.ndarray` or None, optional
            The point-spread function. If None, will not psf-convolve image.
        zpt : float, optional
            The magnitude zero point of the mock image.

        .. note::
            The returned parameters are stored as attributes of a
            `~collections.namedtuple` object.

        Returns
        -------
        image : `~numpy.ndarray`
            The ideal mock image.
        bandpass : str
            Filter of the observation.
        zpt : float
            The magnitude zero point.
        """
        self._check_bandpass(bandpass)
        self._check_source(source)
        flux = 10**(0.4 * (zpt - source.mags[bandpass]))
        image = self.inject_stars(source.x, source.y, flux, source.xy_dim)
        image = self.inject_smooth_model(image, source, bandpass, zpt)
        image = self.apply_seeing(image, psf, **kwargs)
        observation = IdealObservation(image=image, bandpass=bandpass, zpt=zpt)
        return observation


class ArtImager(Imager):
    """
    Imager for making fully artificial images.

    Parameters
    ----------
    phot_system : str or list-like
        Name of the photometric system(s).
    diameter : float or `~astropy.units.Quantity`, optional
        Diameter of the telescope aperture. If a float is given, the units
        will be assumed to be meters.
    read_noise : float, optional
        RMS of Gaussian read noise. Set to zero by default.
    throughput : float, optional
        Throughput factor (e.g., to account for QE). Set to one by default.
        Note the filter response curves used by MIST already include
        atmospheric transmission if applicable.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.
    """

    def __init__(self, phot_system, diameter=10, read_noise=0.0, throughput=1,
                 random_state=None, **kwargs):
        super(ArtImager, self).__init__(phot_system)
        self.throughput = throughput
        self.read_noise = read_noise
        self.diameter = check_units(diameter, 'm')
        self.rng = check_random_state(random_state)
        self.filter_system = FilterSystem(self.phot_system, **kwargs)

    @property
    def area(self):
        """Area of aperture."""
        r = self.diameter / 2
        return np.pi * r**2

    def mag_to_counts(self, mags, bandpass, exptime):
        """
        Convert magnitudes to counts.

        Parameters
        ----------
        mags : `~numpy.ndarray` or `~astropy.table.Column`
            AB magnitudes to be converted into counts.
        bandpass : str
            Filter of observation. Must be a filter in the given
            photometric system(s).
        exptime : float or `~astropy.units.Quantity`
            Exposure time. If float is given, the units are assumed to
            be `~astropy.units.second`.

        Returns
        -------
        counts : `~numpy.ndarray`
            The counts associated with each magnitude.
        """
        exptime = check_units(exptime, 's')
        fnu = fnu_from_AB_mag(mags)
        dlam = self.filter_system.dlam(bandpass)
        lam_eff = self.filter_system.lam_eff(bandpass)
        photon_flux = (dlam / lam_eff) * fnu / constants.h.to('erg s')
        counts = photon_flux * self.area.to('cm2') * exptime.to('s')
        counts = counts.decompose()
        assert counts.unit == u.dimensionless_unscaled
        counts *= self.throughput
        return counts.value

    def sb_to_counts_per_pixel(self, sb, bandpass, exptime, pixel_scale):
        """
        Convert a constant surface brightness into counts per pixel.

        Parameters
        ----------
        sb : float
            Surface brightness in units of `~astropy.units.mag` per square
            `~astropy.units.arcsec`.
        bandpass : str
            Filter of observation. Must be a filter in the given
            photometric system(s).
        exptime : float or `~astropy.units.Quantity`
            Exposure time. If float is given, the units are assumed to
            be `~astropy.units.second`.
        pixel_scale : `~astropy.units.Quantity`
            Pixel scale.

        Returns
        -------
        counts_per_pixel : float
            Counts per pixel associated with the given surface brightness.
        """
        exptime = check_units(exptime, 's')
        dlam = self.filter_system.dlam(bandpass)
        lam_eff = self.filter_system.lam_eff(bandpass)
        pixel_scale = pixel_scale.to('arcsec / pixel')
        E_lam = (constants.h * constants.c / lam_eff).decompose().to('erg')
        fnu_per_square_arcsec = fnu_from_AB_mag(sb) / u.arcsec**2
        flam_per_square_arcsec = fnu_per_square_arcsec *\
            constants.c.to('angstrom/s') / lam_eff**2
        flam_per_pixel = flam_per_square_arcsec * pixel_scale**2
        photon_flux_per_sq_pixel = (flam_per_pixel * dlam / E_lam).\
            decompose().to('1/(cm2*pix2*s)')
        counts_per_pixel = photon_flux_per_sq_pixel * exptime.to('s')
        counts_per_pixel *= self.area.to('cm2') * u.pixel**2
        assert counts_per_pixel.unit == u.dimensionless_unscaled
        counts_per_pixel *= self.throughput
        return counts_per_pixel.value

    def calibration(self, bandpass, exptime, zpt=27.0):
        """
        Calculate the calibration factor, which converts counts into
        calibrated flux units.

        Parameters
        ----------
        bandpass : str
            Filter of observation. Must be a filter in the given
            photometric system(s).
        exptime : float or `~astropy.units.Quantity`
            Exposure time. If float is given, the units are assumed to
            be `~astropy.units.second`.
        zpt : float, optional
            The magnitude zero point of the mock image.

        Returns
        -------
        cali_factor : float
            The calibration factor.
        """
        fs = self.filter_system
        dlam_over_lam = fs.dlam(bandpass) / fs.lam_eff(bandpass)
        lam_factor = (dlam_over_lam / constants.h.cgs).value
        cali_factor = 10**(0.4 * zpt) * 10**(0.4 * mAB_0) / lam_factor
        cali_factor /= exptime.to('s').value
        cali_factor /= self.area.to('cm2').value
        return cali_factor

    def inject_smooth_model(self, image, source, bandpass, exptime, zpt):
        if hasattr(source, 'smooth_model'):
            if source.smooth_model is None:
                image = image
            else:
                flux = np.sum(10**(0.4 * (zpt - source.smooth_mags[bandpass])))
                mag = zpt - 2.5 * np.log10(flux)
                amp, _, name = source.mag_to_image_amplitude(mag, zpt)
                amp_counts = self.sb_to_counts_per_pixel(
                    amp, bandpass, exptime, source.pixel_scale)
                setattr(source.smooth_model, name, amp_counts)
                yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
                image = image + source.smooth_model(xx, yy)
        else:
            image = image
        return image

    def observe(self, source, bandpass, exptime, sky_sb=None, psf=None,
                zpt=27.0, **kwargs):
        """
        Make artificial observation.

        .. note::
            The returned parameters are stored as attributes of a
            `~collections.namedtuple` object.

        Parameters
        ----------
        source : `~artpop.source.Source`
            Artificial source to be observed.
        bandpass : str
            Filter of observation. Must be a filter in the given
            photometric system(s).
        exptime : float or `~astropy.units.Quantity`
            Exposure time. If `float` is given, the units are assumed to
            be `~astropy.units.second`.
        sky_sb : float or None, optional
            Constant surface brightness of the sky. If None is given, then no
            sky noise will be added.
        psf : `~numpy.ndarray` or None, optional
            The point-spread function. If None, will not psf-convolve image.
        zpt : float, optional
            The magnitude zero point of the mock image.

        Returns
        -------
        raw_counts : `~numpy.ndarray`
            Raw count image, including Poisson noise.
        src_counts : `~numpy.ndarray`
            Source count image before Poission noise is added.
        sky_counts : float
            Counts per pixel from the sky.
        var_image : `~numpy.ndarray`
            Variance image.
        calibration : float
            Calibration factor that converts counts to calibrated flux units.
        bandpass : str
            Filter of the observation.
        zpt : float
            The magnitude zero point.
        exptime : `~astropy.units.Quantity`
            The exposure time of the mock observation.
        """
        self._check_bandpass(bandpass)
        self._check_source(source)
        exptime = check_units(exptime, 's')
        if sky_sb is not None:
            sky_counts = self.sb_to_counts_per_pixel(
                sky_sb, bandpass, exptime, source.pixel_scale)
        else:
            sky_counts = 0
        counts = self.mag_to_counts(source.mags[bandpass], bandpass, exptime)
        src_counts = self.inject_stars(source.x, source.y, counts, source.xy_dim)
        src_counts = self.inject_smooth_model(src_counts, source,
                                              bandpass, exptime, zpt)
        src_counts = self.apply_seeing(src_counts, psf, **kwargs)
        if sky_sb is None:
            src_counts[src_counts < 0] = 0
        raw_counts = self.rng.poisson(src_counts + sky_counts)
        if self.read_noise > 0.0:
            rn = self.rng.normal(scale=self.read_noise, size=src_counts.shape)
            raw_counts = raw_counts + rn
        cali = self.calibration(bandpass, exptime, zpt)
        image_cali = (raw_counts - sky_counts) * cali
        var = raw_counts + self.read_noise**2
        observation = ArtObservation(
            raw_counts=raw_counts,
            src_counts=src_counts,
            sky_counts=sky_counts,
            image=image_cali,
            var_image=var,
            calibration=cali,
            zpt=zpt,
            bandpass=bandpass,
            exptime=exptime
        )

        return observation
