import os
import numpy as np
from collections import namedtuple
from astropy import units as u
from astropy import constants
from astropy.table import Table
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import convolve_fft, Gaussian2DKernel, Moffat2DKernel
from fast_histogram import histogram2d
from astropy.nddata.utils import block_reduce
from ..utils import check_random_state, fetch_psf
from ..log import logger
from .. import data_dir
from .filter_info import *
filter_dir = os.path.join(data_dir, 'filters')

__all__ = ['filter_dir', 'ArtPSF', 'FilterSystem', 
           'ArtObservatory', 'fnu_from_AB_mag']

mAB_0 = 48.6
ObsData = namedtuple(
    'ObsData', 
    'raw_counts src_counts sky_counts '
    'flux_cali_img noise cali_factor '
    'zpt band exptime'
)

phot_labels = dict(HSC='hsc', Bessell='Bessell', SDSS='SDSS', LSST='LSST')

vega_to_ab = dict(
    Bessell=dict(U=0.800527, B=-0.107512, V=0.006521, R=0.190278, I=0.431372)
)



def fnu_from_AB_mag(mag):
    fnu = 10.**((mag + mAB_0)/(-2.5))
    return fnu*u.erg/u.s/u.Hz/u.cm**2


class ArtPSF(object):

    def __init__(self, kind, fwhm, filters, pixscale, 
                 moffat_alpha=4.765, x_size=41, **kwargs):
        
        if type(fwhm) != list:
            fwhm = [fwhm]

        assert len(fwhm) == len(filters), '# of fwhm must equal # of bands'
        assert x_size % 2 != 0, 'x_size must be odd'
        self.x_size = x_size

        for filt, width in zip(filters, fwhm):
            
            if kind.lower() == 'moffat':
                width = width / pixscale
                gamma = width / (2 * np.sqrt(2**(1/moffat_alpha) - 1))
                model = Moffat2DKernel(gamma=gamma, 
                                       alpha=moffat_alpha, 
                                       x_size=x_size, 
                                       **kwargs)

            elif kind.lower() == 'gaussian':
                width = gaussian_fwhm_to_sigma * width / pixscale
                model = Gaussian2DKernel(x_stddev=width, 
                                         y_stddev=width, 
                                         x_size=x_size,
                                         **kwargs)

            else:
                raise Exception('{} is not a valid PSF model'.format(kind))

            model.normalize()

            setattr(self, 'model_' + filt, model)
            setattr(self, filt + '_fwhm', width * pixscale * u.arcsec)
            setattr(self, filt, model.array)


class FilterSystem(object):
    """
    Reference: Fukugita et al. (1996) AJ 111, 1748.
    """
    
    def __init__(self, phot_system, data_path=filter_dir):
        files = os.listdir(data_path)
        files = [fn for fn in files if phot_system.lower() in fn]
        self.names = [b.split('_')[-1][0] for b in files]
        self.phot_system = phot_system
        
        for fn in files:
            f = fn.split('_')[-1][0]
            fn = os.path.join(data_path, fn)
            setattr(self, f, Table.read(fn))
        
    def _get_trans(self, band):
        lam = getattr(self, band)['wave']
        trans = getattr(self, band)['trans']
        return lam, trans

    def effective_throughput(self, band):
        lam, trans = self._get_trans(band)
        t = np.trapz(trans, np.log(lam))
        return t
    
    def lam_eff(self, band):
        lam, trans = self._get_trans(band)
        log_leff = np.trapz(np.log(lam) * trans, np.log(lam)) 
        log_leff /= np.trapz(trans, np.log(lam))
        leff =  np.exp(log_leff) * u.angstrom
        return leff
    
    def lam_pivot(self, band):
        lam, trans = self._get_trans(band)
        lpivot = np.trapz(lam * trans, lam)
        lpivot /= np.trapz(trans, np.log(lam))
        lpivot = np.sqrt(lpivot) * u.angstrom
        return lpivot
    
    def dlam(self, band):
        lam, trans = self._get_trans(band)
        norm = np.trapz(trans, lam)
        width = (norm / trans.max()) * u.angstrom
        return width
    

class ArtObservatory(object):
    
    def __init__(self, diameter, phot_system, psf=None, bands=None,
                 t_atm=1.0, t_opt=1.0, qe=1.0, read_noise=4, 
                 random_state=None):
        
        if type(diameter) != u.Quantity:
            logger.warning('Will assume diameter is in meters.')
            logger.warning('Use astropy.units to specify units.')
            diameter *= u.m
        
        self.diameter = diameter
        self.t_atm = t_atm
        self.t_opt = t_opt
        self.qe = qe
        self.read_noise = float(read_noise)
        self.filters = FilterSystem(phot_system)
        self.phot_system = phot_system
        self.rng = check_random_state(random_state)
        self.raw_image = None
        
        if psf is None:
            bands = list(bands) if bands is not None else self.filters.names
            PSF = namedtuple('PSF', bands) 
            psf = [fetch_psf(phot_system, b) for b in bands]
            self._psf = PSF(*psf)
        elif type(psf) == dict:
            bands = list(bands) if bands is not None else self.filters.names
            PSF = namedtuple('PSF', bands) 
            self._psf = PSF(*[psf[b] for b in bands])
        else:
            assert type(psf) == ArtPSF, 'psf must be of type ArtPSF'
            self._psf = psf

    @property
    def area(self):
        r = self.diameter / 2
        return np.pi * r**2
    
    @property
    def throughput(self):
        return self.t_atm * self.t_opt * self.qe

    def instrumental_zpt(self, band):
        unit_time = 1 * u.s
        photon_flux_unit = 1 / self.area.to('cm2') / unit_time 

        # divide because we want time to get to 1 count / sec
        photon_flux_unit /= self.filters.effective_throughput(band)
        photon_flux_unit /= self.throughput 

        dlam = self.filters.dlam(band)
        lam_eff = self.filters.lam_eff(band)
        factor = constants.h.to('erg s') * lam_eff / dlam
        fnu_unit = photon_flux_unit * factor
        m_1 = -2.5 * np.log10(fnu_unit.value) - mAB_0
        return m_1
                    
    def psf(self, band):
        return getattr(self._psf, band)
    
    def _inject_into_image(self, flux, band, boundary='wrap', phase='all'):
       
        if phase != 'all':
            phase_mask = self.art_pop.pop.get_phase_mask(phase)
            _x = self.art_pop.star_x[phase_mask]
            _y = self.art_pop.star_y[phase_mask]
        else:
            _x = self.art_pop.star_x
            _y = self.art_pop.star_y
            
        bins = tuple(self.xy_image_dim.astype(int))
        hist_range = [[0, self.xy_image_dim[0]-1], 
                      [0, self.xy_image_dim[1]-1]]
        
        image = histogram2d(_x, _y, bins=self.xy_image_dim,
                             weights=flux, range=hist_range).T
        
        if self._psf is not None:
            image = convolve_fft(image, self.psf(band), 
                                 boundary=boundary,
                                 normalize_kernel=True)
        return image 
    
    def _mag_to_counts(self, band, exptime, phase='all', mag=None):
        
        label = phot_labels[self.phot_system]

        if mag is None:
            mag = self.art_pop.pop.star_mags(label + '_' + band, phase)
        elif type(mag) == float or type(mag) == int or type(mag) == np.float64:
            mag = np.array([mag])
        else:
            mag = np.asarray(mag)

        if self.phot_system in vega_to_ab.keys():
            mag += vega_to_ab[self.phot_system][band]

        fnu = fnu_from_AB_mag(mag + self.art_pop.dist_mod)      
        
        dlam = self.filters.dlam(band)
        lam_eff = self.filters.lam_eff(band)
        photon_flux = (dlam / lam_eff) * fnu / constants.h.to('erg s')
        counts = photon_flux * self.area.to('cm2') * exptime.to('s')
        counts = counts.decompose()
        assert counts.unit == u.dimensionless_unscaled
        counts *= self.throughput
        return counts.value
    
    def sb_to_counts(self, mu, band, exptime, downsample=1):

        dlam = self.filters.dlam(band)
        lam_eff = self.filters.lam_eff(band)
        
        pixscale = downsample * self.art_pop.pixscale * u.arcsec / u.pixel
        
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

    def cali_factor(self, band, exptime, zpt=27):
        dlam_over_lam = self.filters.dlam(band) / self.filters.lam_eff(band)
        lam_factor = (dlam_over_lam / constants.h.cgs).value
        cali_factor = 10**(0.4 * zpt) * 10**(0.4 * mAB_0) / lam_factor 
        cali_factor /= exptime.to('s').value
        cali_factor /= self.area.to('cm2').value
        return cali_factor

    def observe(self, art_pop, band, exptime, mu_sky, zpt=27.0, 
                phase='all', downsample=1, **kwargs):

        if type(exptime) != u.Quantity:
            logger.warning('Will assume exposure time is in seconds.')
            logger.warning('Use astropy.units to specify units.')
            exptime *= u.second

        self.art_pop = art_pop
        self.xy_image_dim = art_pop.xy_image_dim

        counts = self._mag_to_counts(band, exptime, phase)
        src_counts = self._inject_into_image(counts, band, phase=phase)

        if downsample > 1:
            src_counts = block_reduce(src_counts, downsample)

        sky_counts = self.sb_to_counts(mu_sky, band, exptime, downsample)
        raw_counts = self.rng.poisson(src_counts + sky_counts)
        raw_counts = raw_counts + self.rng.normal(scale=self.read_noise, 
                                                  size=src_counts.shape)
        cali_factor = self.cali_factor(band, exptime, zpt)
        flux_cali = (raw_counts - sky_counts) * cali_factor
        noise = np.sqrt(raw_counts + self.read_noise**2)
        
        results = ObsData(raw_counts=raw_counts,
                          src_counts=src_counts,
                          sky_counts=sky_counts, 
                          flux_cali_img=flux_cali,
                          noise=noise,
                          cali_factor=cali_factor,
                          zpt=zpt, band=band, 
                          exptime=exptime)
        return results
