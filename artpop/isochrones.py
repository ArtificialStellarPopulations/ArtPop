import os
import numpy as np
from scipy.special import logsumexp
from scipy.interpolate import interp1d
from .read_mist_models import IsoCmdReader, IsoReader
from .imf import kroupa, salpeter, scalo
from .filter_info import *
from .log import logger
from . import MIST_PATH 
_imf_funcs = dict(salpeter=salpeter, kroupa=kroupa, scalo=scalo)


__all__ = ['phot_dict', 
           'fetch_mist_iso_cmd', 
           'fetch_mist_iso', 
           'MistIsochrone']
        

phot_dict = dict(SDSS='SDSSugriz', HST_ACS='HST_ACSWF', HST_WFC3='HST_WFC3', 
                 CFHT='CFHTugriz', HSC='HSC', DECam='DECam', 
                 Bessell='UBVRIplus', WFIRST='WFIRST', LSST='LSST', 
                 UKIDSS='UKIDSS')


def fetch_mist_iso_cmd(log_age, feh, phot_system, mist_path=MIST_PATH):
    """
    Fetch MIST isochrone. 
    
    Parameters
    ----------
    log_age : float
        log10(age) with age in Gyr
    feh : float
        [Fe/H] 
    phot_system : str
        Photometric system. 
    mist_path : str
        Path to MIST isochrones.
        
    Returns
    -------
    iso_cmd : structured ndarry
    
    Notes
    -----
    Currently, this only supports discrete ages and [Fe/H] values, 
    which are set in the MIST grids. TODO: Add an interpolation routine.  
    """

    path = os.path.join(
        mist_path, 'MIST_v1.2_vvcrit0.4_' + phot_dict[phot_system])
    sign = 'm' if feh < 0 else 'p'
    fn = 'MIST_v1.2_feh_{}{:.2f}_afe_p0.0_vvcrit0.4_{}.iso.cmd'
    fn = os.path.join(path, fn.format(sign, abs(feh), phot_dict[phot_system]))
    iso_cmd = IsoCmdReader(fn, verbose=False)
    iso_cmd = iso_cmd.isocmds[iso_cmd.age_index(log_age)]
    return iso_cmd        


def fetch_mist_iso(log_age, feh, mist_path=MIST_PATH):
    path = os.path.join(
        mist_path, 'MIST_v1.2_vvcrit0.4_basic_isos')
    sign = 'm' if feh < 0 else 'p'
    fn = 'MIST_v1.2_feh_{}{:.2f}_afe_p0.0_vvcrit0.4_basic.iso'
    fn = os.path.join(path, fn.format(sign, abs(feh)))
    iso = IsoReader(fn, verbose=False)
    iso = iso.isos[iso.age_index(log_age)]
    return iso


class MistIsochrone(object):
    """
    Helper class for calculating MIST stellar population properties. Loading 
    the mist data is a bit slow, so this class is useful if you want to 
    calculate the same quantities for various bands and/or imfs. 
    """

    phase_dict = dict(
        PMS=-1,
        MS=0,
        RGB=2,
        CHeB=3,
        EAGB=4,
        TPAGB=5,
        postAGB=6,
        WR=9
    )

    _phases = ['MS', 'giants', 'RGB', 'CHeB', 'AGB',
               'EAGB', 'TPAGB', 'postAGB', 'WDCS']

    phase_name = dict(map(reversed, phase_dict.items()))
    log_age_grid = np.arange(5.0, 10.3, 0.05)

    # actually have feh = -4, but using 3 for interpolation boundary
    feh_grid = np.concatenate([np.arange(-3.0, -2., 0.5), 
                               np.arange(-2.0, 0.75, 0.25)])
    
    def __init__(self, log_age, feh, phot_system, mist_path=MIST_PATH):
                 

        age_diff = np.abs(self.log_age_grid - log_age)
        self.log_age = self.log_age_grid[age_diff.argmin()]
        if age_diff.min() > 1e-6:
            logger.debug('Using nearest log_age = {:.2f}'.format(self.log_age))
        self.feh = feh
        self.phot_system = phot_system
        self.mist_path = mist_path
        if feh in self.feh_grid:
            self.iso = fetch_mist_iso_cmd(log_age, feh, phot_system, mist_path)
        else:
            self.iso = self._interp_on_feh()

        self.ieep = interp1d(self.iso['initial_mass'], self.iso['EEP'])
        self.mass_min = self.iso['initial_mass'].min()
        self.mass_max = self.iso['initial_mass'].max()
        self.filters = filter_dict[phot_system]

    def _interp_on_feh(self):
        assert self.feh_grid.min() < self.feh < self.feh_grid.max()

        i_feh = self.feh_grid.searchsorted(self.feh)
        feh_lo, feh_hi = self.feh_grid[i_feh - 1: i_feh + 1]

        logger.debug('Interpolating to [Fe/H] = {:.2f} '\
                     'using [Fe/H] = {} and {}'.\
                     format(self.feh, feh_lo, feh_hi))

        mist_0 = fetch_mist_iso_cmd(
            self.log_age, feh_lo, self.phot_system, self.mist_path)
        mist_1 = fetch_mist_iso_cmd(
            self.log_age, feh_hi, self.phot_system, self.mist_path)

        y0, y1 = np.array(mist_0.tolist()), np.array(mist_1.tolist())

        x = self.feh
        x0, x1 = feh_lo, feh_hi
        weight = (x - x0) / (x1 - x0)

        len_0, len_1 = len(y0), len(y1)
        
        # if necessary, extrapolate using trend of the longer array
        if (len_0 < len_1):
            delta = y1[len_0:] - y1[len_0 - 1]
            y0 = np.append(y0, y0[-1] + delta, axis=0)
        elif (len_0 > len_1):
            delta = y0[len_1:] - y0[len_1 - 1]
            y1 = np.append(y1, y1[-1] + delta, axis=0)

        y = y0 * (1 - weight) + y1 * weight
        iso = np.core.records.fromarrays(y.transpose(), dtype=mist_0.dtype)

        return iso

    def imf_weights(self, imf, num_mass_bins, imf_kw={}, phase=None, 
                    min_mass=None, max_mass=None):
        bin_edges = np.linspace(np.log(self.mass_min), 
                                np.log(self.mass_max), 
                                num_mass_bins)
        bin_edges = np.exp(bin_edges)
        masses = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        dm = np.diff(bin_edges)
        xi = _imf_funcs[imf](masses, **imf_kw)
        log_dm = np.log(dm)
        log_xi = np.log(xi)

        if phase or min_mass or max_mass:
            log_dm, log_xi, masses = self._check_mask(
                log_dm, log_xi, masses, phase, min_mass, max_mass)

        return log_dm, log_xi, masses

    def _check_mask(self, log_dm, log_xi, masses, phase, min_mass, max_mass):
        if phase is not None:
            phase_mask = self._get_phase_mask(phase, masses)
            log_dm = log_dm[phase_mask]
            log_xi = log_xi[phase_mask]
            masses = masses[phase_mask]
        elif (min_mass is not None) or (max_mass is not None):
            mass_mask = self._get_mass_mask(min_mass, max_mass, masses)
            log_dm = log_dm[mass_mask]
            log_xi = log_xi[mass_mask]
            masses = masses[mass_mask]
        return log_dm, log_xi, masses

    def _get_mass_mask(self, min_mass, max_mass, masses, **kwargs):
        if min_mass:
            mass_mask = masses > min_mass
        else:
            mass_mask = masses < max_mass
        return mass_mask

    def _get_phase_mask(self, phase, masses, **kwargs):
        """
        From Table II: Primary Equivalent Evolutionary Points (EEPs)
        http://waps.cfa.harvard.edu/MIST/README_tables.pdf
        """
        eep = self.ieep(masses)        

        if phase == 'all':
            mask = np.ones_like(masses, dtype=bool)
        elif phase == 'MS':
            mask = (eep >= 202) & (eep < 454)
        elif phase == 'giants':
            mask = (eep >= 454) & (eep < 1409)
        elif phase == 'RGB':
            mask = (eep >= 454) & (eep < 605)
        elif phase == 'CHeB':
            mask = (eep >= 605) & (eep < 707)
        elif phase == 'AGB':
            mask = (eep >= 707) & (eep < 1409)
        elif phase == 'EAGB':
            mask = (eep >= 707) & (eep < 808)
        elif phase == 'TPAGB':
            mask = (eep >= 808) & (eep < 1409)
        elif phase == 'postAGB':
            mask = (eep >= 1409) & (eep <= 1710)
        elif phase == 'WDCS':
            mask = eep > 1710
        else:
            raise Exception('Uh, what phase u want?')

        return mask

    def sbf_mag(self, band, imf='kroupa', num_mass_bins=1e5, phase=None, 
                imf_kw={}, return_lums=False, min_mass=None, max_mass=None):

        mag = interp1d(self.iso['initial_mass'], self.iso[band])

        log_dm, log_xi, masses = self.imf_weights(imf, num_mass_bins, imf_kw)
        log_lum = np.log(10**(-0.4*mag(masses)))
        lum = np.exp(logsumexp(log_xi + log_dm + log_lum))

        log_dm, log_xi, masses = self.imf_weights(imf, num_mass_bins, imf_kw, 
                                              phase, min_mass, max_mass)

        log_lum = np.log(10**(-0.4*mag(masses)))
        lum_weighted_lum = np.exp(logsumexp(log_xi + log_dm + 2 * log_lum))
        mbar = -2.5 * np.log10(lum_weighted_lum / lum)

        return (mbar, lum, lum_weighted_lum) if return_lums else mbar

    def integrated_color(self, blue_band, red_band, imf='kroupa', 
                         num_mass_bins=1e5, phase=None, imf_kw={}, 
                         return_lums=False, min_mass=None, max_mass=None):

        log_dm, log_xi, masses = self.imf_weights(imf, num_mass_bins, imf_kw, 
                                              phase, min_mass, max_mass)

        mag_blue = interp1d(self.iso['initial_mass'], self.iso[blue_band])
        mag_red = interp1d(self.iso['initial_mass'], self.iso[red_band])

        log_lum = np.log(10**(-0.4*mag_blue(masses)))
        lum_blue = np.exp(logsumexp(log_xi + log_dm + log_lum))
        mag_blue_tot = -2.5 * np.log10(lum_blue)

        log_lum = np.log(10**(-0.4*mag_red(masses)))
        lum_red = np.exp(logsumexp(log_xi + log_dm + log_lum))
        mag_red_tot = -2.5 * np.log10(lum_red)
        integrated_color = mag_blue_tot - mag_red_tot

        return (integrated_color, lum_blue, lum_red)\
                if return_lums else integrated_color

    def mean_mag(self, band, imf='kroupa', num_mass_bins=1e5, 
                 lum_weighted=False, phase=None, imf_kw={}, 
                 min_mass=None, max_mass=None):

        mag = interp1d(self.iso['initial_mass'], self.iso[band])

        log_dm, log_xi, masses = self.imf_weights(imf, num_mass_bins, imf_kw, 
                                                  phase, min_mass, max_mass)
        
        log_lum = np.log(10**(-0.4*mag(masses)))
        power_lum = 2.0 if lum_weighted else 1.0
        lum = np.exp(logsumexp(log_xi + log_dm + power_lum * log_lum))
        norm = np.exp(logsumexp(log_xi + log_dm))
        mean_mag = -2.5 * np.log10(lum/norm)

        return mean_mag
