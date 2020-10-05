import os, abc, six
import numpy as np
from scipy.interpolate import interp1d
from .imf import sample_imf, build_galaxy, imf_dict    
from .isochrones import MistIsochrone, phot_dict
from .utils import check_random_state
from .log import logger
from . import MIST_PATH
from .filter_info import *

__all__ = ['SSP', 'MultiSSP']


@six.add_metaclass(abc.ABCMeta)
class SPBase(object):

    _phases = ['MS', 'giants', 'RGB', 'CHeB', 'AGB',
               'EAGB', 'TPAGB', 'postAGB', 'WDCS']

    def __init__(self, phot_system, imf='kroupa', mist_path=MIST_PATH):
        self.imf = imf
        self.filters = filter_dict[phot_system]
        self.mist_path = mist_path
        self.phot_system = phot_system
        self.pop_exist = False

    def star_mags(self, band):
        return NotImplementedError()

    def build_pop(self, num_stars=None, **kwargs):
        return NotImplementedError()

    def _check_pop(self):
        if not self.pop_exist:
            logger.error('You still need to build a stellar pop!')
            return False
        else:
            return True

    def get_phase_mask(self, phase, **kwargs):
        """
        From Table II: Primary Equivalent Evolutionary Points (EEPs)
        http://waps.cfa.harvard.edu/MIST/README_tables.pdf
        """
        if not self._check_pop(): return None

        if phase == 'all':
            mask = np.ones_like(self.eep, dtype=bool)
        elif phase == 'MS':
            mask = (self.eep >= 202) & (self.eep < 454)
        elif phase == 'giants':
            mask = (self.eep >= 454) & (self.eep < 1409)
        elif phase == 'RGB':
            mask = (self.eep >= 454) & (self.eep < 605)
        elif phase == 'CHeB':
            mask = (self.eep >= 605) & (self.eep < 707)
        elif phase == 'AGB':
            mask = (self.eep >= 707) & (self.eep < 1409)
        elif phase == 'EAGB':
            mask = (self.eep >= 707) & (self.eep < 808)
        elif phase == 'TPAGB':
            mask = (self.eep >= 808) & (self.eep < 1409)
        elif phase == 'postAGB':
            mask = (self.eep >= 1409) & (self.eep <= 1710)
        elif phase == 'WDCS':
            mask = self.eep > 1710
        else:
            raise Exception('Uh, what phase u want?')

        return mask

    def sbf_mag(self, band):
        if not self._check_pop(): return None
        f_i = 10**(-0.4 * (self.star_mags(band)))
        mbar = -2.5 * np.log10(np.sum(f_i**2) / np.sum(f_i))
        return mbar

    def integrated_color(self, blue_band, red_band, phase='all'):
        if not self._check_pop(): return None
        mask = self.get_phase_mask(phase)
        if mask.sum() > 0:
            blue_mag = self.star_mags(blue_band)[mask]
            F_blue = 10**(-0.4 * blue_mag)
            red_mag = self.star_mags(red_band)[mask]
            F_red = 10**(-0.4 * red_mag)
            color = -2.5 * np.log10(F_blue.sum()/F_red.sum())
        else:
            color = np.nan
        return color

    def total_mag(self, band, phase='all'):
        if not self._check_pop(): return None
        mask = self.get_phase_mask(phase)
        mags = self.star_mags(band)[mask]
        total_flux = (10**(-0.4*mags)).sum()
        mag = -2.5 * np.log10(total_flux)
        return mag


class SSP(SPBase):
    """
    Simple Stellar Population
    """

    def __init__(self, log_age, feh, phot_system, total_mass=None, 
                 num_stars=None, imf='kroupa', mist_path=MIST_PATH, 
                 imf_kw={}, mist=None, random_state=None):

        super(SSP, self).__init__(phot_system, 
                                  imf=imf, 
                                  mist_path=mist_path)
        self.imf_kw = imf_kw
        self.log_age = log_age
        self.feh = feh
        if mist is None:
            self.mist = MistIsochrone(log_age, feh, phot_system, mist_path) 
        else:
            self.mist = mist
        self.mass_min = self.mist.mass_min
        self.mass_max = self.mist.mass_max
        self.rng = check_random_state(random_state)

        if total_mass or num_stars:
            self.build_pop(total_mass, num_stars)

    def star_mags(self, band, phase='all'): 
        iso_mass = self.mist.iso['initial_mass']
        mag = self.mist.iso[band]
        if phase == 'all':
            masses = self.star_masses
        else:
            masses = self.star_masses[self.get_phase_mask(phase)]
        mag = interp1d(iso_mass, mag)(masses)
        return mag

    def sbf_mag(self, band):
        if not self._check_pop(): return None
        f_i = 10**(-0.4 * (self.star_mags(band)))
        mbar = -2.5 * np.log10(np.sum(f_i**2) / np.sum(f_i))
        return mbar

    def build_pop(self, total_mass=None, num_stars=None):

        m_min, m_max = self.mass_min, self.mass_max
        imf_kw = self.imf_kw.copy()

        if total_mass is not None:
            self.star_masses = build_galaxy(
                total_mass, m_min=m_min, m_max=m_max, imf=self.imf, 
                random_state=self.rng, **imf_kw)
            self.num_stars = len(self.star_masses)
        elif num_stars is not None:
            imf_kw['norm_mass_min'] = self.mist.mass_min
            self.star_masses = sample_imf(
                int(num_stars), m_min=m_min, m_max=m_max, imf=self.imf, 
                random_state=self.rng, imf_kw=imf_kw)
            self.num_stars = int(num_stars)
        else:
            raise Exception('must give total mass *or* number of stars')
        self.total_mass = self.star_masses.sum()

        iso_mass = self.mist.iso['initial_mass']
        self.eep = interp1d(
            iso_mass, self.mist.iso['EEP'])(self.star_masses)
        self.log_L = interp1d(
            iso_mass, self.mist.iso['log_L'])(self.star_masses)
        self.log_Teff = interp1d(
            iso_mass, self.mist.iso['log_Teff'])(self.star_masses) 
        self.pop_exist = True


class MultiSSP(SPBase):
    """
    """

    def __init__(self, log_age_list, feh_list, mass_frac_list, 
                 total_mass, phot_system, mist_path=MIST_PATH, 
                 imf='kroupa', random_state=None, **kwargs):

        super(MultiSSP, self).__init__(phot_system, 
                                       imf=imf, 
                                       mist_path=mist_path)
                                      

        assert np.sum(mass_frac_list) == 1.0, 'mass fractions must add to 1!'
        
        log_age_list = np.asarray(log_age_list)
        feh_list = np.asarray(feh_list)
        mass_frac_list = np.asarray(mass_frac_list)

        idx = log_age_list.argsort()
        log_age_list = log_age_list[idx]
        feh_list = feh_list[idx]
        mass_frac_list = mass_frac_list[idx]
        
        ssp_list = []    
        for log_age, feh, frac in zip(log_age_list, feh_list, mass_frac_list):
            ssp_list.append(
                SSP(log_age, feh, phot_system, frac * total_mass, **kwargs)
            )
        
        self.iso = np.concatenate([_s.mist.iso for _s in ssp_list])
        self.mass_min = self.iso['initial_mass'].min()
        self.mass_max = self.iso['initial_mass'].max()

        self.total_mass = np.sum([_s.total_mass for _s in ssp_list])
        self.eep = np.concatenate([_s.eep for _s in ssp_list])
        self.log_L = np.concatenate([_s.log_L for _s in ssp_list])
        self.log_Teff = np.concatenate([_s.log_Teff for _s in ssp_list])
        self.star_masses = np.concatenate([_s.star_masses for _s in ssp_list])
        self.num_stars = len(self.star_masses)
        self.num_stars_per_ssp = [_s.num_stars for _s in ssp_list]

        for band in ssp_list[-1].filters:
            mag_list = [_s.star_mags(band) for _s in ssp_list]
            setattr(self, band, np.concatenate(mag_list))

        self.log_age = log_age_list
        self.feh = feh_list
        self.mass_frac = mass_frac_list
        self.pop_exist = True

    def star_mags(self, band, phase='all'): 
        if phase == 'all':
            mag = getattr(self, band)
        else:
            mag = getattr(self, band)[self.get_phase_mask(phase)]
        return mag
