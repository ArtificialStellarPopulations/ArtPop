import os, abc
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy import units as u
from ..util import check_random_state, check_units
from ..log import logger
from .. import MIST_PATH
from ..filters import *
from .imf import sample_imf, build_galaxy, imf_dict    
from .isochrones import MistIsochrone


__all__ = ['SSP']


class StellarPopulation(metaclass=abc.ABCMeta):

    _phases = ['MS', 'giants', 'RGB', 'CHeB', 'AGB',
               'EAGB', 'TPAGB', 'postAGB', 'WDCS']

    def __init__(self, phot_system, distance, imf='kroupa', 
                 mist_path=MIST_PATH):
        self.imf = imf
        self.distance = check_units(distance, 'Mpc')
        self.mist_path = mist_path
        self.phot_system = phot_system
        self.pop_exist = False

    def _check_pop(self):
        if not self.pop_exist:
            logger.error('You still need to build a stellar pop!')
            return False
        else:
            return True

    def build_pop(self, num_stars=None, **kwargs):
        return NotImplementedError()

    @property
    def dist_mod(self):
        return 5 * np.log10(self.distance.to('pc').value) - 5

    @property
    def num_pops(self):
        return len(self.log_age)

    @property
    def total_mass(self):
        return self.star_masses.sum()

    @property
    def num_stars(self):
        return len(self.star_masses)

    @property 
    def abs_mag_table(self):
        return Table(self.abs_mags)

    @property 
    def mag_table(self):
        _mags = {}
        for filt in self.filters:
            _mags[filt] = self.star_mags(filt)
        return Table(_mags)

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

    def set_distance(self, distance):
        self.distance = check_units(distance, 'Mpc')

    def star_mags(self, band, phase='all'): 
        mags = self.abs_mags[band] + self.dist_mod
        if phase != 'all':
            mags = mags[self.get_phase_mask(phase)]
        return mags

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
        if mask.sum() > 0:
            mags = self.star_mags(band)[mask]
            total_flux = (10**(-0.4*mags)).sum()
            mag = -2.5 * np.log10(total_flux)
        else:
            mag = np.nan
        return mag

    def __add__(self, pop):
        assert StellarPopulation in pop.__class__.__mro__
        assert self.filters == pop.filters, 'must have same filters'
        assert self.distance == pop.distance, 'pops must have same distance'
        new = deepcopy(self)
        new.iso = np.concatenate([new.iso, pop.iso])
        new.mass_min = new.iso['initial_mass'].min()
        new.mass_max = new.iso['initial_mass'].max()
        new.eep = np.concatenate([new.eep, pop.eep])
        new.log_L = np.concatenate([new.log_L, pop.log_L])
        new.log_Teff = np.concatenate([new.log_L, pop.log_Teff])
        new.star_masses = np.concatenate([new.star_masses, pop.star_masses])
        if type(new.log_age) != list:
            new.log_age = [new.log_age]
        new.log_age.append(pop.log_age)
        if type(new.feh) != list:
            new.feh = [new.feh]
        new.feh.append(pop.feh)
        new_label = np.ones(len(pop.star_masses), dtype=int) * len(new.feh)
        new.labels = np.concatenate([new.labels, new_label])
        labels = np.unique(new.labels)
        pop_fracs = [(l==new.labels).sum() / new.num_stars for l in labels]
        new.log_age_mean = np.average(new.log_age, weights=pop_fracs)
        new.feh_mean  = np.average(new.feh, weights=pop_fracs)
        new.pop_fracs = pop_fracs
        for filt in new.filters:
            _mags = [new.abs_mags[filt], pop.abs_mags[filt]]
            new.abs_mags[filt] = np.concatenate(_mags)
        return CompositePopulation(new)


class SSP(StellarPopulation):
    """
    Simple Stellar Population
    """

    def __init__(self, log_age, feh, phot_system, total_mass=None, 
                 num_stars=None, distance=10*u.pc, imf='kroupa', 
                 mist_path=MIST_PATH, imf_kw={}, random_state=None):

        super(SSP, self).__init__(phot_system, 
                                  distance=distance,
                                  imf=imf, 
                                  mist_path=mist_path)
        self.imf_kw = imf_kw
        self.log_age = log_age
        self.feh = feh
        mist = MistIsochrone(log_age, feh, phot_system, mist_path) 
        self.iso = mist.iso
        self.filters = mist.filters
        self.mass_min = self.iso['initial_mass'].min()
        self.mass_max = self.iso['initial_mass'].max()
        self.rng = check_random_state(random_state)
        self.build_pop(total_mass, num_stars)

    def build_pop(self, total_mass=None, num_stars=None):
        m_min, m_max = self.mass_min, self.mass_max
        imf_kw = self.imf_kw.copy()

        if total_mass is not None:
            self.star_masses = build_galaxy(
                total_mass, m_min=m_min, m_max=m_max, imf=self.imf, 
                random_state=self.rng, **imf_kw)
        elif num_stars is not None:
            imf_kw['norm_mass_min'] = self.mass_min
            self.star_masses = sample_imf(
                int(num_stars), m_min=m_min, m_max=m_max, imf=self.imf, 
                random_state=self.rng, imf_kw=imf_kw)
        else:
            raise Exception('you must give total mass *or* number of stars')

        iso_mass = self.iso['initial_mass']
        self.eep = interp1d(
            iso_mass, self.iso['EEP'])(self.star_masses)
        self.log_L = interp1d(
            iso_mass, self.iso['log_L'])(self.star_masses)
        self.log_Teff = interp1d(
            iso_mass, self.iso['log_Teff'])(self.star_masses) 
        self.labels = np.ones(len(self.eep), dtype=int)
        self.abs_mags = {}
        for filt in self.filters:
            self.abs_mags[filt] = interp1d(
                iso_mass, self.iso[filt])(self.star_masses)
        self.pop_exist = True

    def __repr__(self):
        r = dict(total_mass=f'{self.total_mass:.2e}', 
                 log_age=self.log_age,
                 feh=self.feh,
                 phot_system=self.phot_system)
        r = [f'{k} = {v}' for k, v in r.items()]
        return '\n'.join(r)

    def __repr__(self):
        r = {'M_star': f'{self.total_mass:.2e} M_sun', 
             'log(age/yr)': self.log_age,
             '[Fe/H]': self.feh,
             'photometric system': self.phot_system}
        r = [f'{k} = {v}' for k, v in r.items()]
        t = 'Simple Stellar Population\n-------------------------\n'
        return t + '\n'.join(r)


class CompositePopulation(StellarPopulation):

    def __init__(self, pop):

        super(CompositePopulation, self).__init__(phot_system=pop.phot_system, 
                                                  distance=pop.distance,
                                                  imf=pop.imf, 
                                                  mist_path=pop.mist_path)
        self.star_masses = pop.star_masses
        self.log_age = pop.log_age
        self.feh = pop.feh
        self.iso = pop.iso
        self.imf_kw = pop.imf_kw
        self.filters = pop.filters
        self.mass_min = pop.mass_min
        self.mass_max = pop.mass_max
        self.rng = pop.rng
        self.labels = pop.labels
        self.pop_fracs = pop.pop_fracs
        self.log_age_mean = pop.log_age_mean
        self.feh_mean  = pop.feh_mean
        self.abs_mags = pop.abs_mags
        self.eep = pop.eep
        self.log_L = pop.log_L
        self.log_Teff = pop.log_Teff
        self.pop_exist = True

    def __repr__(self):
        r = {'N_pops': self.num_pops,
             'M_star': f'{self.total_mass:.2e} M_sun', 
             'log(age/yr)': self.log_age,
             '[Fe/H]': self.feh,
             'pop fractions': [f'{p * 100:.2f}%' for p in self.pop_fracs],
             'photometric system': self.phot_system}
        r = [f'{k} = {v}' for k, v in r.items()]
        t = 'Composite Population\n--------------------\n'
        return t + '\n'.join(r)
