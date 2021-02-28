# Standard library
import os, abc
import numpy as np
from copy import deepcopy

# Third-party
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy import units as u

# Project
from ..util import check_random_state, check_units, MIST_PATH
from ..log import logger
from ..filters import *
from .imf import sample_imf, build_galaxy, imf_dict
from .isochrones import MistIsochrone


__all__ = ['constant_sb_stars_per_pix', 'SSP']


def constant_sb_stars_per_pix(sb, mean_mag, distance=10*u.pc, pixel_scale=0.2):
    """
    Calculate the number of stars per pixel for a uniform
    distribution (i.e., constant surface brightness) of stars.

    Parameters
    ----------
    sb : float
        Surface brightness of stellar population.
    mean_mag : float
        Mean stellar magnitude of the stellar population.
    distance : float or `~astropy.units.Quantity`
        Distance to source. If float is given, the units are assumed
        to be `~astropy.units.Mpc`.
    pixel_scale : float or `~astropy.units.Quantity`, optional
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `~astropy.units.arcsec` per `~astropy.units.pixels`.

    Returns
    -------
    num_stars_per_pix : float
        The number of stars per pixel.
    """
    distance = check_units(distance, 'Mpc').to('pc').value
    pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel).value
    dist_mod = 5 * np.log10(distance) - 5
    num_stars_per_arsec_sq = 10**(0.4 * (mean_mag + dist_mod -  sb))
    num_stars_per_pix = num_stars_per_arsec_sq * pixel_scale**2
    return num_stars_per_pix


class StellarPopulation(metaclass=abc.ABCMeta):
    """
    Stellar population base class.

    Parameters
    ----------
    phot_system : str or list-like
        Name of the photometric system(s).
    distance : float or `~astropy.units.Quantity`, optional
        Distance to source. If float is given, the units are assumed
        to be `~astropy.units.Mpc`. Default distance is 10 `~astropy.units.pc`.
    imf : str, optional
        The initial stellar mass function. Default is `'kroupa'`.
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different
        path from the `MIST_PATH` environment variable.
    """

    phases = ['MS', 'giants', 'RGB', 'CHeB', 'AGB',
              'EAGB', 'TPAGB', 'postAGB', 'WDCS']

    def __init__(self, phot_system, distance=10.0 * u.pc, imf='kroupa',
                 mist_path=MIST_PATH):
        self.imf = imf
        self.distance = check_units(distance, 'Mpc')
        self.mist_path = mist_path
        self.phot_system = phot_system

    def build_pop(self, num_stars=None, **kwargs):
        """Build stellar population."""
        return NotImplementedError()

    @property
    def dist_mod(self):
        """The distance modulus."""
        return 5 * np.log10(self.distance.to('pc').value) - 5

    @property
    def num_pops(self):
        """
        The number of simple stellar populations that composes this pop.
        """
        return len(self.log_age)

    @property
    def total_mass(self):
        """Total stellar mass in solar units."""
        return self.star_masses.sum()

    @property
    def num_stars(self):
        """Number stars in population."""
        return len(self.star_masses)

    @property
    def abs_mag_table(self):
        """Absolute magnitudes in a `~astropy.table.Table` object."""
        return Table(self.abs_mags)

    @property
    def mag_table(self):
        """Apparent magnitudes in a `~astropy.table.Table` object."""
        _mags = {}
        for filt in self.filters:
            _mags[filt] = self.star_mags(filt)
        return Table(_mags)

    def get_phase_mask(self, phase):
        """
        Generate stellar evolutionary phase mask. The mask will be `True` for
        sources that are in the give phase according to the MIST EEPs.

        Parameters
        ----------
        phase : str
            Evolutionary phase to select. Options are 'all', 'MS', 'giants',
            'RGB', 'CHeB', 'AGB', 'EAGB', 'TPAGB', 'postAGB', or 'WDCS'.

        Returns
        -------
        mask : `~numpy.ndarray`
            Mask that is `True` for stars in input phase and `False` otherwise.

        Notes
        -----
        The MIST EEP phases were taken from Table II: Primary Equivalent
        Evolutionary Points (EEPs):
        http://waps.cfa.harvard.edu/MIST/README_tables.pdf
        """
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
        """
        Change the distance to the stellar population.

        Parameters
        ----------
        distance : float or `~astropy.units.Quantity`
            Distance to source. If float is given, the units are assumed
            to be `~astropy.units.Mpc`.
        """
        self.distance = check_units(distance, 'Mpc')

    def star_mags(self, bandpass, phase='all'):
        """
        Get the stellar apparent magnitudes.

        Parameters
        ----------
        bandpass : str
            Filter of observation. Must be a filter in the given
            photometric system(s).
        phase : str, optional
            Evolutionary phase to select. Options are 'all', 'MS', 'giants',
            'RGB', 'CHeB', 'AGB', 'EAGB', 'TPAGB', 'postAGB', or 'WDCS'.

        Returns
        -------
        mags : `~numpy.ndarray`
            The stellar apparent magnitudes in the given bandpass.
        """
        mags = self.abs_mags[bandpass] + self.dist_mod
        if phase != 'all':
            mags = mags[self.get_phase_mask(phase)]
        return mags

    def sbf_mag(self, bandpass):
        """
        Calculate the apparent SBF magnitude of the stellar population.

        Parameters
        ----------
        bandpass : str
            Filter of observation. Must be a filter in the given
            photometric system(s).

        Returns
        -------
        mbar : float
            The apparent SBF magnitude of the stellar population in the given
            bandpass.
        """
        f_i = 10**(-0.4 * (self.star_mags(bandpass)))
        mbar = -2.5 * np.log10(np.sum(f_i**2) / np.sum(f_i))
        return mbar

    def integrated_color(self, blue, red, phase='all'):
        """
        Calculate the population's integrated color.

        Parameters
        ----------
        blue : str
            The blue bandpass. Must be a filter in the
            given photometric system(s).
        red : str
            The red bandpass. Must be a filter in the
            given photometric system(s).
        phase : str, optional
            Evolutionary phase to select. Options are 'all', 'MS', 'giants',
            'RGB', 'CHeB', 'AGB', 'EAGB', 'TPAGB', 'postAGB', or 'WDCS'.

        Returns
        -------
        color : float
            The integrated color.
        """
        mask = self.get_phase_mask(phase)
        if mask.sum() > 0:
            blue_mag = self.star_mags(blue)[mask]
            F_blue = 10**(-0.4 * blue_mag)
            red_mag = self.star_mags(red)[mask]
            F_red = 10**(-0.4 * red_mag)
            color = -2.5 * np.log10(F_blue.sum()/F_red.sum())
        else:
            color = np.nan
        return color

    def mean_mag(self, bandpass, phase='all'):
        """
        Calculate the population's mean magnitude.

        Parameters
        ----------
        bandpass : str
            Filter of observation. Must be a filter in the given
            photometric system(s).
        phase : str
            Evolutionary phase to select. Options are 'all', 'MS', 'giants',
            'RGB', 'CHeB', 'AGB', 'EAGB', 'TPAGB', 'postAGB', or 'WDCS'.

        Returns
        -------
        mag : float
            The mean magnitude in the given bandpass.
        """
        mask = self.get_phase_mask(phase)
        if mask.sum() > 0:
            mags = self.star_mags(bandpass)[mask]
            mean_flux = (10**(-0.4*mags)).sum() / mask.sum()
            mag = -2.5 * np.log10(mean_flux)
        else:
            logger.warning(f'No stars in phase {phase}!')
            mag = np.nan
        return mag

    def total_mag(self, bandpass, phase='all'):
        """
        Calculate the population's total magnitude.

        Parameters
        ----------
        bandpass : str
            Filter of observation. Must be a filter in the given
            photometric system(s).
        phase : str
            Evolutionary phase to select. Options are 'all', 'MS', 'giants',
            'RGB', 'CHeB', 'AGB', 'EAGB', 'TPAGB', 'postAGB', or 'WDCS'.

        Returns
        -------
        mag : float
            The total magnitude in the given bandpass.
        """
        mask = self.get_phase_mask(phase)
        if mask.sum() > 0:
            mags = self.star_mags(bandpass)[mask]
            total_flux = (10**(-0.4*mags)).sum()
            mag = -2.5 * np.log10(total_flux)
        else:
            logger.warning(f'No stars in phase {phase}!')
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
        new.initial_masses = np.concatenate(
            [new.initial_masses, pop.initial_masses])
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
    Simple Stellar Population.

    Parameters
    ----------
    log_age : float
        Log (base 10) of the simple stellar population age in years.
    feh : float
        Metallicity [Fe/H] of the simple stellar population.
    phot_system : str or list-like
        Name of the photometric system(s).
    total_mass : float or `None`
        Stellar mass of the source. If `None`, then must give `num_stars`.
    num_stars : int or `None`
        Number of stars in source. If `None`, then must give `total_mass`.
    distance : float or `~astropy.units.Quantity`, optional
        Distance to source. If float is given, the units are assumed
        to be `~astropy.units.Mpc`. Default distance is 10 `~astropy.units.pc`.
    imf : str, optional
        The initial stellar mass function. Default is `'kroupa'`.
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different
        path from the `MIST_PATH` environment variable.
    imf_kw : dict, optional
        Optional keyword arguments for sampling the stellar mass function.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.

    Notes
    -----
    You must give `total_mass` *or* `num_stars`.
    """

    def __init__(self, log_age, feh, phot_system, total_mass=None,
                 num_stars=None, distance=10*u.pc, imf='kroupa',
                 mist_path=MIST_PATH, imf_kw={}, random_state=None,
                 **kwargs):

        super(SSP, self).__init__(phot_system,
                                  distance=distance,
                                  imf=imf,
                                  mist_path=mist_path)
        self.imf_kw = imf_kw
        self.log_age = log_age
        self.feh = feh
        mist = MistIsochrone(log_age, feh, phot_system, mist_path, **kwargs)
        self.iso = mist.iso
        self.filters = mist.filters
        self.mass_min = self.iso['initial_mass'].min()
        self.mass_max = self.iso['initial_mass'].max()
        self.rng = check_random_state(random_state)
        self.build_pop(total_mass, num_stars)

    def build_pop(self, total_mass=None, num_stars=None, mass_tolerance=0.05):
        """
        Build the stellar population. You must give `total_mass`
        *or* `num_stars` as an argument.

        Parameters
        ----------
        total_mass : float or `None`
            Stellar mass of the source in solar masses. If `None`, then must
            give `num_stars`.
        num_stars : int or `None`
            Number of stars in source. If `None`, then must give `total_mass`.

        Notes
        -----
        Running this method will set the following attributes: `star_masses`,
        `eep`, `log_L`, `log_Teff`, `labels`, and `abs_mags`.
        """
        m_min, m_max = self.mass_min, self.mass_max
        imf_kw = self.imf_kw.copy()
        iso_mass = self.iso['initial_mass']

        if num_stars is not None:
            imf_kw['norm_mass_min'] = self.mass_min
            self.initial_masses = sample_imf(
                int(num_stars), m_min=m_min, m_max=m_max, imf=self.imf,
                random_state=self.rng, imf_kw=imf_kw)
            self.star_masses = interp1d(
                iso_mass, self.iso['star_mass'])(self.initial_masses)
        elif total_mass is not None:
            self.star_masses = build_galaxy(
                total_mass, m_min=m_min, m_max=m_max, imf=self.imf,
                random_state=self.rng, **imf_kw)
        else:
            raise Exception('you must give total mass *or* number of stars')

        self.eep = interp1d(
            iso_mass, self.iso['EEP'])(self.initial_masses)
        self.log_L = interp1d(
            iso_mass, self.iso['log_L'])(self.initial_masses)
        self.log_Teff = interp1d(
            iso_mass, self.iso['log_Teff'])(self.initial_masses)
        self.labels = np.ones(len(self.eep), dtype=int)
        self.abs_mags = {}
        for filt in self.filters:
            self.abs_mags[filt] = interp1d(
                iso_mass, self.iso[filt])(self.initial_masses)

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
    """
    Composite stellar populations.
    """

    def __init__(self, pop):

        super(CompositePopulation, self).__init__(phot_system=pop.phot_system,
                                                  distance=pop.distance,
                                                  imf=pop.imf,
                                                  mist_path=pop.mist_path)
        self.initial_masses = pop.initial_masses
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
