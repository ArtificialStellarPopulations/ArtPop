# Standard library 
import os

# Third-party 
import numpy as np
from numpy.lib.recfunctions import append_fields

# Project
from ._read_mist_models import IsoCmdReader, IsoReader
from ..log import logger
from ..filters import phot_system_list, get_filter_names
from .. import MIST_PATH 
phot_str_helper = {p.lower():p for p in phot_system_list}


__all__ = ['fetch_mist_iso_cmd', 'MistIsochrone']


def fetch_mist_iso_cmd(log_age, feh, phot_system, mist_path=MIST_PATH):
    """
    Fetch MIST isochrone grid. 
    
    Parameters
    ----------
    log_age : float
        Logarithm base 10 of the simple stellar population age in years.
    feh : float
        Metallicity [Fe/H] of the simple stellar population.
    phot_system : str 
        Name of the photometric system.
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different
        path from the `MIST_PATH` environment variable.
        
    Returns
    -------
    iso_cmd : `~numpy.ndarray`
        Structured ``numpy`` array with isochrones and stellar magnitudes. 
    """

    p = phot_system.lower()
    path = os.path.join(
        mist_path, 'MIST_v1.2_vvcrit0.4_' + phot_str_helper[p])
    sign = 'm' if feh < 0 else 'p'
    fn = 'MIST_v1.2_feh_{}{:.2f}_afe_p0.0_vvcrit0.4_{}.iso.cmd'
    fn = os.path.join(path, fn.format(sign, abs(feh), phot_str_helper[p]))
    iso_cmd = IsoCmdReader(fn, verbose=False)
    iso_cmd = iso_cmd.isocmds[iso_cmd.age_index(log_age)]
    return iso_cmd        


class MistIsochrone(object):
    """
    Class for fetching and storing MIST isochrones.

    .. note::
        Currently, the models are interpolated in metallicity but not in age. 
        Ages are therefore limited to the age grid of the MIST models. The 
        [Fe/H] and log(Age/yr) grids are stored as the private class attributes 
        `_feh_grid` and `_log_age_grid`.

    Parameters
    ----------
    log_age : float
        Logarithm base 10 of the simple stellar population age in years.
    feh : float
        Metallicity [Fe/H] of the simple stellar population.
    phot_system : str or list-like
        Name of the photometric system(s).
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different
        path from the `MIST_PATH` environment variable.
    """

    # the age grid
    _log_age_grid = np.arange(5.0, 10.3, 0.05)
    _log_age_min = _log_age_grid.min()
    _log_age_max = _log_age_grid.max()

    # the [Fe/H] metallicity grid
    # we have feh <= -4, but using <=3 for interpolation boundary
    _feh_grid = np.concatenate([np.arange(-3.0, -2., 0.5), 
                                np.arange(-2.0, 0.75, 0.25)])
    _feh_min = _feh_grid.min()
    _feh_max = _feh_grid.max()
    
    def __init__(self, log_age, feh, phot_system, mist_path=MIST_PATH):

        # verify age are metallicity are within model grids
        if log_age < self._log_age_min or log_age > self._log_age_max:
            raise Exception(f'log_age = {log_age} not in range of age grid')
        if feh < self._feh_min or feh > self._feh_max:
            raise Exception(f'feh = {feh} not in range of feh grid')

        self.feh = feh
        self.mist_path = mist_path
        self.phot_system = phot_system
        
        # use nearest age (currently not interpolating on age)
        age_diff = np.abs(self._log_age_grid - log_age)
        self.log_age = self._log_age_grid[age_diff.argmin()]
        if age_diff.min() > 1e-6:
            logger.debug('Using nearest log_age = {:.2f}'.format(self.log_age))

        # store phot_system as list to allow multiple photometric systems
        if type(phot_system) == str: 
            phot_system = [phot_system]

        # fetch first isochrone grid, interpolating on [Fe/H] if necessary 
        self._iso = self._fetch_iso(phot_system[0])

        # iterate over photometric systems and fetch remaining isochrones   
        filter_dict = get_filter_names()
        self._filters = filter_dict[phot_system[0]].copy()
        for p in phot_system[1:]:
            filt = filter_dict[p].copy()
            self._filters.extend(filt)
            _iso = self._fetch_iso(p)
            mags = [_iso[f].data for f in filt]
            self._iso = append_fields(self.iso, filt, mags)
        
        self._mass_min = self.iso['initial_mass'].min()
        self._mass_max = self.iso['initial_mass'].max()

    @property
    def iso(self):
        """MIST isochrone in a structured `~numpy.ndarray`."""
        return self._iso

    @property
    def filters(self):
        """List of filters in the given photometric system(s)."""
        return self._filters

    @property
    def mass_min(self):
        """The minimum mass of the isochrone."""
        return self._mass_min

    @property
    def mass_max(self):
        """The maximum mass of the isochrone."""
        return self._mass_max

    def _fetch_iso(self, phot_system):
        """Fetch MIST isochrone grid, interpolating on [Fe/H] if necessary."""
        if self.feh in self._feh_grid:
            iso = fetch_mist_iso_cmd(self.log_age, self.feh, 
                                     phot_system, self.mist_path)
        else:
            iso = self._interp_on_feh(phot_system)
        return iso

    def _interp_on_feh(self, phot_system):
        """Interpolate isochrones between two [Fe/H] grid points."""
        i_feh = self._feh_grid.searchsorted(self.feh)
        feh_lo, feh_hi = self._feh_grid[i_feh - 1: i_feh + 1]

        logger.debug('Interpolating to [Fe/H] = {:.2f} '\
                     'using [Fe/H] = {} and {}'.\
                     format(self.feh, feh_lo, feh_hi))

        mist_0 = fetch_mist_iso_cmd(
            self.log_age, feh_lo, phot_system, self.mist_path)
        mist_1 = fetch_mist_iso_cmd(
            self.log_age, feh_hi, phot_system, self.mist_path)

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
