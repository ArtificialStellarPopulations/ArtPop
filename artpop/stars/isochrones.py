import os
import numpy as np
from numpy.lib.recfunctions import append_fields
from ._read_mist_models import IsoCmdReader, IsoReader
from ..log import logger
from ..image.filter_info import *
from .. import MIST_PATH 


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
    Class for fetching MIST isochrones.
    """

    log_age_grid = np.arange(5.0, 10.3, 0.05)
    log_age_min = log_age_grid.min()
    log_age_max = log_age_grid.max()

    # actually have feh = -4, but using 3 for interpolation boundary
    feh_grid = np.concatenate([np.arange(-3.0, -2., 0.5), 
                               np.arange(-2.0, 0.75, 0.25)])
    feh_min = feh_grid.min()
    feh_max = feh_grid.max()
    
    def __init__(self, log_age, feh, phot_system, mist_path=MIST_PATH):

        if log_age < self.log_age_min or log_age > self.log_age_max:
            raise Exception(f'log_age = {log_age} not in range of age grid')
        if feh < self.feh_min or feh > self.feh_max:
            raise Exception(f'feh = {feh} not in range of feh grid')
        
        age_diff = np.abs(self.log_age_grid - log_age)
        self.log_age = self.log_age_grid[age_diff.argmin()]
        if age_diff.min() > 1e-6:
            logger.debug('Using nearest log_age = {:.2f}'.format(self.log_age))
        self.feh = feh
        self.mist_path = mist_path
        self.phot_system = phot_system
        if type(phot_system) != list:
            phot_system = [phot_system]
        self.iso = self._fetch_iso(phot_system[0])
        self.filters = filter_dict[phot_system[0]].copy()
        for p in phot_system[1:]:
            filt = filter_dict[p].copy()
            self.filters.extend(filt)
            _iso = self._fetch_iso(p)
            mags = [_iso[f].data for f in filt]
            self.iso = append_fields(self.iso, filt, mags)
        self.mass_min = self.iso['initial_mass'].min()
        self.mass_max = self.iso['initial_mass'].max()

    def _fetch_iso(self, phot_system):
        if self.feh in self.feh_grid:
            iso = fetch_mist_iso_cmd(self.log_age, self.feh, 
                                     phot_system, self.mist_path)
        else:
            iso = self._interp_on_feh(phot_system)
        return iso

    def _interp_on_feh(self, phot_system):
        i_feh = self.feh_grid.searchsorted(self.feh)
        feh_lo, feh_hi = self.feh_grid[i_feh - 1: i_feh + 1]

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
