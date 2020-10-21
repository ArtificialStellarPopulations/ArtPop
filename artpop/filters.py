import os
import pickle
from glob import glob
import numpy as np
from astropy import units as u
from astropy.table import Table
from . import data_dir
filter_dir = os.path.join(data_dir, 'filters')


__all__ = ['phot_system_list', 
           'get_filter_names',
           'phot_system_lookup',
           'FilterSystem',
           'get_mist_zero_point_converter']


phot_system_list = [
    'HST_WFC3', 'HST_ACSWF', 'SDSSugriz', 'CFHTugriz', 'DECam', 'HSC', 
    'HST_ACSWF', 'JWST', 'LSST', 'UBVRIplus', 'UKIDSS', 'WFIRST'
]


def get_filter_names(phot_system=None):
    fn = os.path.join(data_dir, 'filters/mist_filter_names.pkl')
    pickle_in = open(fn, 'rb')
    mist_filter_names = pickle.load(pickle_in)
    pickle_in.close()
    if phot_system is not None:
        mist_filter_names = mist_filter_names[phot_system]
    return mist_filter_names


def phot_system_lookup(filt=None):
    fn = os.path.join(data_dir, 'filters/phot_system_lookup.pkl')
    pickle_in = open(fn, 'rb')
    lookup = pickle.load(pickle_in)
    pickle_in.close()
    if filt is not None:
        lookup = lookup[filt]
    return lookup


class FilterSystem(object):
    """
    Reference: Fukugita et al. (1996) AJ 111, 1748.
    """
    def __init__(self, phot_system, filter_dir=filter_dir, **kwargs):
        if type(phot_system) == str:
            phot_system = [phot_system]
        self.phot_system = phot_system
        self.filter_dir = filter_dir 
        for p in phot_system:
            phot_system_path = os.path.join(filter_dir, p)
            files = glob(os.path.join(phot_system_path, '*.csv'))
            for fn in files:
                name = os.path.basename(fn)[:-4]
                setattr(self, name, Table.read(fn))

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


class ZeroPointConverter(object):

    def __init__(self, table):
        for f, system, v_to_st, v_to_ab in table:
            setattr(self, f, [system, v_to_st, v_to_ab])
        self.zpt_table = table
    
    def to_vega(self, filt):
        system, v_to_st, v_to_ab = getattr(self, filt)
        if system == 'Vega':
            zpt_convert = 0.
        elif system == 'AB':
            zpt_convert = -v_to_ab
        return zpt_convert

    def to_ab(self, filt):
        system, v_to_st, v_to_ab = getattr(self, filt)
        if system == 'AB':
            zpt_convert = 0.
        elif system == 'Vega':
            zpt_convert = v_to_ab
        return zpt_convert

    def to_st(self, filt):
        system, v_to_st, v_to_ab = getattr(self, filt)
        if system == 'AB':
            zpt_convert = v_to_st - v_to_ab
        elif system == 'Vega':
            zpt_convert = v_to_st
        return zpt_convert

    def color_to_vega(self, blue, red):
        blue_convert = self.to_vega(blue)
        red_convert = self.to_vega(red)
        return blue_convert - red_convert

    def color_to_ab(self, blue, red):
        blue_convert = self.to_ab(blue)
        red_convert = self.to_ab(red)
        return blue_convert - red_convert

    def color_to_st(self, blue, red):
        blue_convert = self.to_st(blue)
        red_convert = self.to_st(red)
        return blue_convert - red_convert


def get_mist_zero_point_converter():
    from astropy.io import ascii
    from . import data_dir
    fn = os.path.join(package_dir, 'filters', 'zeropoints.txt')
    table= ascii.read(fn)
    return ZeroPointConverter(table)
