# Standard library
import os
import pickle
from glob import glob

# Third-party
import numpy as np
from astropy import units as u
from astropy.table import Table

# Project
from .util import data_dir


__all__ = ['phot_system_list',
           'get_filter_names',
           'phot_system_lookup',
           'FilterSystem',
           'get_zero_point_converter']


# default filter curve directory
filter_dir = os.path.join(data_dir, 'filters')

# list of implemented MIST photometric systems
phot_system_list = [
    'HST_WFC3', 'HST_ACSWF', 'SDSSugriz', 'CFHTugriz', 'DECam', 'HSC',
    'HST_ACSWF', 'JWST', 'LSST', 'UBVRIplus', 'UKIDSS', 'WFIRST'
]


def get_filter_names(phot_system=None):
    """
    Get MIST photometric systems and filter names.

    Parameters
    ----------
    phot_system : str or None
        The desired photometric system.

    Returns
    -------
    mist_filter_names : list or dict
        If ``phot_system`` given, then a list of filter names is returned. If
        ``phot_system`` is ``None``, then a dict of all photometric systems
        and filters is returned.
    """
    fn = os.path.join(data_dir, 'filters/mist_filter_names.pkl')
    pickle_in = open(fn, 'rb')
    mist_filter_names = pickle.load(pickle_in)
    pickle_in.close()
    if phot_system is not None:
        if type(phot_system) == str:
            phot_system = [phot_system]
        names = []
        for p in phot_system:
            names.extend(mist_filter_names[p])
        mist_filter_names = names
    return mist_filter_names


def phot_system_lookup(filter_name=None):
    """
    Lookup the photometric system name associated with a given filter name.

    Parameters
    ----------
    filter_name : str, optional
        Filter name to lookup.

    Returns
    -------
    lookup : dict or str
        If ``filter_name`` is ``None``, a dictionary with the filter names as
        keywords and the photometric systems as values. If ``filter_name`` is
        not ``None``, its photometric system is returned.
    """
    fn = os.path.join(data_dir, 'filters/phot_system_lookup.pkl')
    pickle_in = open(fn, 'rb')
    lookup = pickle.load(pickle_in)
    pickle_in.close()
    if filter_name is not None:
        lookup = lookup[filter_name]
    return lookup


class FilterSystem(object):
    """
    Class for calculating filter parameters from throughput curves.
    The parameter definitions are taken from
    `Fukugita et al. (1996) AJ 111, 1748.
    <https://ui.adsabs.harvard.edu/abs/1996AJ....111.1748F/abstract>`_

    Parameters
    ----------
    phot_system : str
        Name of the filter photometric system.
    filter_dir : str, optional
        Path to directory containing the filter throughput curves.
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

    def _get_trans(self, bandpass):
        """
        Get the filter throughput curve for the given bandpass.
        """
        lam = getattr(self, bandpass)['wave']
        trans = getattr(self, bandpass)['trans']
        return lam, trans

    def effective_throughput(self, bandpass):
        """
        Calculate the effective throughput for the given bandpass.

        Parameters
        ----------
        bandpass : str
            The name of the filter.

        Returns
        -------
        theff : float
            Effective effective throughput.
        """
        lam, trans = self._get_trans(bandpass)
        theff = np.trapz(trans, np.log(lam))
        return theff

    def lam_eff(self, bandpass):
        """
        Calculate effective wavelength for the given bandpass.

        Parameters
        ----------
        bandpass : str
            The name of the filter.

        Returns
        -------
            leff : float
                The effective wavelength.
        """
        lam, trans = self._get_trans(bandpass)
        log_leff = np.trapz(np.log(lam) * trans, np.log(lam))
        log_leff /= np.trapz(trans, np.log(lam))
        leff =  np.exp(log_leff) * u.angstrom
        return leff

    def lam_pivot(self, bandpass):
        """
        Calculate pivot wavelength for the given bandpass.

        Parameters
        ----------
        bandpass : str
            The name of the filter.

        Returns
        -------
            lpivot : float
                The pivot wavelength.
        """
        lam, trans = self._get_trans(bandpass)
        lpivot = np.trapz(lam * trans, lam)
        lpivot /= np.trapz(trans, np.log(lam))
        lpivot = np.sqrt(lpivot) * u.angstrom
        return lpivot

    def dlam(self, bandpass):
        """
        Calculate the bandpass width.

        Parameters
        ----------
        bandpass : str
            The name of the filter.

        Returns
        -------
            width : float
                The bandpass width.
        """
        lam, trans = self._get_trans(bandpass)
        norm = np.trapz(trans, lam)
        width = (norm / trans.max()) * u.angstrom
        return width


class ZeroPointConverter(object):
    """
    Help class for converting between AB, ST, and Vega magnitudes & colors.

    Parameters
    ----------
    zpt_table : `~astropy.table.Table`
        MIST zero point table, which can be `downloaded here
        <http://waps.cfa.harvard.edu/MIST/BC_tables/zeropoints.txt>`_.
    """

    def __init__(self, zpt_table):
        for f, system, v_to_st, v_to_ab in zpt_table:
            setattr(self, f, [system, v_to_st, v_to_ab])
        self.zpt_table = zpt_table

    def to_vega(self, bandpass):
        """
        Convert to Vega magnitudes.

        Parameters
        ----------
        bandpass : str
            The name of the filter.

        Returns
        -------
        zpt_convert : float
            Zero point conversion magnitude.
        """
        system, v_to_st, v_to_ab = getattr(self, bandpass)
        if system == 'Vega':
            zpt_convert = 0.
        elif system == 'AB':
            zpt_convert = -v_to_ab
        return zpt_convert

    def to_ab(self, bandpass):
        """
        Convert to AB magnitudes.

        Parameters
        ----------
        bandpass : str
            The name of the filter.

        Returns
        -------
        zpt_convert : float
            Zero point conversion magnitude.
        """
        system, v_to_st, v_to_ab = getattr(self, bandpass)
        if system == 'AB':
            zpt_convert = 0.
        elif system == 'Vega':
            zpt_convert = v_to_ab
        return zpt_convert

    def to_st(self, bandpass):
        """
        Convert to ST magnitudes.

        Parameters
        ----------
        bandpass : str
            The name of the filter.

        Returns
        -------
        zpt_convert : float
            Zero point conversion magnitude.
        """
        system, v_to_st, v_to_ab = getattr(self, bandpass)
        if system == 'AB':
            zpt_convert = v_to_st - v_to_ab
        elif system == 'Vega':
            zpt_convert = v_to_st
        return zpt_convert

    def color_to_vega(self, blue, red):
        """
        Convert to Vega colors.

        Parameters
        ----------
        blue : str
            The name of the blue filter.
        red : str
            The name of the red filter.

        Returns
        -------
        zpt_convert : float
            Zero point conversion magnitude.
        """
        blue_convert = self.to_vega(blue)
        red_convert = self.to_vega(red)
        return blue_convert - red_convert

    def color_to_ab(self, blue, red):
        """
        Convert to AB colors.

        Parameters
        ----------
        blue : str
            The name of the blue filter.
        red : str
            The name of the red filter.

        Returns
        -------
        zpt_convert : float
            Zero point conversion magnitude.
        """
        blue_convert = self.to_ab(blue)
        red_convert = self.to_ab(red)
        return blue_convert - red_convert

    def color_to_st(self, blue, red):
        """
        Convert to ST colors.

        Parameters
        ----------
        blue : str
            The name of the blue filter.
        red : str
            The name of the red filter.

        Returns
        -------
        zpt_convert : float
            Zero point conversion magnitude.
        """
        blue_convert = self.to_st(blue)
        red_convert = self.to_st(red)
        return blue_convert - red_convert


def get_zero_point_converter():
    """
    Create and return a `~artpop.filters.ZeroPointConverter` object.
    """
    from astropy.io import ascii
    from .util import data_dir
    fn = os.path.join(data_dir, 'filters', 'zeropoints.txt')
    table= ascii.read(fn)
    return ZeroPointConverter(table)
