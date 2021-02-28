# Standard library
from copy import deepcopy

# Third-party
import numpy as np
from astropy import units as u
from astropy.table import Table, vstack

# Project
from .stars import SSP, constant_sb_stars_per_pix
from .space import sersic_xy, plummer_xy, uniform_xy
from .util import check_units, check_xy_dim, MIST_PATH


__all__ = ['Source', 'SersicSSP', 'PlummerSSP', 'UniformSSP']


class Source(object):
    """
    Artificial stellar system to be mock imaged. 

    Parameters
    ----------
    xy : `~numpy.ndarray` or `~numpy.ma.MaskedArray`
        Positions of the stars in (x, y) coordinates. 
    mags : dict or `~astropy.table.Table`
        Stellar magnitudes.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    pixel_scale : float or `~astropy.units.Quantity`, optional
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`.
    labels : list-like, optional
        Labels for the stars. For example, EEP values (int or float) or name 
        of evolutionary phase (str).
    """
    def __init__(self, xy, mags, xy_dim, pixel_scale=0.2, labels=None):
        if type(mags) == dict:
            self.mags = Table(mags)
        else:
            self.mags = mags.copy()
        if len(xy) != len(self.mags):
            raise Exception('numbers of magnitudes and positions must match')
        if type(xy) == np.ma.MaskedArray:
            mask = np.any(~xy.mask, axis=1)
            self.mags = self.mags[mask]
            self.xy = xy.data[mask]
        else:
            self.xy = np.asarray(xy)
            mask = np.ones(len(self.xy), dtype=bool)
        self.labels = labels if labels is None else np.asarray(labels)[mask]
        self.xy_dim = check_xy_dim(xy_dim)
        self.pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel)

    @property
    def x(self):
        """x coordinates"""
        return self.xy[:, 0]

    @property
    def y(self):
        """y coordinates"""
        return self.xy[:, 1]

    @property
    def num_stars(self):
        """Number of stars in source"""
        return len(self.x)

    def __add__(self, src):
        if Source not in src.__class__.__mro__:
            raise Exception(f'{type(src)} is not a valid Source object')
        new = deepcopy(self)
        if new.labels is None:
            new.labels = np.ones(new.num_stars, dtype=int)
        labels = np.ones(src.num_stars, dtype=int) * (new.labels.max() + 1)
        new_labels = np.concatenate([new.labels, labels])
        new.xy = np.concatenate([new.xy, src.xy])
        new.mags = vstack([new.mags, src.mags])
        new.xy_dim = np.max(np.vstack([self.xy_dim, src.xy_dim]), axis=0)
        comp_src = CompositeSource(new.xy, new.mags, new.xy_dim, 
                                   new.pixel_scale, new_labels)
        return comp_src


class CompositeSource(Source):
    """A source consisting of 2 or more `~artpop.source.Source` objects."""
    pass


class SersicSSP(Source):
    """
    Simple stellar population with a Sersic spatial distribution. This is a 
    convenience class that combines `~artpop.space.sersic_xy` and 
    `~artpop.stars.SSP` to make a `~artpop.source.Source` object.

    Parameters
    ----------
    log_age : float
        Log (base 10) of the simple stellar population age in years.
    feh : float
        Metallicity [Fe/H] of the simple stellar population.
    phot_system : str or list-like
        Name of the photometric system(s). 
    r_eff : float or `~astropy.units.Quantity`
        Effective radius of the source. If a float is given, the units are 
        assumed to be `kpc`. Must be greater than zero.
    n : float 
        Sersic index. Must be greater than zero.
    theta : float or `~astropy.units.Quantity`
        Rotation angle, counterclockwise from the positive x-axis. If a float 
        is given, the units are assumed to be `degree`.
    ellip : float
        Ellipticity. 
    distance : float or `~astropy.units.Quantity`
        Distance to source. If float is given, the units are assumed 
        to be `Mpc`.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given, 
        will make the x and y dimensions the same. 
    total_mass : float or `None`
        Stellar mass of the source. If `None`, then must give `num_stars`. 
    num_stars : int or `None`
        Number of stars in source. If `None`, then must give `total_mass`.
    imf : str, optional
        The initial stellar mass function. Default is `'kroupa'`.
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different 
        path from the `MIST_PATH` environment variable. 
    imf_kw : dict, optional
        Optional keyword arguments for sampling the stellar mass function.
    pixel_scale : float or `~astropy.units.Quantity`, optional
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`. Default is `0.2 arcsec / pixel`.
    num_r_eff : float, optional
        Number of r_eff to sample positions within. This parameter is needed 
        because the current Sersic sampling function samples from within a 
        discrete grid. Default is 10.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState` 
        If `None`, return the `~numpy.random.RandomState` singleton used by 
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState` 
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.

    Notes
    -----
    You must give `total_mass` *or* `num_stars`.
    """

    def __init__(self, log_age, feh, phot_system, r_eff, n, theta, ellip, 
                 distance, xy_dim, total_mass=None, num_stars=None, 
                 imf='kroupa', mist_path=MIST_PATH, imf_kw={}, pixel_scale=0.2, 
                 num_r_eff=10, random_state=None):

        self.ssp_kw = dict(log_age=log_age, feh=feh, phot_system=phot_system,
                           distance=distance, total_mass=total_mass, 
                           num_stars=num_stars, imf=imf, mist_path=mist_path,
                           imf_kw=imf_kw, random_state=random_state)
        self.ssp = SSP(**self.ssp_kw)

        self.xy_kw = dict(num_stars=self.ssp.num_stars, r_eff=r_eff, n=n,
                          theta=theta, ellip=ellip, distance=distance, 
                          xy_dim=xy_dim, num_r_eff=num_r_eff, 
                          pixel_scale=pixel_scale, random_state=self.ssp.rng)
        _xy = sersic_xy(**self.xy_kw)

        super(SersicSSP, self).__init__(_xy, self.ssp.mag_table, xy_dim)


class PlummerSSP(Source):
    """
    Simple stellar population with a Plummer spatial distribution.

    Parameters
    ----------
    log_age : float
        Log (base 10) of the simple stellar population age in years.
    feh : float
        Metallicity [Fe/H] of the simple stellar population.
    phot_system : str or list-like
        Name of the photometric system(s). 
    scale_radius : float or `~astropy.units.Quantity`
        Scale radius of the source. If a float is given, the units are 
        assumed to be `kpc`. Must be greater than zero.
    distance : float or `~astropy.units.Quantity`
        Distance to source. If float is given, the units are assumed 
        to be `Mpc`.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given, 
        will make the x and y dimensions the same. 
    total_mass : float or `None`
        Stellar mass of the source. If `None`, then must give `num_stars`. 
    num_stars : int or `None`
        Number of stars in source. If `None`, then must give `total_mass`.
    imf : str, optional
        The initial stellar mass function. Default is `'kroupa'`.
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different 
        path from the `MIST_PATH` environment variable. 
    imf_kw : dict, optional
        Optional keyword arguments for sampling the stellar mass function.
    pixel_scale : float or `~astropy.units.Quantity`, optional
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`. Default is `0.2 arcsec / pixel`.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState` 
        If `None`, return the `~numpy.random.RandomState` singleton used by 
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState` 
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.

    Notes
    -----
    You must give `total_mass` *or* `num_stars`.
    """

    def __init__(self, log_age, feh, phot_system, scale_radius, distance, 
                 xy_dim, total_mass=None, num_stars=None, imf='kroupa', 
                 mist_path=MIST_PATH, imf_kw={}, pixel_scale=0.2, 
                 random_state=None):

        self.ssp_kw = dict(log_age=log_age, feh=feh, phot_system=phot_system,
                           distance=distance, total_mass=total_mass, 
                           num_stars=num_stars, imf=imf, mist_path=mist_path,
                           imf_kw=imf_kw, random_state=random_state)
        self.ssp = SSP(**self.ssp_kw)

        self.xy_kw = dict(num_stars=self.ssp.num_stars, 
                          scale_radius=scale_radius, distance=distance, 
                          xy_dim=xy_dim, pixel_scale=pixel_scale, 
                          random_state=self.ssp.rng)
        _xy = plummer_xy(**self.xy_kw)

        super(PlummerSSP, self).__init__(_xy, self.ssp.mag_table, xy_dim)


class UniformSSP(Source):
    """
    Simple stellar population with a uniform spatial distribution.

    Parameters
    ----------
    log_age : float
        Log (base 10) of the simple stellar population age in years.
    feh : float
        Metallicity [Fe/H] of the simple stellar population.
    phot_system : str or list-like
        Name of the photometric system(s). 
    distance : float or `~astropy.units.Quantity`
        Distance to source. If float is given, the units are assumed 
        to be `Mpc`.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given, 
        will make the x and y dimensions the same. 
    sb : float
        Surface brightness of of uniform stellar distribution.  
    sb_bandpass : str
        Photometric filter of ``sb``. Must be part of ``phot_system``.
    imf : str, optional
        The initial stellar mass function. Default is `'kroupa'`.
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different 
        path from the `MIST_PATH` environment variable. 
    imf_kw : dict, optional
        Optional keyword arguments for sampling the stellar mass function.
    pixel_scale : float or `~astropy.units.Quantity`, optional
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`. Default is `0.2 arcsec / pixel`.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState` 
        If `None`, return the `~numpy.random.RandomState` singleton used by 
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState` 
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.

    Notes
    -----
    You must give `total_mass` *or* `num_stars`.
    """
    
    def __init__(self, log_age, feh, phot_system, distance, xy_dim, sb, 
                 sb_bandpass, imf='kroupa', mist_path=MIST_PATH, imf_kw={}, 
                 pixel_scale=0.2, random_state=None):

        self.ssp_kw = dict(log_age=log_age, feh=feh, phot_system=phot_system,
                           distance=10*u.pc, total_mass=1e6, 
                           num_stars=None, imf=imf, mist_path=mist_path,
                           imf_kw=imf_kw, random_state=random_state)

        # Here we calculate the mean mag to the SSP.
        # We use a relatively large mass to ensure we 
        # fully sample the imf. Also need absolute mags.
        _ssp = SSP(**self.ssp_kw)
        mean_mag = _ssp.mean_mag(sb_bandpass)

        stars_per_pix = constant_sb_stars_per_pix(sb, mean_mag, 
                                                  distance, pixel_scale)

        xy_dim  = check_xy_dim(xy_dim)
        self.ssp_kw['distance'] = distance
        self.ssp_kw['total_mass'] = None
        self.ssp_kw['num_stars'] = int(stars_per_pix * np.multiply(*xy_dim))
        self.ssp = SSP(**self.ssp_kw)

        self.xy_kw = dict(num_stars=self.ssp.num_stars, 
                          xy_dim=xy_dim, random_state=self.ssp.rng)

        _xy = uniform_xy(**self.xy_kw)
        super(UniformSSP, self).__init__(_xy, self.ssp.mag_table, xy_dim)
