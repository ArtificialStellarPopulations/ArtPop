# Standard library
from copy import deepcopy

# Third-party
import numpy as np
from astropy import units as u
from astropy.table import Table, vstack
from astropy.modeling.models import Sersic2D
from scipy.special import gammaincinv, gamma

# Project
from .stars import SSP, MISTSSP, MISTIsochrone, constant_sb_stars_per_pix
from .space import sersic_xy, plummer_xy, uniform_xy, Plummer2D, Constant2D
from .util import check_units, check_xy_dim, MIST_PATH


__all__ = [
    'Source',
    'SersicSP',
    'MISTSersicSSP',
    'PlummerSP',
    'MISTPlummerSSP',
    'UniformSSP',
    'MISTUniformSSP',
]


class Source(object):
    """
    Artificial stellar system to be mock imaged.

    Parameters
    ----------
    xy : `~numpy.ndarray` or `~numpy.ma.MaskedArray`
        Positions of the stars in (x, y) coordinates.
    mags : dict of `~numpy.ndarray` or `~astropy.table.Table`
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
    def __init__(self, xy, mags, xy_dim, pixel_scale=None, labels=None):
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
        if pixel_scale is not None:
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

    def copy(self):
        """Create deep copy of source object."""
        return deepcopy(self)

    def __add__(self, src):
        if Source not in src.__class__.__mro__:
            raise Exception(f'{type(src)} is not a valid Source object')
        new = deepcopy(self)
        if not np.allclose(new.xy_dim, src.xy_dim):
            raise Exception('Sources must have the same xy_dim')
        if new.labels is None:
            new.labels = np.ones(new.num_stars, dtype=int)
        labels = np.ones(src.num_stars, dtype=int) * (new.labels.max() + 1)
        new_labels = np.concatenate([new.labels, labels])
        new.xy = np.concatenate([new.xy, src.xy])
        new.mags = vstack([new.mags, src.mags])
        comp_src = CompositeSource(new.xy, new.mags, new.xy_dim,
                                   new.pixel_scale, new_labels)
        return comp_src


class CompositeSource(Source):
    """A source consisting of 2 or more `~artpop.source.Source` objects."""
    pass


class SersicSP(Source):
    """
    Stellar population with a Sersic spatial distribution.

    Parameters
    ----------
    sp : `~artpop.stars.populations.StellarPopulation`
        A stellar population object.
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
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    pixel_scale : float or `~astropy.units.Quantity`
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`. Default is `0.2 arcsec / pixel`.
    num_r_eff : float, optional
        Number of r_eff to sample positions within. This parameter is needed
        because the current Sersic sampling function samples from within a
        discrete grid. Default is 10.
    dx : float, optional
        Shift from center of image in the x direction.
    dy : float, optional
        Shift from center of image in the y direction.
    labels : list-like, optional
        Labels for the stars. For example, EEP values (int or float) or name
        of evolutionary phase (str).
    """

    def __init__(self, sp, r_eff, n, theta, ellip, xy_dim, pixel_scale,
                 num_r_eff=10, dx=0, dy=0, labels=None):

        self.sp = sp
        self.mag_limit = sp.mag_limit
        self.mag_limit_band = sp.mag_limit_band
        self.smooth_model = None

        if self.mag_limit is not None and sp.frac_num_sampled < 1.0:
            _r_eff = check_units(r_eff, 'kpc').to('Mpc').value
            _theta = check_units(theta, 'deg').to('radian').value
            _distance = check_units(sp.distance, 'Mpc').to('Mpc').value
            _pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel)

            if _r_eff <= 0:
                raise Exception('Effective radius must be greater than zero.')

            xy_dim = check_xy_dim(xy_dim)
            x_0, y_0 = xy_dim//2
            x_0 += dx
            y_0 += dy
            self.n = n
            self.ellip = ellip
            self.r_sky = np.arctan2(_r_eff, _distance) * u.radian.to('arcsec')
            self.r_sky *= u.arcsec
            r_pix = self.r_sky.to('pixel', u.pixel_scale(_pixel_scale)).value

            self.smooth_model = Sersic2D(
                x_0=x_0, y_0=y_0, n=n, r_eff=r_pix, theta=_theta, ellip=ellip)

        self.xy_kw = dict(num_stars=sp.num_stars, r_eff=r_eff, n=n,
                          theta=theta, ellip=ellip, distance=sp.distance,
                          xy_dim=xy_dim, num_r_eff=num_r_eff, dx=dx, dy=dy,
                          pixel_scale=pixel_scale, random_state=sp.rng)

        _xy = sersic_xy(**self.xy_kw)

        super(SersicSP, self).__init__(
            _xy, sp.mag_table, xy_dim, pixel_scale, labels)

    def mag_to_image_amplitude(self, m_tot, zpt):
        """
        Convert total magnitude into amplitude parameter for the smooth model.

        Parameters
        ----------
        m_tot : float
            Total magnitude in the smooth component of the system.
        zpt : float
            Photometric zero point.

        Returns
        -------
        mu_e : float
            Surface brightness at the effective radius of the Sersic
            distribution in mags per square arcsec.
        amplitude : float
            Amplitude parameter for the smooth model in image flux units.
        param_name : str
            Name of amplitude parameter (needed to set its value when
            generating the smooth model).
        """
        param_name = 'amplitude'
        b_n = gammaincinv(2.0 * self.n, 0.5)
        f_n = gamma(2 * self.n) *self.n * np.exp(b_n) / b_n**(2 * self.n)
        r_circ = self.r_sky * np.sqrt(1 - self.ellip)
        area = np.pi * r_circ.to('arcsec').value**2
        mu_e = m_tot + 2.5 * np.log10(2 * area) + 2.5 * np.log10(f_n)
        amplitude = 10**(0.4 * (zpt - mu_e)) * self.pixel_scale.value**2
        return mu_e, amplitude, param_name


class MISTSersicSSP(SersicSP):
    """
    MIST simple stellar population with a Sersic spatial distribution. This
    is a convenience class that combines `~artpop.space.sersic_xy` and
    `~artpop.stars.MISTSSP` to make a `~artpop.source.Source` object.

    .. note::
        You must give `total_mass` *or* `num_stars`.

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
        Ellipticity defined as `1 - b/a`, where `b` is the semi-minor axis
        and `a` is the semi-major axis.
    distance : float or `~astropy.units.Quantity`
        Distance to source. If float is given, the units are assumed
        to be `Mpc`.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    pixel_scale : float or `~astropy.units.Quantity`
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`. Default is `0.2 arcsec / pixel`.
    num_stars : int or `None`
        Number of stars in source. If `None`, then must give `total_mass`.
    total_mass : float or `None`
        Stellar mass of the source. If `None`, then must give `num_stars`. This
        mass accounts for stellar remnants, so the actual sampled mass will be
        less than the given value.
    mag_limit : float, optional
        Only sample individual stars that are brighter than this magnitude. All
        fainter stars will be combined into an integrated component. Otherwise,
        all stars in the population will be sampled. You must also give the
        `mag_limit_band` if you use this parameter.
    mag_limit_band : str, optional
        Bandpass of the limiting magnitude. You must give this parameter if
        you use the `mag_limit` parameter.
    imf : str, optional
        The initial stellar mass function. Default is `'kroupa'`.
    imf_kw : dict, optional
        Optional keyword arguments for sampling the stellar mass function.
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different
        path from the `MIST_PATH` environment variable.
    num_r_eff : float, optional
        Number of r_eff to sample positions within. This parameter is needed
        because the current Sersic sampling function samples from within a
        discrete grid. Default is 10.
    mass_tolerance : float, optional
        Tolerance in the fractional difference between the input mass and the
        final mass of the population. The parameter is only used when
        `total_mass` is given.
    dx : float, optional
        Shift from center of image in the x direction.
    dy : float, optional
        Shift from center of image in the y direction.
    labels : list-like, optional
        Labels for the stars. For example, EEP values (int or float) or name
        of evolutionary phase (str).
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.
    """

    def __init__(self, log_age, feh, phot_system, r_eff, n, theta, ellip,
                 distance, xy_dim, pixel_scale, num_stars=None,
                 total_mass=None, mag_limit=None, mag_limit_band=None,
                 imf='kroupa', imf_kw={}, mist_path=MIST_PATH, num_r_eff=10,
                 mass_tolerance=0.01, dx=0, dy=0, labels=None,
                 random_state=None):

        self.ssp_kw = dict(log_age=log_age, feh=feh, phot_system=phot_system,
                           distance=distance, total_mass=total_mass,
                           num_stars=num_stars, imf=imf, mist_path=mist_path,
                           imf_kw=imf_kw, random_state=random_state,
                           mag_limit=mag_limit, mag_limit_band=mag_limit_band,
                           mass_tolerance=mass_tolerance)

        ssp = MISTSSP(**self.ssp_kw)

        super(MISTSersicSSP, self).__init__(
            sp=ssp, r_eff=r_eff, n=n, theta=theta, ellip=ellip, xy_dim=xy_dim,
            pixel_scale=pixel_scale, num_r_eff=num_r_eff, dx=dx, dy=dy,
            labels=labels)


class PlummerSP(Source):
    """
    Simple stellar population with a Plummer spatial distribution.

    Parameters
    ----------
    sp : `~artpop.stars.populations.StellarPopulation`
        A stellar population object.
    scale_radius : float or `~astropy.units.Quantity`
        Scale radius of the source. If a float is given, the units are
        assumed to be `kpc`. Must be greater than zero.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    pixel_scale : float or `~astropy.units.Quantity`
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`. Default is `0.2 arcsec / pixel`.
    dx : float, optional
        Shift from center of image in the x direction.
    dy : float, optional
        Shift from center of image in the y direction.
    labels : list-like, optional
        Labels for the stars. For example, EEP values (int or float) or name
        of evolutionary phase (str).
    """

    def __init__(self, sp, scale_radius, xy_dim, pixel_scale, dx=0, dy=0,
                 labels=None):

        self.sp = sp
        self.mag_limit = sp.mag_limit
        self.mag_limit_band = sp.mag_limit_band
        self.smooth_model = None

        if self.mag_limit is not None and sp.frac_num_sampled < 1.0:
            _rs = check_units(scale_radius, 'kpc').to('Mpc').value
            _distance = check_units(sp.distance, 'Mpc').to('Mpc').value
            _pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel)

            self.r_sky = np.arctan2(_rs, _distance) * u.radian.to('arcsec')
            self.r_sky *= u.arcsec
            r_pix = self.r_sky.to('pixel', u.pixel_scale(_pixel_scale)).value

            xy_dim = check_xy_dim(xy_dim)
            x_0, y_0 = xy_dim//2
            x_0 += dx
            y_0 += dy

            self.smooth_model = Plummer2D(x_0=x_0, y_0=y_0, scale_radius=r_pix)

        self.xy_kw = dict(num_stars=sp.num_stars, scale_radius=scale_radius,
                          distance=sp.distance, xy_dim=xy_dim,
                          pixel_scale=pixel_scale, dx=dx, dy=dy,
                          random_state=sp.rng)

        _xy = plummer_xy(**self.xy_kw)
        super(PlummerSP, self).__init__(
            _xy, sp.mag_table, xy_dim, pixel_scale, labels)

    def mag_to_image_amplitude(self, m_tot, zpt):
        """
        Convert total magnitude into amplitude parameter for the smooth model.

        Parameters
        ----------
        m_tot : float
            Total magnitude in the smooth component of the system.
        zpt : float
            Photometric zero point.

        Returns
        -------
        mu_0 : float
            Central surface brightness of the Plummer distribution in
            mags per square arcsec.
        amplitude : float
            Amplitude parameter for the smooth model in image flux units.
        param_name : str
            Name of amplitude parameter (needed to set its value when
            generating the smooth model).
        """
        param_name = 'amplitude'
        area = np.pi * self.r_sky.to('arcsec').value**2
        flux = 10**(0.4 * (zpt - m_tot))
        mu_0 = zpt - 2.5 * np.log10(flux / area)
        amplitude = (flux / area) * self.pixel_scale.value**2
        return mu_0, amplitude, param_name


class MISTPlummerSSP(PlummerSP):
    """
    MIST simple stellar population with a Plummer spatial distribution.

    .. note::
        You must give `total_mass` *or* `num_stars`.

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
    pixel_scale : float or `~astropy.units.Quantity`
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`. Default is `0.2 arcsec / pixel`.
    num_stars : int or `None`
        Number of stars in source. If `None`, then must give `total_mass`.
    total_mass : float or `None`
        Stellar mass of the source. If `None`, then must give `num_stars`. This
        mass accounts for stellar remnants, so the actual sampled mass will be
        less than the given value.
    mag_limit : float, optional
        Only sample individual stars that are brighter than this magnitude. All
        fainter stars will be combined into an integrated component. Otherwise,
        all stars in the population will be sampled. You must also give the
        `mag_limit_band` if you use this parameter.
    mag_limit_band : str, optional
        Bandpass of the limiting magnitude. You must give this parameter if
        you use the `mag_limit` parameter.
    imf : str, optional
        The initial stellar mass function. Default is `'kroupa'`.
    imf_kw : dict, optional
        Optional keyword arguments for sampling the stellar mass function.
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different
        path from the `MIST_PATH` environment variable.
    mass_tolerance : float, optional
        Tolerance in the fractional difference between the input mass and the
        final mass of the population. The parameter is only used when
        `total_mass` is given.
    dx : float, optional
        Shift from center of image in the x direction.
    dy : float, optional
        Shift from center of image in the y direction.
    labels : list-like, optional
        Labels for the stars. For example, EEP values (int or float) or name
        of evolutionary phase (str).
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.
    """

    def __init__(self, log_age, feh, phot_system, scale_radius,
                 distance, xy_dim, pixel_scale, num_stars=None,
                 total_mass=None, mag_limit=None, mag_limit_band=None,
                 imf='kroupa', imf_kw={}, mist_path=MIST_PATH,
                 mass_tolerance=0.01, dx=0, dy=0, labels=None,
                 random_state=None):

        self.ssp_kw = dict(log_age=log_age, feh=feh, phot_system=phot_system,
                           distance=distance, total_mass=total_mass,
                           num_stars=num_stars, imf=imf, mist_path=mist_path,
                           imf_kw=imf_kw, random_state=random_state,
                           mag_limit=mag_limit, mag_limit_band=mag_limit_band,
                           mass_tolerance=mass_tolerance)

        ssp = MISTSSP(**self.ssp_kw)

        super(MISTPlummerSSP, self).__init__(
            sp=ssp, scale_radius=scale_radius, xy_dim=xy_dim,
            pixel_scale=pixel_scale, dx=dx, dy=dy, labels=labels)


class UniformSSP(Source):
    """
    Simple stellar population with a uniform spatial distribution.

    Parameters
    ----------
    isochrone  : `~artpop.stars.Isochrone`
        Isochrone object.
    distance : float or `~astropy.units.Quantity`
        Distance to source. If float is given, the units are assumed
        to be `Mpc`.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    sb : float
        Surface brightness of of uniform stellar distribution.
    sb_band : str
        Photometric filter of ``sb``. Must be part of ``phot_system``.
    mag_limit : float, optional
        Only sample individual stars that are brighter than this magnitude. All
        fainter stars will be combined into an integrated component. Otherwise,
        all stars in the population will be sampled. You must also give the
        `mag_limit_band` if you use this parameter.
    mag_limit_band : str, optional
        Bandpass of the limiting magnitude. You must give this parameter if
        you use the `mag_limit` parameter.
    pixel_scale : float or `~astropy.units.Quantity`, optional
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`. Default is `0.2 arcsec / pixel`.
    imf : str, optional
        The initial stellar mass function. Default is `'kroupa'`.
    imf_kw : dict, optional
        Optional keyword arguments for sampling the stellar mass function.
    ssp_kw : dict, optional
        Additional keyword arguments for `~artpop.stars.SSP`.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.
    """

    def __init__(self, isochrone, distance, xy_dim, pixel_scale, sb, sb_band,
                 mag_limit=None, mag_limit_band=None, imf='kroupa', imf_kw={},
                 ssp_kw={}, random_state=None):

        self.isochrone = isochrone
        self.mag_limit = mag_limit
        self.mag_limit_band = mag_limit_band
        self.smooth_model = None

        mean_mag = isochrone.ssp_mag(
            sb_band, norm_type='number', m_min_norm=isochrone.m_min,
            m_max_norm=isochrone.m_max)

        stars_per_pix = constant_sb_stars_per_pix(
            sb, mean_mag, distance, pixel_scale)

        xy_dim  = check_xy_dim(xy_dim)
        self.ssp_kw = ssp_kw
        self.ssp_kw['mag_limit'] = mag_limit
        self.ssp_kw['mag_limit_band'] = mag_limit_band
        self.ssp_kw['distance'] = distance
        self.ssp_kw['total_mass'] = None
        self.ssp_kw['num_stars'] = int(stars_per_pix * np.multiply(*xy_dim))
        self.sp = SSP(isochrone, **self.ssp_kw)

        if mag_limit is not None and self.sp.frac_num_sampled < 1.0:
            self.smooth_model = Constant2D()

        self.xy_kw = dict(num_stars=self.sp.num_stars,
                          xy_dim=xy_dim, random_state=self.sp.rng)

        _xy = uniform_xy(**self.xy_kw)
        super(UniformSSP, self).__init__(
            _xy, self.sp.mag_table, xy_dim, pixel_scale)

    def mag_to_image_amplitude(self, m_tot, zpt):
        """
        Convert total magnitude into amplitude parameter for the smooth model.

        Parameters
        ----------
        m_tot : float
            Total magnitude in the smooth component of the system.
        zpt : float
            Photometric zero point.

        Returns
        -------
        mu : float
            Constant surface brightness in mags per square arcsec.
        amplitude : float
            Amplitude parameter for the smooth model in image flux units.
        param_name : str
            Name of amplitude parameter (needed to set its value when
            generating the smooth model).
        """
        param_name = 'amplitude'
        area = np.multiply(*self.xy_dim)
        flux = 10**(0.4 * (zpt - m_tot))
        amplitude = flux / area
        mu = zpt - 2.5 * np.log10(amplitude / self.pixel_scale.value**2)
        return mu, amplitude, param_name


class MISTUniformSSP(UniformSSP):
    """
    MIST simple stellar population with a uniform spatial distribution.

    .. note::
        You must give `total_mass` *or* `num_stars`.

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
    pixel_scale : float or `~astropy.units.Quantity`
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `arcsec / pixels`. Default is `0.2 arcsec / pixel`.
    sb : float
        Surface brightness of of uniform stellar distribution.
    sb_band : str
        Photometric filter of ``sb``. Must be part of ``phot_system``.
    mag_limit : float, optional
        Only sample individual stars that are brighter than this magnitude. All
        fainter stars will be combined into an integrated component. Otherwise,
        all stars in the population will be sampled. You must also give the
        `mag_limit_band` if you use this parameter.
    mag_limit_band : str, optional
        Bandpass of the limiting magnitude. You must give this parameter if
        you use the `mag_limit` parameter.
    imf : str, optional
        The initial stellar mass function. Default is `'kroupa'`.
    imf_kw : dict, optional
        Optional keyword arguments for sampling the stellar mass function.
    ssp_kw : dict, optional
        Additional keyword arguments for `~artpop.stars.SSP`.
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different
        path from the `MIST_PATH` environment variable.
    labels : list-like, optional
        Labels for the stars. For example, EEP values (int or float) or name
        of evolutionary phase (str).
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.
    """

    def __init__(self, log_age, feh, phot_system, distance, xy_dim,
                 pixel_scale, sb, sb_band, mag_limit=None, mag_limit_band=None,
                 imf='kroupa', imf_kw={}, ssp_kw={}, mist_path=MIST_PATH,
                 labels=None, random_state=None, **kwargs):

        self.iso_kw = dict(log_age=log_age, feh=feh, phot_system=phot_system,
                           mist_path=mist_path)
        self.iso_kw.update(kwargs)

        iso = MISTIsochrone(**self.iso_kw)

        super(MISTUniformSSP, self).__init__(
            isochrone=iso, distance=distance, xy_dim=xy_dim, sb=sb,
            sb_band=sb_band, mag_limit=mag_limit,
            mag_limit_band=mag_limit_band, pixel_scale=pixel_scale, imf=imf,
            imf_kw=imf_kw, ssp_kw=ssp_kw, labels=labels,
            random_state=random_state)
