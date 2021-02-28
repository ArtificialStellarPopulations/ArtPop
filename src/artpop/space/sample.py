# Third-party
import numpy as np
from astropy import units as u
from astropy.modeling.models import Sersic2D

# Project
from ..util import check_random_state, check_units, check_xy_dim
from ..log import logger


__all__ = ['xy_from_grid', 'sersic_xy', 'plummer_xy', 'uniform_xy']


def xy_from_grid(num_stars, model, xy_dim, sample_dim=None,
                 random_state=None):
    """
    Sample xy positions from a discrete grid weighted by an arbitrary model.

    Parameters
    ----------
    num_stars : int
        Number of stars (i.e., positions) to sample.
    model : `~astropy.modeling.Fittable2DModel`
        A two-dimensional ``astropy`` model or a callable object that takes
        x and y grid positions arguments: `model(xx, yy)`.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    sample_dim : int or None, optional
        Square dimension to sample within. This is useful if the mock image
        size is small compared to the scale of the model. To be useful,
        `sample_dim` should be larger than `xy_dim`.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.

    Returns
    -------
    xy : `~numpy.ma.MaskedArray`
        Masked numpy array of xy positions. Positions that fall outside the
        mock image are masked.
    """
    xy_dim = check_xy_dim(xy_dim)

    if sample_dim is None:
        sample_dim = max(xy_dim)

    num_pix = 2 * sample_dim**2
    byte_per_pixel = 8
    memory_in_GB = (byte_per_pixel * num_pix) / 1e9
    if memory_in_GB > 1:
        msg = "This sampling grid will exceed 1 Gb in memory usage."
        logger.warning(msg)

    x_0, y_0 = sample_dim//2, sample_dim//2
    yy, xx = np.meshgrid(np.arange(sample_dim),
                         np.arange(sample_dim))

    image = model(xx, yy)
    weights = image.flatten()
    weights /= np.sum(weights)

    rng = check_random_state(random_state)
    samples = rng.choice(len(weights), size=int(num_stars), p=weights)
    x, y = np.unravel_index(samples, (sample_dim, sample_dim), order='C')

    shift = (sample_dim - xy_dim) // 2
    x -= shift[0]
    y -= shift[1]
    x_0 -= shift[0]
    y_0 -= shift[1]

    xy = np.vstack([x, y]).T
    xy = np.ma.masked_array(xy, mask=np.zeros_like(xy, dtype=bool))

    if xy_dim[0] < sample_dim or xy_dim[1] < sample_dim:
        outside_image = x < 0
        outside_image |= x > xy_dim[0] - 1
        outside_image |= y < 0
        outside_image |= y > xy_dim[1] - 1

        if outside_image.sum() > 0:
            msg = '{} stars outside the image'.format(outside_image.sum())
            logger.warning(msg)
            xy.mask = np.column_stack((outside_image, outside_image))

    return xy


def sersic_xy(num_stars, r_eff, n, theta, ellip, distance, xy_dim,
              pixel_scale=0.2, num_r_eff=10, random_state=None):
    """
    Sample xy positions from a two-dimensional Sersic distribution.


    .. note::
        To sample the Sersic distribution, `~artpop.space.sample.xy_from_grid`
        is used. This means that large sources (and/or large `num_r_eff`) eat
        up a lot of memory.

    Parameters
    ----------
    num_stars : int
        Number of stars (i.e., positions) to sample.
    r_eff : float or `~astropy.units.Quantity`
        Effective radius of the source. If a float is given, the units are
        assumed to be `~astropy.units.kpc`. Must be greater than zero.
    n : float
        Sersic index. Must be greater than zero.
    theta : float or `~astropy.units.Quantity`
        Rotation angle, counterclockwise from the positive x-axis. If a float
        is given, the units are assumed to be `degree`.
    ellip : float
        Ellipticity.
    distance : float or `~astropy.units.Quantity`
        Distance to source. If float is given, the units are assumed
        to be `~astropy.units.Mpc`.
    xy_dim : list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    pixel_scale : float or `~astropy.units.Quantity`, optional
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `~astropy.units.arcsec` per `~astropy.units.pixels`.
    num_r_eff : float, optional
        Number of r_eff to sample positions within. This parameter is needed
        because the current Sersic sampling function samples from within a
        discrete grid. Default is 10.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.

    Returns
    -------
    xy : `~numpy.ma.MaskedArray`
        Masked numpy array of xy positions. Positions that fall outside the
        mock image are masked.
    """
    if n <= 0:
        raise Exception('Sersic index n must be greater than zero.')

    xy_dim = check_xy_dim(xy_dim)

    r_eff = check_units(r_eff, 'kpc').to('Mpc').value
    theta = check_units(theta, 'deg').to('radian').value
    distance = check_units(distance, 'Mpc').to('Mpc').value
    pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel)

    if r_eff <= 0:
        raise Exception('Effective radius must be greater than zero.')

    r_pix = np.arctan2(r_eff, distance) * u.radian.to('arcsec') * u.arcsec
    r_pix = r_pix.to('pixel', u.pixel_scale(pixel_scale)).value
    sample_dim = 2 * np.ceil(r_pix * num_r_eff).astype(int) + 1
    x_0, y_0 = sample_dim//2, sample_dim//2

    model = Sersic2D(x_0=x_0, y_0=y_0, amplitude=1, r_eff=r_pix,
                     n=n, ellip=ellip, theta=theta)

    xy = xy_from_grid(num_stars, model, xy_dim, sample_dim,
                      random_state=random_state)

    return xy


def plummer_xy(num_stars, scale_radius, distance, xy_dim, pixel_scale=0.2,
               random_state=None):
    """
    Sample xy positions from a two-dimensional Plummer distributions using
    inverse transform sampling.

    Parameters
    ----------
    num_stars : int
        Number of stars (i.e., positions) to sample.
    scale_radius : float or `~astropy.units.Quantity`
        Scale radius of the source. If a float is given, the units are
        assumed to be `~astropy.units.kpc`. Must be greater than zero.
    distance : float or `~astropy.units.Quantity`
        Distance to source. If float is given, the units are assumed
        to be `~astropy.units.Mpc`.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    pixel_scale : float or `~astropy.units.Quantity`, optional
        The pixel scale of the mock image. If a float is given, the units will
        be assumed to be `~astropy.units.arcsec` per `~astropy.units.pixels`.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.

    Returns
    -------
    xy : `~numpy.ma.MaskedArray`
        Masked numpy array of xy positions. Positions that fall outside the
        mock image are masked.
    """
    xy_dim = check_xy_dim(xy_dim)
    rng = check_random_state(random_state)

    scale_radius = check_units(scale_radius, 'kpc').to('Mpc').value
    distance = check_units(distance, 'Mpc').to('Mpc').value
    r_sky = np.arctan2(scale_radius, distance) * u.radian.to('arcsec')
    pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel)

    s = rng.random(size=int(num_stars))
    r = u.arcsec * r_sky / np.sqrt(s**(-2 / 3) - 1)
    r = r.to('pixel', u.pixel_scale(pixel_scale)).value

    phi = rng.uniform(0, 2 * np.pi, size=len(r))
    theta = np.arccos(2 * rng.random(size=len(r)) - 1)
    xy = np.zeros((len(r), 2))

    shift = xy_dim // 2
    xy[:, 0] = r * np.cos(phi) * np.sin(theta) + shift[0]
    xy[:, 1] = r * np.sin(phi) * np.sin(theta) + shift[1]

    x_outside = (xy[:, 0] < 0) | (xy[:, 0] >= xy_dim[0])
    y_outside = (xy[:, 1] < 0) | (xy[:, 1] >= xy_dim[1])
    xy = np.ma.masked_array(xy, mask=np.zeros_like(xy, dtype=bool))

    outside_image = x_outside | y_outside
    if outside_image.sum() > 0:
        msg = '{} stars outside the image'.format(outside_image.sum())
        logger.warning(msg)
        xy.mask = np.column_stack((outside_image, outside_image))

    return xy


def uniform_xy(num_stars, xy_dim, random_state=None):
    """
    Sample xy positions from a uniform distribution.

    Parameters
    ----------
    num_stars : int
        Number of stars (i.e., positions) to sample.
    xy_dim : int or list-like
        Dimensions of the mock image in xy coordinates. If int is given,
        will make the x and y dimensions the same.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.

    Returns
    -------
    xy : `~numpy.array`
        xy pixel positions.
    """
    xy_dim = check_xy_dim(xy_dim)
    rng = check_random_state(random_state)
    num_stars = int(num_stars)
    xy = np.vstack([rng.uniform(0, xy_dim[0], num_stars),
                    rng.uniform(0, xy_dim[1], num_stars)]).T
    return xy
