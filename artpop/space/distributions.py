import numpy as np
from astropy import units as u
from astropy.modeling.models import Sersic2D
from ..util import check_random_state, check_units, check_xy_dim
from ..log import logger


__all__ = ['xy_from_grid', 'sersic_xy', 'plummer_xy']


def xy_from_grid(num_stars, model, xy_dim, sample_side=None,
                 random_state=None):

    if sample_side is None:
        sample_side = max(xy_dim)

    x_0, y_0 = sample_side//2, sample_side//2
    yy, xx = np.meshgrid(np.arange(sample_side), 
                         np.arange(sample_side))

    image = model(xx, yy)
    weights = image.flatten()
    weights /= np.sum(weights)

    rng = check_random_state(random_state)
    samples = rng.choice(len(weights), size=int(num_stars), p=weights)
    x, y = np.unravel_index(samples, (sample_side, sample_side), order='C')

    shift = (sample_side - xy_dim) // 2
    x -= shift[0]
    y -= shift[1]
    x_0 -= shift[0]
    y_0 -= shift[1]

    xy = np.vstack([x, y]).T
    xy = np.ma.masked_array(xy, mask=np.zeros_like(xy, dtype=bool))

    if xy_dim[0] < sample_side or xy_dim[1] < sample_side:
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

    if n <= 0:
        raise Exception('Sersic index n must be greater than zero.')

    xy_dim = check_xy_dim(xy_dim)

    r_eff = check_units(r_eff, 'kpc').to('Mpc').value
    theta = check_units(theta, 'deg').to('radian').value
    distance = check_units(distance, 'Mpc').value
    pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel)
    
    if r_eff <= 0:
        raise Exception('Effective radius must be greater than zero.')

    r_pix = np.arctan2(r_eff, distance) * u.radian.to('arcsec') * u.arcsec
    r_pix = r_pix.to('pixel', u.pixel_scale(pixel_scale)).value
    sample_side = 2 * np.ceil(r_pix * num_r_eff).astype(int) + 1
    x_0, y_0 = sample_side//2, sample_side//2

    model = Sersic2D(x_0=x_0, y_0=y_0, amplitude=1, r_eff=r_pix, 
                     n=n, ellip=ellip, theta=theta)

    xy = xy_from_grid(num_stars, model, xy_dim, sample_side, 
                      random_state=random_state)

    return xy


def plummer_xy(num_stars, scale_radius, distance, xy_dim, pixel_scale=0.2, 
               random_state=None):

    xy_dim = check_xy_dim(xy_dim)
    rng = check_random_state(random_state)

    scale_radius = check_units(scale_radius, 'kpc').to('Mpc').value
    distance = check_units(distance, 'Mpc').value
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
