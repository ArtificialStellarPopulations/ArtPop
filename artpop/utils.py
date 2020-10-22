import os
import numpy as np
from astropy import units as u
from astropy.utils.misc import isiterable
from . import data_dir


__all__ = ['fetch_psf', 'check_random_state', 'check_units', 
           'check_odd', 'check_xy_dim']


psf_dir = os.path.join(data_dir, 'psfs')
_psf_file_dict = dict(
    hsc=lambda band: os.path.join(psf_dir, 'hsc-psf-' + band + '.fits'),
    sdss=lambda band: os.path.join(psf_dir, 'sdss-psf-' + band + '.fits'),
    lbc=lambda band: os.path.join(psf_dir, 'lbc-psf-' + band + '.fits')
)


def fetch_psf(phot_system, band):
    """
    Fetch the point spread function. 
    
    Parameters
    ----------
    phot_system : str
        Photometric system 
    band : str
        Photometric filter

    Returns
    -------
    psf : 2D ndarray
        The PSF

    Notes
    -----
    The PSF file names are defined in _psf_file_dict.
    """
    from astropy.io import fits
    file_name = _psf_file_dict[phot_system.lower()](band)
    psf = fits.getdata(file_name)
    psf = psf / psf.sum()
    return psf


def check_random_state(seed):
    """
    Turn seed into a `numpy.random.RandomState` instance.
    Parameters
    ----------
    seed : `None`, int, list of ints, or `numpy.random.RandomState`
        If ``seed`` is `None`, return the `~numpy.random.RandomState`
        singleton used by ``numpy.random``.  If ``seed`` is an `int`,
        return a new `~numpy.random.RandomState` instance seeded with
        ``seed``.  If ``seed`` is already a `~numpy.random.RandomState`,
        return it.  Otherwise raise ``ValueError``.
    Returns
    -------
    random_state : `numpy.random.RandomState`
        RandomState object.
    Notes
    -----
    This routine is adapted from scikit-learn.  See
    http://scikit-learn.org/stable/developers/utilities.html#validation-tools.
    """
    import numbers

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    if type(seed)==list:
        if type(seed[0])==int:
            return np.random.RandomState(seed)

    raise ValueError('{0!r} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))


def check_units(value, default_unit):
    t = type(default_unit)
    if type(value) == u.Quantity:
        quantity = value
    elif (t == u.IrreducibleUnit) or (t == u.Unit) or (t == u.CompositeUnit):
        quantity = value * default_unit
    elif t == str:
        quantity = value * getattr(u, default_unit)
    else:
        raise Exception('default_unit must be an astropy unit or string')
    return quantity


def check_odd(val, name='value'):
    if isiterable(val):
        if np.any(np.asarray(val) % 2 == 0):
            raise Exception(f'{name} must be odd')
    else:
        if val % 2 == 0:
            raise Exception(f'{name} must be odd')


def check_xy_dim(xy_dim, force_odd=True):
    if not isiterable(xy_dim):
        xy_dim = [xy_dim, xy_dim]
    xy_dim = np.asarray(xy_dim).astype(int)
    if force_odd:
        check_odd(xy_dim, 'xy dimensions')
    return xy_dim
