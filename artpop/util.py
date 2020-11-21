#Standard library
import os

# Third-party
import numpy as np
from astropy import units as u
from astropy.utils.misc import isiterable


__all__ = ['check_random_state', 
           'check_units', 
           'check_odd', 
           'check_xy_dim', 
           'embed_slices']


project_dir = os.path.dirname(os.path.dirname(__file__))
package_dir = os.path.join(project_dir, 'artpop')
data_dir = os.path.join(package_dir, 'data')

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    MIST_PATH = 'MIST_PATH'
else:
    MIST_PATH = os.getenv('MIST_PATH')
    if MIST_PATH is None:
        print('\033[33mWARNING:\033[0m Environment variable MIST_PATH does '
               'not exist. You will need to pass the path to the MIST grids '
               'to all functions that use them.')


def check_random_state(seed):
    """
    Turn seed into a `~numpy.random.RandomState` instance.

    Parameters
    ----------
    seed : `None`, int, list of ints, or `~numpy.random.RandomState`
        If ``seed`` is `None`, return the `~numpy.random.RandomState`
        singleton used by ``numpy.random``.  If ``seed`` is an `int`,
        return a new `~numpy.random.RandomState` instance seeded with
        ``seed``.  If ``seed`` is already a `~numpy.random.RandomState`,
        return it.  Otherwise raise ``ValueError``.

    Returns
    -------
    random_state : `~numpy.random.RandomState`
        RandomState object.

    Notes
    -----
    This routine is adapted from scikit-learn. See
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
    """
    Check if an object has units. If not, apply the default unit. 

    Parameters
    ----------
    value : float, list-like, or `~astropy.units.Quantity`
        Parameter that has units.
    default_unit : str or `astropy` unit
        The default unit to apply to `value` if it does not have units. 
        
    Returns
    -------
    quantity : `~astropy.units.Quantity`
        `value` with ``astropy`` units.
    """
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
    """
    Raise Exception if  `val` is not odd. 
    """
    if isiterable(val):
        if np.any(np.asarray(val) % 2 == 0):
            raise Exception(f'{name} must be odd')
    else:
        if val % 2 == 0:
            raise Exception(f'{name} must be odd')


def check_xy_dim(xy_dim, force_odd=True):
    """
    Check the format of `xy_dim`. 

    Parameters
    ----------
    xy_dim : int or list-like
        The dimensions of mock image in xy units. If `int` is given, it is 
        assumed to be both the x and y dimensions and (xy_dim, xy_dim) will be
        returned. Otherwise xy_dim will be returned. 
    force_odd : bool, optional
        If True (default), force both the x and y dimensions to be odd.

    Returns
    -------
    xy_dim : `~numpy.ndarray`
        Dimensions of mock image in xy units.
    """
    if not isiterable(xy_dim):
        xy_dim = [xy_dim, xy_dim]
    xy_dim = np.asarray(xy_dim).astype(int)
    if force_odd:
        check_odd(xy_dim, 'xy dimensions')
    return xy_dim


def embed_slices(center, arr_shape, img_shape):
    """
    Get slices to embed smaller array into larger image.

    Parameters
    ----------
    center : `~numpy.ndarray`
        Center of array in the image coordinates.
    arr_shape : tuple
        Shape of the array to embed (dimensions must be odd).
    img_shape : tuple
        Shape of the main image array.

    Returns
    -------
    img_slice, arr_slice : tuples of slices
        Slicing indices. To embed array in image,
        use the following: img[img_slice] = arr[arr_slice]
    """
    arr_shape = np.asarray(arr_shape)
    img_shape = np.asarray(img_shape)

    check_odd(arr_shape, 'embed_slices array shape')

    imin = center - arr_shape//2
    imax = center + arr_shape//2

    amin = (imin < np.array([0,0]))*(-imin)
    amax = arr_shape*(imax<=img_shape-1) +\
           (arr_shape-(imax-(img_shape-1)))*(imax>img_shape-1)

    imin = np.maximum(imin, np.array([0, 0]))
    imax = np.minimum(imax, np.array(img_shape)-1)
    imax += 1

    img_slice = np.s_[imin[0]:imax[0], imin[1]:imax[1]]
    arr_slice = np.s_[amin[0]:amax[0], amin[1]:amax[1]]

    return img_slice, arr_slice
