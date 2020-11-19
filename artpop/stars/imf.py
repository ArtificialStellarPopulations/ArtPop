# Third-party
import numpy as np
from scipy.interpolate import interp1d

# Project
from ..util import check_random_state


__all__ = ['broken_power_law', 'broken_power_law_normed', 'salpeter', 
           'kroupa', 'scalo', 'kroupa_func', 'scalo_func', 'imf_dict',
           'salpeter_func', 'sample_imf', 'build_galaxy']


def salpeter(mass_grid, norm_unit='number'):
    """
    Calculate weights for the Salpeter IMF (`Salpeter 1955
    <https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/abstract>`_).

    Parameters
    ----------
    mass_grid : `~numpy.ndarray`
        Stellar mass grid.
    norm_unit : str, optional
        How to normalize the weights: by 'number', 'mass', or the 'sum'.

    Returns
    -------
    weights : `~numpy.ndarray`
        The weights associated with each mass in the input `mass_grid`.
    """
    alpha = 2.35

    weights =  mass_grid**-alpha

    if norm_unit is None:
        return weights

    m_min = mass_grid.min()
    m_max = mass_grid.max()

    if norm_unit == 'number':
        beta = alpha - 1
        norm = beta / (m_min**-beta - m_max**-beta)
    elif norm_unit == 'mass':
        beta = alpha - 2
        norm = beta / (m_min**-beta - m_max**-beta)
    elif norm_unit == 'sum':
        norm = 1 / weights.sum()

    weights *= norm

    return weights


def broken_power_law(mass_grid, alphas, breaks):
    """
    Calculate weights for broken power law. 
    
    Parameters
    ----------
    mass_grid : `~numpy.ndarray`
        Stellar mass grid.
    alphas : list-like
        The three slopes of the broken power law.
    breaks  : list-like
        The two break point masses of the broken power law.
        
    Returns
    -------
    weights : `~numpy.ndarray`
        The weights associated with each mass in the input `mass_grid`.
    """
    mass_grid = np.asarray(mass_grid)
    a1, a2, a3 = alphas 
    b1, b2 = breaks
    alpha = np.where(mass_grid < b1, a1, np.where(mass_grid < b2, a2, a3))
    m_break = np.where(mass_grid < b2, b1, b2 * (b1 / b2)**(a2 / a3))
    weights = (mass_grid / m_break)**(-alpha)
    return weights


def broken_power_law_normed(mass_grid, alphas, breaks, norm_unit='number', 
                            num_norm_bins=1e5, norm_mass_min=None, 
                            norm_mass_max=None):
    """
    Calculate normalized weights for broken power law. 
    
    Parameters
    ----------
    mass_grid : `~numpy.ndarray`
        Stellar mass grid.
    alphas : list-like
        The three slopes of the broken power law.
    breaks  : list-like
        The two break point masses of the broken power law.
    norm_unit : str, optional
        How to normalize the weights: by 'number', 'mass', or the 'sum'.
    num_norm_bins : int, optional
        Number of mass bins to use for integration (if needed to normalize).
    norm_mass_min : int or None, optional
        Minimum mass to use for normalization. If None, use minimum of 
        `mass_grid` will be used.
    norm_mass_max : int or None, optional
        Maximum mass to use for normalization. If None, use maximum of 
        `mass_grid` will be used.
        
    Returns
    -------
    weights : `~numpy.ndarray`
        The weights associated with each mass in the input `mass_grid`.
    """
    weights = broken_power_law(mass_grid, alphas, breaks)
    if norm_unit ==  'sum':
        norm = weights.sum()
    else:
        m_min = norm_mass_min if norm_mass_min else mass_grid.min() 
        m_max = norm_mass_max if norm_mass_max else mass_grid.max()
        m_integrate = np.linspace(m_min, m_max, int(num_norm_bins))
        w_integrate = broken_power_law(m_integrate, alphas, breaks)
        if norm_unit == 'number':
            norm = np.trapz(w_integrate, m_integrate)
        elif norm_unit == 'mass':
            w_integrate = m_integrate * w_integrate
            norm = np.trapz(w_integrate, m_integrate)
    weights /= norm
    return weights


def kroupa(mass_grid, norm_unit='number', num_norm_bins=1e5, 
           norm_mass_min=None, norm_mass_max=None):
    """
    The Kroupa stellar initial mass function (`Kroupa 2001
    https://ui.adsabs.harvard.edu/abs/2001MNRAS.322..231K/abstract>`_).

    Parameters
    ----------
    mass_grid : `~numpy.ndarray`
        Stellar mass grid.
    norm_unit : str, optional
        How to normalize the weights: by 'number', 'mass', or the 'sum'.
    num_norm_bins : int, optional
        Number of mass bins to use for integration (if needed to normalize).
    norm_mass_min : int or None, optional
        Minimum mass to use for normalization. If None, use minimum of 
        `mass_grid` will be used.
    norm_mass_max : int or None, optional
        Maximum mass to use for normalization. If None, use maximum of 
        `mass_grid` will be used.

    Returns
    -------
    weights : `~numpy.ndarray`
        The weights associated with each mass in the input `mass_grid`.
    """
    alphas = [0.3, 1.3, 2.3]
    breaks = [0.08, 0.5]
    if norm_unit is not None: 
        kw = dict(norm_unit=norm_unit, num_norm_bins=num_norm_bins, 
                  norm_mass_min=norm_mass_min, norm_mass_max=norm_mass_max)
        weights = broken_power_law_normed(mass_grid, alphas, breaks, **kw)
    else:
        weights = broken_power_law(mass_grid, alphas, breaks)
    return weights


def scalo(mass_grid, norm_unit='number', num_norm_bins=1e5):
    """
    The Scalo stellar initial mass function (`Scalo 1998
    https://ui.adsabs.harvard.edu/abs/1998ASPC..142..201S/abstract>`_).

    Parameters
    ----------
    mass_grid : `~numpy.ndarray`
        Stellar mass grid.
    norm_unit : str, optional
        How to normalize the weights: by 'number', 'mass', or the 'sum'.
    num_norm_bins : int, optional
        Number of mass bins to use for integration (if needed to normalize).

    Returns
    -------
    weights : `~numpy.ndarray`
        The weights associated with each mass in the input `mass_grid`.
    """
    alphas = [1.2, 2.7, 2.3]
    breaks = [1, 10]
    if norm_unit is not None: 
        kw = dict(norm_unit=norm_unit, num_norm_bins=num_norm_bins)
        weights = broken_power_law_normed(mass_grid, alphas, breaks, **kw)
    else:
        weights = broken_power_law(mass_grid, alphas, breaks)
    return weights


# IMF helper dictionary 
imf_dict = dict(salpeter=salpeter, kroupa=kroupa, scalo=scalo)

# IMF helper functions
kroupa_func = lambda m: broken_power_law(m, [0.3, 1.3, 2.3], [0.08, 0.5])
scalo_func = lambda m: broken_power_law(m, [1.2, 2.7, 2.3], [1, 10])
salpeter_func = lambda m: m**-2.35


def sample_imf(num_stars, m_min=0.08, m_max=120, imf='kroupa', 
               num_mass_bins=100000, random_state=None, imf_kw={}):
    """
    Sample stellar IMF via inverse transform sampling. 

    Parameters
    ----------
    num_stars : int
        Number of stars to sample. 
    m_min : float
        Minimum stellar mass. 
    m_max : float
        Maxium stellar mass. 
    imf : str
        The desired IMF (salpeter or kroupa)
    num_mass_bins : int
        Number of mass bins in logarithmic spaced mass grid.
    random_state : `None`, int, list of ints, or `~numpy.random.RandomState`
        If `None`, return the `~numpy.random.RandomState` singleton used by
        ``numpy.random``. If `int`, return a new `~numpy.random.RandomState`
        instance seeded with the `int`.  If `~numpy.random.RandomState`,
        return it. Otherwise raise ``ValueError``.
    imf_kw : dict, optional
        Keyword arguments for the imf function. 

    Returns
    -------
    masses : ndarray
        The sampled stellar masses.
    """
    rng = check_random_state(random_state)
    bin_edges = np.logspace(np.log10(m_min), 
                            np.log10(m_max), 
                            int(num_mass_bins))
    mass_grid = (bin_edges[1:] + bin_edges[:-1]) / 2.0 
    weights = imf_dict[imf](mass_grid, **imf_kw)
    dm = np.diff(bin_edges)
    cdf = np.cumsum(weights * dm)  
    cdf /= cdf.max()
    cdf = interp1d(cdf, mass_grid, bounds_error=False, fill_value=m_min)
    rand_num = rng.uniform(low=0., high=1.0, size=int(num_stars))
    masses = cdf(rand_num)
    return masses


def build_galaxy(stellar_mass, num_stars_iter=1e5, **kwargs):
    """
    Build galaxy of a given stellar mass.

    Parameters
    ----------
    stellar_mass : float
        Stellar mass of galaxy.
    num_stars_iter : int
        Number of stars to generate at each iteration. Lower this 
        number (at the expense of speed) to get a more accurate total mass. 
	
    Returns
    -------
    stars : ndarray
        Stellar masses of all the stars.
    """
    
    stars = []
    total_mass = 0.0

    while total_mass < stellar_mass:
        new_stars = sample_imf(int(num_stars_iter), **kwargs)
        total_mass += new_stars.sum()
        stars = np.concatenate([stars, new_stars])

    return stars
