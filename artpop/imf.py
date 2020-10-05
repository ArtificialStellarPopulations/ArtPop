import numpy as np
from scipy.interpolate import interp1d
from .utils import check_random_state


__all__ = ['broken_power_law', 'broken_power_law_normed', 'salpeter', 
           'kroupa', 'scalo', 'kroupa_func', 'scalo_func', 'imf_dict',
           'salpeter_func', 'sample_imf', 'build_galaxy']


def salpeter(masses, alpha=2.35, norm_unit='number'):

    weights =  masses**-alpha

    if norm_unit is None:
        return weights

    m_min = masses.min()
    m_max = masses.max()

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


def broken_power_law(masses, alphas, breaks):
    masses = np.asarray(masses)
    a1, a2, a3 = alphas 
    b1, b2 = breaks
    alpha = np.where(masses < b1, a1, np.where(masses < b2, a2, a3))
    m_break = np.where(masses < b2, b1, b2 * (b1 / b2)**(a2 / a3))
    weights = (masses / m_break)**(-alpha)
    return weights


def broken_power_law_normed(masses, alphas, breaks, norm_unit='number', 
                            num_norm_bins=1e5, norm_mass_min=None, 
                            norm_mass_max=None):
    weights = broken_power_law(masses, alphas, breaks)
    if norm_unit ==  'sum':
        norm = weights.sum()
    else:
        m_min = norm_mass_min if norm_mass_min else masses.min() 
        m_max = norm_mass_max if norm_mass_max else masses.max()
        m_integrate = np.linspace(m_min, m_max, int(num_norm_bins))
        w_integrate = broken_power_law(m_integrate, alphas, breaks)
        if norm_unit == 'number':
            norm = np.trapz(w_integrate, m_integrate)
        elif norm_unit == 'mass':
            w_integrate = m_integrate * w_integrate
            norm = np.trapz(w_integrate, m_integrate)
    weights /= norm
    return weights


def kroupa(masses, norm_unit='number', num_norm_bins=1e5, 
           norm_mass_min=None, norm_mass_max=None):
    """
    The Kroupa IMF
    https://ui.adsabs.harvard.edu/abs/2001MNRAS.322..231K/abstract
    """
    alphas = [0.3, 1.3, 2.3]
    breaks = [0.08, 0.5]
    if norm_unit is not None: 
        kw = dict(norm_unit=norm_unit, num_norm_bins=num_norm_bins, 
                  norm_mass_min=norm_mass_min, norm_mass_max=norm_mass_max)
        weights = broken_power_law_normed(masses, alphas, breaks, **kw)
    else:
        weights = broken_power_law(masses, alphas, breaks)
    return weights


def scalo(masses, norm_unit='number', num_norm_bins=1e5):
    """
    The Scalo 1998 IMF
    https://ui.adsabs.harvard.edu/abs/1998ASPC..142..201S/abstract
    """
    alphas = [1.2, 2.7, 2.3]
    breaks = [1, 10]
    if norm_unit is not None: 
        kw = dict(norm_unit=norm_unit, num_norm_bins=num_norm_bins)
        weights = broken_power_law_normed(masses, alphas, breaks, **kw)
    else:
        weights = broken_power_law(masses, alphas, breaks)
    return weights


imf_dict = dict(salpeter=salpeter, kroupa=kroupa, scalo=scalo)
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
        Number of mass bins.
    random_state : None, int, list of ints, or np.random.RandomState
        Random state for random number generator. 

    Returns
    -------
    sampled_masses : ndarray
        The sampled stellar masses
    """
    rng = check_random_state(random_state)
    bin_edges = np.logspace(np.log10(m_min), np.log10(m_max), int(num_mass_bins))
    masses = (bin_edges[1:] + bin_edges[:-1]) / 2.0 
    weights = imf_dict[imf](masses, **imf_kw)
    dm = np.diff(bin_edges)
    cdf = np.cumsum(weights * dm)  
    cdf /= cdf.max()
    cdf = interp1d(cdf, masses, bounds_error=False, fill_value=m_min)
    rand_num = rng.uniform(low=0., high=1.0, size=int(num_stars))
    sampled_masses = cdf(rand_num)
    return sampled_masses


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
