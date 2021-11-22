# Third-party
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy import units as u

# Project
from ..util import check_random_state, check_units


__all__ = ['IMFIntegrator', 'salpeter_prop', 'kroupa_prop', 'scalo_prop',
            'imf_prop_dict', 'sample_imf', 'build_galaxy',
            'kroupa','scalo','salpeter']


#Define commonly used IMFs

#Can set up similar dict for custom option but need to use 3 segment
#broken power or hack it (see salpeter below)
kroupa_prop = {'a':[-0.3,-1.3,-2.3],'b':[0.08,0.5]}
scalo_prop = {'a':[-1.2,-2.7,-2.3],'b':[1,10]}

#Simple 'hack' to allow for a single power law
salpeter_prop = {'a':[-2.35]*3,'b':[2e2,3e2]}

imf_prop_dict = {'salpeter':salpeter_prop,'kroupa':kroupa_prop,'scalo':scalo_prop}

#Simple wrapper functions to re-create old functions

def kroupa(m,**kw):
    """
    Wrapper function to calculate weights for the Kroupa stellar initial
    mass function
    (`Kroupa 2001 <https://ui.adsabs.harvard.edu/abs/2001MNRAS.322..231K/abstract>`_).
    Parameters
    ----------
    mass_grid : `~numpy.ndarray`
        Stellar mass grid.
    norm_type : str, optional
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
    return IMFIntegrator('kroupa').weights(m, **kw)

def scalo(m,**kw):
    """
    The Scalo stellar initial mass function (`Scalo 1998
    <https://ui.adsabs.harvard.edu/abs/1998ASPC..142..201S/abstract>`_).
    """
    return IMFIntegrator('scalo').weights(m, **kw)

def salpeter(m,**kw):
    """
    Wrapper function to calculate weights for the Salpeter IMF (`Salpeter 1955
    <https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/abstract>`_).
    Parameters
    ----------
    mass_grid : `~numpy.ndarray`
        Stellar mass grid.
    norm_type : str, optional
        How to normalize the weights: by 'number', 'mass', or the 'sum'.
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
    return IMFIntegrator('salpeter').weights(m, **kw)

def sample_imf(num_stars, m_min=0.08, m_max=120, imf='kroupa',
               num_mass_bins=100000, random_state=None, imf_kw={}):
    """
    Sample stellar IMF via inverse transform sampling.

    Parameters
    ----------
    num_stars : int
        Number of stars to sample.
    m_min : float, optional
        Minimum stellar mass.
    m_max : float, optional
        Maximum stellar mass.
    imf : str, optional
        The desired IMF (salpeter or kroupa)
    num_mass_bins : int, optional
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
    masses : `~numpy.ndarray`
        The sampled stellar masses.
    """
    rng = check_random_state(random_state)
    bin_edges = np.logspace(np.log10(m_min),
                            np.log10(m_max),
                            int(num_mass_bins))
    mass_grid = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    weights = IMFIntegrator(imf).weights(mass_grid, **imf_kw)
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
    stellar_mass : float or `~astropy.units.Quantity`
        Stellar mass of galaxy. If float is given, the units are assumed to be
        solar masses.
    num_stars_iter : int
        Number of stars to generate at each iteration. Lower this
        number (at the expense of speed) to get a more accurate total mass.

    Returns
    -------
    stars : `~numpy.ndarray`
        Stellar masses of all the stars.
    """

    stars = []
    total_mass = 0.0
    stellar_mass = check_units(stellar_mass, 'Msun').to('Msun').value

    while total_mass < stellar_mass:
        new_stars = sample_imf(int(num_stars_iter), **kwargs)
        total_mass += new_stars.sum()
        stars = np.concatenate([stars, new_stars])

    return stars


class IMFIntegrator(object):
    """
    A helper class for numerically integrating the IMF.

    Parameters
    ----------
    prop : str or dict
        Which IMF to use, if str then must be one of pre-defined: 'kroupa',
        'scalo' or 'salpeter'. If dict must contain 'a' with 3 elements describing
        the slopes of the broken power law and 'b' with two elements describing
        the locations of the breaks.
    m_min : float, optional
        Minimum stellar mass.
    m_max : float, optional
        Maximum stellar mass.
    eval_min : float, optional
        Technical detail describing where to evaluate the indefinite integral,
        should not need to change this.
    """

    def __init__(self, prop, m_min=0.1, m_max=120.0,eval_min = 1e-5):

        if type(prop) == str:
            if prop in imf_prop_dict.keys():
                prop_dict = imf_prop_dict[prop]
                self.a = prop_dict['a']
                self.b = prop_dict['b']
                self.name = prop
            else:
                raise Exception(f'{prop} is not one of the pre-defined IMFs: ' \
                 + ', '.join(imf_prop_dict.keys()) )

        elif type(prop) == dict:
            if 'a' in prop.keys() and 'b' in prop.keys():
                self.a = prop['a']
                self.b = prop['b']
                self.name = 'custom'
            else:
                raise ValueError("dict must have both 'a' and 'b' for alphas and breaks")
        self.m_min = m_min
        self.m_max = m_max
        self.eval_min = eval_min
        self.num_norm = self.integrate(m_min, m_max, None)
        self.mass_norm = self.m_integrate(m_min, m_max, None)

    def weights(self, mass_grid, norm_type=None,
        norm_mass_min=None, norm_mass_max=None):

        mass_grid = np.asarray(mass_grid)
        a1, a2, a3 = self.a
        b1, b2 = self.b
        alpha = np.where(mass_grid < b1, a1, np.where(mass_grid < b2, a2, a3))
        m_break = np.where(mass_grid < b2, b1, b2 * (b1 / b2)**(a2 / a3))
        weights = (mass_grid / m_break)**(alpha)

        if norm_type is None:
            norm = 1.
        elif norm_type ==  'sum':
            norm = weights.sum()
        else:
            m_min = norm_mass_min if norm_mass_min else mass_grid.min()
            m_max = norm_mass_max if norm_mass_max else mass_grid.max()
            if norm_type == 'number':
                norm = self.integrate(m_min = m_min, m_max = m_max)
            elif norm_type == 'mass':
                norm = self.m_integrate(m_min = m_min, m_max = m_max)
        weights /= norm
        return weights

    def func(self,m):
        return self.weights(m, norm_type = None)

    def m_func(self, m):
        return m * self.weights(m, norm_type=None)

    def _indef_int(self,m):

        a0,a1,a2 = self.a
        b0,b1 = self.b

        #Define constants to normalize functions
        c0 = b0**a0
        c1 = b0**a1
        c2 = (b1 * (b0 / b1)**(a1 / a2) )**a2

        if m > b1:
            return (b0**(a0+1.) - self.eval_min**(a0+1.)) / (a0+1)/c0  \
             + (b1**(a1+1.) - b0**(a1+1.))/(a1+1)/c1  \
             + (m**(a2+1.) - b1**(a2+1.))/(a2+1)/c2
        elif m > b0:
            return (b0**(a0+1.) - self.eval_min**(a0+1.))/(a0+1)/c0 \
             + (m**(a1+1.)- b0**(a1+1.))/(a1+1) /c1
        else:
            return (m**(a0+1.) - self.eval_min**(a0+1.))/(a0+1)/c0

    def integrate(self, m_min=None, m_max=None, norm=False, **kwargs):
        m_min = m_min if m_min else self.m_min
        m_max = m_max if m_max else self.m_max
        if norm == True:
            n = self.num_norm
        elif norm is None or norm == False:
            n = 1.0
        elif type(norm) == int or type(norm) == float:
            n = norm
        else:
            raise Exception(f'{norm} is not a valid normalization.')
        return ( self._indef_int(m_max) - self._indef_int(m_min) ) / n

    def _indef_m_int(self,m):

        a0,a1,a2 = self.a
        b0,b1 = self.b

        #Define constants to normalize functions
        c0 = b0**a0
        c1 = b0**a1
        c2 = (b1 * (b0 / b1)**(a1 / a2) )**a2

        #Multiply by M
        a0 += 1
        a1 += 1
        a2 += 1

        if m > b1:
            return (b0**(a0+1.) - self.eval_min**(a0+1.)) / (a0+1)/c0 \
             + (b1**(a1+1.) - b0**(a1+1.))/(a1+1)/c1 \
             + (m**(a2+1.) - b1**(a2+1.))/(a2+1)/c2
        elif m > b0:
            return (b0**(a0+1.) - self.eval_min**(a0+1.))/(a0+1)/c0 \
             + (m**(a1+1.) - b0**(a1+1.))/(a1+1) /c1
        else:
            return (m**(a0+1.) - self.eval_min**(a0+1.))/(a0+1)/c0

    def m_integrate(self, m_min=None, m_max=None, norm=False, **kwargs):
        m_min = m_min if m_min else self.m_min
        m_max = m_max if m_max else self.m_max
        if norm == True:
            n = self.mass_norm
        elif norm is None or norm == False:
            n = 1.0
        elif type(norm) == int or type(norm) == float:
            n = norm
        else:
            raise Exception(f'{norm} is not a valid normalization.')
        return ( self._indef_m_int(m_max) - self._indef_m_int(m_min) ) / n
