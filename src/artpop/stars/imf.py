# Third-party
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy import units as u
from typing import Iterable

# Project
from ..util import check_random_state, check_units


__all__ = ['IMFIntegrator', 'salpeter_params', 'kroupa_params', 'scalo_params',
            'imf_params_dict', 'sample_imf', 'build_galaxy',
            'kroupa','scalo','salpeter']


# define commonly used IMFs

# pre-defined IMF parameter dictionaries
kroupa_params = {'a':[-0.3,-1.3,-2.3],'b':[0.08,0.5]}
scalo_params = {'a':[-1.2,-2.7,-2.3],'b':[1,10]}

#Simple 'hack' to allow for a single power law
salpeter_params = {'a':-2.35,'b':[2e2,3e2]}

imf_params_dict = {'salpeter': salpeter_params, 'kroupa': kroupa_params, 'scalo': scalo_params}

#Simple wrapper functions to re-create old functions

def kroupa(m, **kwargs):
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
    return IMFIntegrator('kroupa').weights(m, **kwargs)

def scalo(m, **kwargs):
    """
    The Scalo stellar initial mass function (`Scalo 1998
    <https://ui.adsabs.harvard.edu/abs/1998ASPC..142..201S/abstract>`_).

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
    return IMFIntegrator('scalo').weights(m, **kwargs)

def salpeter(m, **kwargs):
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
    return IMFIntegrator('salpeter').weights(m, **kwargs)

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
    imf : str or dict
        Which IMF to use, if str then must be one of pre-defined: 'kroupa',
        'scalo' or 'salpeter'. Can also specify custom (broken) power law as dict,
        which must contain either 'a' as a Float (describing the slope of a
        single power law) or 'a' (a list with 3 elements describing the slopes
        of a broken power law) and 'b' (a list  with 2 elements describing the
        locations of the breaks).
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
    params : str or dict
        Which IMF to use, if str then must be one of pre-defined: 'kroupa',
        'scalo' or 'salpeter'. Can also specify custom (broken) power law as dict,
        which must contain either 'a' as a Float (describing the slope of a
        single power law) or 'a' (a list with 3 elements describing the slopes
        of a broken power law) and 'b' (a list  with 2 elements describing the
        locations of the breaks).
    m_min : float, optional
        Minimum stellar mass.
    m_max : float, optional
        Maximum stellar mass.
    """

    def __init__(self, params, m_min=0.1, m_max=120.0):

        if type(params) == str:
            if params in imf_params_dict.keys():
                params_dict = imf_params_dict[params]
                if params == 'salpeter':
                    self.a = [ params_dict['a'] ]*3
                    self.b = [301.,302.]
                else:
                    self.a = params_dict['a']
                    self.b = params_dict['b']
                self.name = params
            else:
                raise Exception(f'{params} is not one of the pre-defined IMFs: ' \
                 + ', '.join(imf_params_dict.keys()) )

        elif type(params) == dict:
            if 'a' in params.keys() and 'b' in params.keys() and isinstance(params['a'],Iterable):
                self.a = params['a']
                self.b = params['b']
                self.name = 'custom'
            elif 'a' in params.keys() and isinstance(params['a'],float):
                self.a = [ params['a'] ]*3
                self.b = [301.,302.]
                self.name = 'custom'
            else:
                raise Exception("dict must have both 'a' and 'b' for broken power law or float in 'a' for single power law")
        self.m_min = m_min
        self.m_max = m_max
        self.eval_min = 1e-3
        self.num_norm = self.integrate(m_min, m_max, None)
        self.mass_norm = self.m_integrate(m_min, m_max, None)

    def weights(self, mass_grid, norm_type=None,
        norm_mass_min=None, norm_mass_max=None):
        """
        Calculate the weights of the IMF at grid of stellar masses.

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
        """
        Wrapper function to calculate the un-normalized values of
        of the IMF (i.e. dN / dM ) at grid of stellar masses.

        Parameters
        ----------
        mass_grid : `~numpy.ndarray`
            Stellar mass grid.

        Returns
        -------
        weights : `~numpy.ndarray`
            The values of dN/dM associated with each mass in the input `mass_grid`.
        """

        return self.weights(m, norm_type = None)

    def m_func(self, m):
        """
        Wrapper function to calculate the un-normalized values of
        the mass times the IMF (i.e. dN / d logM ) at grid of stellar masses.

        Parameters
        ----------
        mass_grid : `~numpy.ndarray`
            Stellar mass grid.

        Returns
        -------
        weights : `~numpy.ndarray`
            The values dN/d logM associated with each mass in the input `mass_grid`.
        """

        return m * self.weights(m, norm_type=None)

    def _indef_int(self,m):
        """
        Helper function to calculate integral for `0` to some mass
        """

        a0,a1,a2 = self.a
        b0,b1 = self.b

        #Define constants to normalize functions
        c0 = b0**a0
        c1 = b0**a1
        c2 = (b1 * (b0 / b1)**(a1 / a2) )**a2

        ans = 0
        if m > b1:
            ans = (b0**(a0+1.) - self.eval_min**(a0+1.)) / (a0+1)/c0  \
             + (b1**(a1+1.) - b0**(a1+1.))/(a1+1)/c1  \
             + (m**(a2+1.) - b1**(a2+1.))/(a2+1)/c2
        elif m > b0:
            ans = (b0**(a0+1.) - self.eval_min**(a0+1.))/(a0+1)/c0 \
             + (m**(a1+1.)- b0**(a1+1.))/(a1+1) /c1
        else:
            ans = (m**(a0+1.) - self.eval_min**(a0+1.))/(a0+1)/c0
        return ans

    def integrate(self, m_min=None, m_max=None, norm=False, ):
        """"
        Function to calculate the integral under the IMF.

        Parameters
        ----------
        m_min : Float
            Lower stellar mass bound of integral.
        m_max : Float
            Upper stellar mass bound of integral.
        norm: : Bool or Float
            Whether or not to normalize the inegral, default False. If True
            will normalize by number of stars. If a Float is given, then will use
            that value as the normalization.

        Returns
        -------
        weights : Float
            Value of the integral of the IMF between m_min and m_mass.
        """

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
        """
        Helper function to calculate integral for `0` to some mass
        """

        a0, a1, a2 = self.a
        b0, b1 = self.b

        # define constants to normalize functions
        c0 = b0**a0
        c1 = b0**a1
        c2 = (b1 * (b0 / b1)**(a1 / a2))**a2

        # multiply by M
        a0 += 1
        a1 += 1
        a2 += 1

        ans = 0

        if m > b1:
            ans = (b0**(a0+1.) - self.eval_min**(a0+1.)) / (a0+1)/c0 \
             + (b1**(a1+1.) - b0**(a1+1.))/(a1+1)/c1 \
             + (m**(a2+1.) - b1**(a2+1.))/(a2+1)/c2
        elif m > b0:
            ans = (b0**(a0+1.) - self.eval_min**(a0+1.))/(a0+1)/c0 \
             + (m**(a1+1.) - b0**(a1+1.))/(a1+1) /c1
        else:
            ans = (m**(a0+1.) - self.eval_min**(a0+1.))/(a0+1)/c0

        return ans

    def m_integrate(self, m_min=None, m_max=None, norm=False):
        """"
        Function to calculate the integral under mass times the IMF.

        Parameters
        ----------
        m_min : Float
            Lower stellar mass bound of integral.
        m_max : Float
            Upper stellar mass bound of integral.
        norm: : Bool or Float
            Whether or not to normalize the integral, default False. If True
            will normalize by the total stelllar mass. If a Float is given,
            then will use that value as the normalization.

        Returns
        -------
        weights : Float
            Value of the integral of mass times the IMF between m_min and m_mass.
        """
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
        return (self._indef_m_int(m_max) - self._indef_m_int(m_min)) / n
