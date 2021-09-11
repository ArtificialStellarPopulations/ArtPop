# Standard library
import os

# Third-party
import numpy as np
from numpy.lib.recfunctions import append_fields
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy import units as u

# Project
from ._read_mist_models import IsoCmdReader, IsoReader
from .imf import imf_dict, IMFIntegrator
from ..log import logger
from ..filters import phot_system_list, get_filter_names
from ..util import MIST_PATH, check_units, fetch_mist_grid_if_needed
phot_str_helper = {p.lower():p for p in phot_system_list}


__all__ = ['fetch_mist_iso_cmd', 'Isochrone', 'MISTIsochrone']


class Isochrone(object):
    """
    Class for storing generic isochrones. It also has methods for calculating
    IMF-weighted properties of a simple stellar population with the given
    age and metallicity.

    Parameters
    ----------
    mini : `~numpy.ndarray`
        Initial stellar masses. The masses must be monotonically increasing.
    mact : `~numpy.ndarray`
        Actual stellar masses after mass loss.
    mag_table : `~astropy.table.Table`, dict, or structured `~numpy.ndarray`
        The stellar magnitudes.
    eep : `~numpy.ndarray`, optional
        The Primary Equivalent Evolutionary Points. Needed to identify phases
        of stellar evolution.
    log_L : `~numpy.ndarray`, optional
        Stellar luminosities.
    log_Teff : `~numpy.ndarray`, optional
        Stellar effective temperatures.
    """

    def __init__(self, mini, mact, mags, eep=None, log_L=None,
                 log_Teff=None):
        self.mini = np.asarray(mini)
        self.mact = np.asarray(mact)
        if (np.diff(self.mini) < 0).sum() > 0:
            raise Exception('Initial masses must be monotonically increasing.')
        self.eep = None if eep is None else np.asarray(eep)
        self.log_L = None if log_L is None else np.asarray(log_L)
        self.log_Teff = None if log_Teff is None else np.asarray(log_Teff)
        if type(mags) == dict or type(mags) == np.ndarray:
            self.mag_table = Table(mags)
        elif type(mags) == Table:
            self.mag_table = mags
        else:
            raise Exception(f'{type(mags)} is not a valid type for mags')

    @property
    def m_min(self):
        """The minimum mass of the isochrone."""
        return self.mini.min()

    @property
    def m_max(self):
        """The maximum mass of the isochrone."""
        return self.mini.max()

    @property
    def filters(self):
        """List of filters in the given photometric system(s)."""
        return self.mag_table.colnames

    @staticmethod
    def from_parsec(file_name, log_age=None, zini=None, num_filters=None):
        """
        Read isochrone generated from the `PARSEC website
        <http://stev.oapd.inaf.it/cgi-bin/cmd>`_. If more than one age and/or
        metallicity is included in the file, you must provide the log_age
        and/or zini parameters.


        Parameters
        ----------
        file_name : str
            Isochrone file name.
        log_age : float, optional
            Log of age in years. You must provided this parameter if there is
            more than one age in the isochrone file. Note this function does
            not interpolate ages, so it must be included in the file.
        zini : float, optional
            Initial metal fraction. You must provided this parameter if there
            is more than one metallicity in the isochrone file. Note this
            function does not interpolate metallicity, so it must be included
            in the file.
        num_filters : int, optional
            Number of filters included in the isochrone file. If None, will
            assume the last non-filter parameter is `mbolmag`.

        Returns
        -------
        iso : `artpop.stars.Isochrone`
            PARSEC Isochrone object.
        """
        if os.path.isfile(file_name):
            file = open(file_name, 'r')
            lines = file.readlines()
            file.close()
            for l in lines:
                if 'Zini' in l.split()[1]:
                    names = l.split()[1:]
                    break
            data = np.loadtxt(file_name)
            parsec = Table(data, names=names)
        else:
            raise Exception(f'{file_name} does not exist.')
        isochrone_full = parsec.copy()
        if log_age is not None:
            age_cut = np.abs(parsec['logAge'] - log_age) < 1e-5
            if age_cut.sum() < 1:
                raise Exception(f'log_age = {log_age} not found.')
            parsec = parsec[age_cut]
        if zini is not None:
            zini_cut = np.abs(parsec['Zini'] - zini) < 1e-8
            if zini_cut.sum() < 1:
                raise Exception(f'Zini = {zini} not found.')
            parsec = parsec[zini_cut]
        if num_filters is None:
            filt_idx = np.argwhere(np.array(names) == 'mbolmag')[0][0] + 1
        else:
            filt_idx = len(names) - num_filters
        iso = Isochrone(mini=parsec['Mini'],
                        mact=parsec['Mass'],
                        mags=parsec[names[filt_idx:]],
                        log_L=parsec['logL'],
                        log_Teff=parsec['logTe'])
        iso.isochrone_full = isochrone_full
        return iso

    def interpolate(self, y_name, x_interp, x_name='mini',
                    slice_interp=np.s_[:], **kwargs):
        """
        Interpolate isochrone.

        Parameters
        ----------
        y_name : str
            Parameter name of interpolated variable.
        x_interp : `~numpy.ndarray`
            New x values to interpolate over.
        x_name : str
            Parameter name of x values. Usually want this to be initial mass.

        Returns
        -------
        y_interp : `~numpy.ndarray`
            Interpolated y values.
        """
        x = self.mag_table[x_name] if x_name in self.filters\
                                   else getattr(self, x_name)
        y = self.mag_table[y_name] if y_name in self.filters\
                                   else getattr(self, y_name)
        x = x[slice_interp]
        y = y[slice_interp]
        y_interp = interp1d(x, y, **kwargs)(x_interp)
        return y_interp

    def mag_to_mass(self, mag, bandpass):
        """
        Interpolate isochrone to calculate initial stellar masses that have
        the given magnitude. Since more than one mass can have the same
        magnitude (from stars being in different phases of stellar evolution),
        we step through each isochrone and check if the given magnitude falls
        between two bins, in which case we interpolate the associated mass.

        Parameters
        ----------
        mag : float
            Magnitude to be converted in to stellar mass(es).
        bandpass : str
            Photometric bandpass of the `mag`.

        Returns
        -------
        mass_interp : `~numpy.ndarray`
            Interpolated mass(es) associated with `mag`.
        """
        y = self.mini
        x = self.mag_table[bandpass]

        if mag < x.min() or mag > x.max():
            raise Exception(f'mag = {mag} is outside the isochrone range.')

        y_interp = []
        # Loop over isochrone and check if desired magnitude is in between two
        # bins. If it is, interpolate the mass. Looping like this is necessary
        # because different masses can have the same magnitude due to
        # being at different phases of stellar evolution.
        for i in range(len(x)):
            if x[i] == mag:
                y_interp.append(y[i])
                continue
            if i < len(x) - 1:
                _x = [x[i], x[i + 1]]
                _y = [y[i], y[i + 1]]
                dx = x[i + 1] - x[i]
                if dx > 0:
                    if (mag <= x[i + 1]) and (mag > x[i]):
                        y_interp.append(interp1d(_x, _y)([mag])[0])
                else:
                    if (mag >= x[i + 1]) and (mag < x[i]):
                        y_interp.append(interp1d(_x, _y)([mag])[0])
        mass_interp = np.array(y_interp)

        return mass_interp

    def calculate_mag_limit(self, imf, bandpass, frac_mass_sampled=None,
                            frac_num_sampled=None, distance=10*u.pc):
        """
        Calculate the limiting faint magnitude to sample a given fraction
        of mass or number of a stellar population. This is used for when you
        only what to sample stars that are brighter than a magnitude limit.
        You must specify either `frac_mass_sampled` or `frac_num_sampled`.

        .. note::
            You must give `frac_mass_sampled` *or* `frac_num_sampled`.

        Parameters
        ----------
        imf : str
            Name for the IMF.
        bandpass : str
            Observation bandpass of the limiting magnitude.
        frac_mass_sampled: float, optional
            Fraction of mass that will be sampled if only stars that are
            brighter than the magnitude limit are sampled.
        frac_num_sampled: float, optional
            Fraction of stars by number that will be sampled if only stars that
            are brighter than the magnitude limit are sampled.
        distance : float or `~astropy.units.Quantity`, optional
            Distance to source. If float is given, the units are assumed
            to be `~astropy.units.Mpc`. Default distance is 10
            `~astropy.units.pc` (i.e., the mags are in absolute units).

        Returns
        -------
        mag_limit : float
            The desired limiting magnitude.
        mass_limit : float
            Mass of stars (in solar masses) that have the limiting magnitude.
        """
        mfint = IMFIntegrator(imf, self.m_min, self.m_max)
        m_vals = np.logspace(np.log10(self.m_min), np.log10(self.m_max), 500)
        err_msg = 'You must give a fraction such that 0 < fraction <= 1.'
        if frac_mass_sampled is not None:
            assert frac_mass_sampled > 0 and frac_mass_sampled <= 1, err_msg
            frac_interp = frac_mass_sampled
            fracs = [mfint.m_integrate(_m, self.m_max, True) for _m in m_vals]
        elif frac_num_sampled is not None:
            assert frac_num_sampled > 0 and frac_num_sampled <= 1, err_msg
            frac_interp = frac_num_sampled
            fracs = [mfint.integrate(_m, self.m_max, True) for _m in m_vals]
        else:
            msg = 'You must give either frac_mass_sampled or frac_num_sampled.'
            raise Exception(msg)
        d = check_units(distance, 'Mpc')
        mass_limit = interp1d(fracs, m_vals)([frac_interp])[0]
        mag_limit = self.interpolate(bandpass, [mass_limit])[0]
        mag_limit = mag_limit + 5 * np.log10(d.to('pc').value) - 5
        return mag_limit, mass_limit

    def nearest_mini(self, m):
        """
        Find the nearest mass to `m` in the initial mass array.

        Parameters
        ----------
        m : float:
            The mass for which we want the nearest initial mass.

        Retruns
        -------
        m_nearest : float
            Value of the nearest mass.
        arg_nearest : int
            Argument of the nearest mass.
        """
        arg_nearest = np.abs(self.mini - m).argmin()
        m_nearest = self.mini[arg_nearest]
        return m_nearest, arg_nearest

    def imf_weights(self, imf, m_min_norm=None, m_max_norm=120,
                    norm_type='mass', **kwargs):
        """
        Calculate IMF weights.

        Parameters
        ----------
        imf : str or callable function that takes mass as its argument
            Stellar Initial mass function.
        m_min_norm : None or float, optional
            Minimum mass for the normalization. Must be less than or equal to
            the mini mass of isochrone, which will be used if None is given.
        m_max_norm : float, optional
            Maximum mass for normalization.
        norm_type : str, optional
            Type of IMF normalization (mass or number).

        Returns
        -------
        wght : `~numpy.ndarray`
            IMF weights calculated such that an integral over mass is simply
            given by SUM(m * wght).
        """
        m_min = m_min_norm if m_min_norm else self.mini.min()
        m_max = m_max_norm if m_max_norm else self.mini.max()
        if m_min > self.m_min:
            raise Exception('Minimum mass must be <= isochrone min mass.')
        if callable(imf):
            imf_func = imf
        else:
            imf_func = lambda m: imf_dict[imf](m, norm_type=None)
        m_imf_func = lambda m: m * imf_func(m)
        norm_func = dict(mass=m_imf_func, number=imf_func)[norm_type]
        norm = quad(norm_func, m_min, m_max, **kwargs)[0]

        wght = []
        mini = self.mini
        # Assume mass is constant in each bin and integrate.
        # This means an integral over mass is simply SUM(m * wght)
        # This trick was stolen from Charlie Conroy's FSPS code :)
        for i in range(len(mini)):
            if i == 0:
                m1 = m_min
            else:
                m1 = mini[i] - 0.5 * (mini[i] - mini[i-1])
            if i == len(mini) - 1:
                m2 = mini[i]
            else:
                m2 = mini[i] + 0.5 * (mini[i+1] - mini[i])
            if m2 < m1:
                raise Exception('Masses must be monotonically increasing.')
            wght.append(quad(imf_func, m1, m2, **kwargs)[0])
        wght = np.array(wght) / norm
        return wght

    def ssp_color(self, blue, red, imf='kroupa', **kwargs):
        """
        Calculate IMF-weighted integrated color.

        Parameters
        ----------
        blue : str
            Blue bandpass.
        red : str
            Red bandpass.
        imf : str, optional
            IMF name.

        Returns
        -------
        color : float
            Integrated color of stellar population.
        """
        wght = self.imf_weights(imf, **kwargs)
        lum_blue = np.sum(wght * 10**(-0.4 * self.mag_table[blue]))
        lum_red = np.sum(wght * 10**(-0.4 * self.mag_table[red]))
        color = -2.5 * np.log10(lum_blue / lum_red)
        return color

    def ssp_sbf_mag(self, bandpass, imf='kroupa', **kwargs):
        """
        Calculate IMF-weighted SBF magnitude.

        Parameters
        ----------
        bandpass : str
            Bandpass to of SBF mag.
        imf : str, optional
            IMF name.

        Returns
        -------
        sbf : float
            SBF magnitude of stellar population.
        """
        wght = self.imf_weights(imf, **kwargs)
        lumlum = np.sum(wght * 10**(-0.8 * self.mag_table[bandpass]))
        lum = np.sum(wght * 10**(-0.4 * self.mag_table[bandpass]))
        sbf = -2.5 * np.log10(lumlum / lum)
        return sbf

    def ssp_mag(self, bandpass, imf='kroupa', norm_type='mass', **kwargs):
        """
        Calculate IMF-weighted magnitude.

        Parameters
        ----------
        bandpass : str
            Bandpass to of mean mag.
        imf : str, optional
            IMF name.
        norm_type : str, optional
            Normalization type (mass or number)

        Returns
        -------
        m : float
            Integrated magnitude of stellar population.
        """
        wght = self.imf_weights(imf, norm_type=norm_type, **kwargs)
        lum = np.sum(wght * 10**(-0.4 * self.mag_table[bandpass]))
        m = -2.5 * np.log10(lum)
        return m

    def ssp_mean_star_mass(self, imf):
        """
        Calculate IMF-weighted mean stellar mass, where the IMF is normalized
        by number to one star formed. The mean stellar mass is equivalent to
        dividing a large of samples (distributed according to the IMF) by the
        number of stars.

        Parameters
        ----------
        imf : str:
            Initial mass function name.

        Returns
        -------
        mean_mass : float
            The mean stellar mass.
        """
        w = self.imf_weights(imf, norm_type='number',
                             m_max_norm=self.mini.max())
        mean_mass = np.sum(self.mact * w)
        return mean_mass

    def ssp_surviving_mass(self, imf, m_min=None, m_max=120,
                           add_remnants=True, mlim_bh=40.0, mlim_ns=8.5):
        """
        Calculate IMF-weighted mass normalized such that 1 solar mass is
        formed by the age of the population.

        The initial-mass-dependent remnant formulae are taken
        from `Renzini & Ciotti 1993
        <https://ui.adsabs.harvard.edu/abs/1993ApJ...416L..49R/abstract>`_.

        Parameters
        ----------
        imf : str
            IMF name.
        add_remnants : bool
            If True, add stellar remnants to surviving mass.

        Returns
        -------
        mass : float
            Total surviving mass.
        """
        m_min = m_min if m_min else self.mini.min()
        m_max = m_max if m_max else self.mini.max()
        wght = self.imf_weights(imf, m_min_norm=m_min, m_max_norm=m_max)
        mass = np.sum(self.mact * wght)

        if add_remnants:
            imf_func = lambda m: imf_dict[imf](m, norm_type=None)
            m_imf_func = lambda m: m * imf_func(m)
            norm = quad(m_imf_func, m_min, m_max)[0]

            # BH remnants
            m_low = max(mlim_bh, self.m_max)
            mass = mass + 0.5 * quad(m_imf_func, m_low, m_max)[0] / norm

            # NS remnants
            if self.m_max <= mlim_bh:
                m_low = max(mlim_ns, self.m_max)
                mass = mass + 1.4 * quad(imf_func, m_low, mlim_bh)[0] / norm

            # WD remnants
            if self.m_max < mlim_ns:
                mmax = self.m_max
                mass = mass + 0.48 * quad(imf_func, mmax, mlim_ns)[0] / norm
                mass = mass + 0.077 * quad(m_imf_func, mmax, mlim_ns)[0] / norm

        return mass


def fetch_mist_iso_cmd(log_age, feh, phot_system, mist_path=MIST_PATH,
                       v_over_vcrit=0.4):
    """
    Fetch MIST isochrone grid.

    Parameters
    ----------
    log_age : float
        Logarithm base 10 of the simple stellar population age in years.
    feh : float
        Metallicity [Fe/H] of the simple stellar population.
    phot_system : str
        Name of the photometric system.
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different
        path from the `MIST_PATH` environment variable.
    v_over_vcrit : float, optional
        Rotation rate divided by the critical surface linear velocity. Current
        options are 0.4 (default) and 0.0.

    Returns
    -------
    iso_cmd : `~numpy.ndarray`
        Structured ``numpy`` array with isochrones and stellar magnitudes.
    """

    v = f'{v_over_vcrit:.1f}'
    ver = 'v1.2'
    p = phot_str_helper[phot_system.lower()]
    path = os.path.join(mist_path, 'MIST_' + ver + f'_vvcrit{v}_' + p)
    sign = 'm' if feh < 0 else 'p'
    fn = f'MIST_{ver}_feh_{sign}{abs(feh):.2f}_afe_p0.0_vvcrit{v}_{p}.iso.cmd'
    fn = os.path.join(path, fn)
    iso_cmd = IsoCmdReader(fn, verbose=False)
    iso_cmd = iso_cmd.isocmds[iso_cmd.age_index(log_age)]
    return iso_cmd


class MISTIsochrone(Isochrone):
    """
    Class for fetching and storing MIST isochrones. It also has several methods
    for calculating IMF-weighted photometric properties of a stellar population
    with the given age an metallicity.

    .. note::
        Currently, the models are interpolated in metallicity but not in age.
        Ages are therefore limited to the age grid of the MIST models. The
        [Fe/H] and log(Age/yr) grids are stored as the private class attributes
        ``_feh_grid`` and ``_log_age_grid``.

    Parameters
    ----------
    log_age : float
        Logarithm base 10 of the simple stellar population age in years.
    feh : float
        Metallicity [Fe/H] of the simple stellar population.
    phot_system : str or list-like
        Name of the photometric system(s).
    mist_path : str, optional
        Path to MIST isochrone grids. Use this if you want to use a different
        path from the ``MIST_PATH`` environment variable.
    v_over_vcrit : float, optional
        Rotation rate divided by the critical surface linear velocity. Current
        options are 0.4 (default) and 0.0.
    """

    # the age grid
    _log_age_grid = np.arange(5.0, 10.3, 0.05)
    _log_age_min = _log_age_grid.min()
    _log_age_max = _log_age_grid.max()

    # the [Fe/H] metallicity grid
    # we have feh <= -4, but using <=3 for interpolation boundary
    _feh_grid = np.concatenate([np.arange(-3.0, -2., 0.5),
                                np.arange(-2.0, 0.75, 0.25)])
    _feh_min = _feh_grid.min()
    _feh_max = _feh_grid.max()

    def __init__(self, log_age, feh, phot_system, mist_path=MIST_PATH,
                 v_over_vcrit=0.4):

        # fetch the mist grid if necessary
        fetch_mist_grid_if_needed(phot_system, v_over_vcrit, mist_path)

        # verify age are metallicity are within model grids
        if log_age < self._log_age_min or log_age > self._log_age_max:
            raise Exception(f'log_age = {log_age} not in range of age grid')
        if feh < self._feh_min or feh > self._feh_max:
            raise Exception(f'feh = {feh} not in range of feh grid')

        self.feh = feh
        self.mist_path = mist_path
        self.phot_system = phot_system
        self.v_over_vcrit = v_over_vcrit

        # use nearest age (currently not interpolating on age)
        age_diff = np.abs(self._log_age_grid - log_age)
        self.log_age = self._log_age_grid[age_diff.argmin()]
        if age_diff.min() > 1e-6:
            logger.debug('Using nearest log_age = {:.2f}'.format(self.log_age))

        # store phot_system as list to allow multiple photometric systems
        if type(phot_system) == str:
            phot_system = [phot_system]

        # fetch first isochrone grid, interpolating on [Fe/H] if necessary
        self._iso_full = self._fetch_iso(phot_system[0])

        # iterate over photometric systems and fetch remaining isochrones
        filter_dict = get_filter_names()
        filters = filter_dict[phot_system[0]].copy()
        for p in phot_system[1:]:
            filt = filter_dict[p].copy()
            filters.extend(filt)
            _iso = self._fetch_iso(p)
            mags = [_iso[f].data for f in filt]
            self._iso_full = append_fields(self._iso_full, filt, mags)

        super(MISTIsochrone, self).__init__(
            mini = self._iso_full['initial_mass'],
            mact = self._iso_full['star_mass'],
            mags = Table(self._iso_full[filters]),
            eep = self._iso_full['EEP'],
            log_L = self._iso_full['log_L'],
            log_Teff = self._iso_full['log_Teff'],
        )

    @property
    def isochrone_full(self):
        """MIST entire isochrone in a structured `~numpy.ndarray`."""
        return self._iso_full

    @staticmethod
    def from_parsec(fn, **kwargs):
        msg = 'PARSEC isochrones do not with MISTIsochrone.'
        raise Exception(msg + ' Use artpop.Isochrone instead.')

    def _fetch_iso(self, phot_system):
        """Fetch MIST isochrone grid, interpolating on [Fe/H] if necessary."""
        if self.feh in self._feh_grid:
            args = [self.log_age, self.feh, phot_system, self.mist_path,
                    self.v_over_vcrit]
            iso = fetch_mist_iso_cmd(*args)
        else:
            iso = self._interp_on_feh(phot_system)
        return iso

    def _interp_on_feh(self, phot_system):
        """Interpolate isochrones between two [Fe/H] grid points."""
        i_feh = self._feh_grid.searchsorted(self.feh)
        feh_lo, feh_hi = self._feh_grid[i_feh - 1: i_feh + 1]

        logger.debug('Interpolating to [Fe/H] = {:.2f} '\
                     'using [Fe/H] = {} and {}'.\
                     format(self.feh, feh_lo, feh_hi))

        mist_0 = fetch_mist_iso_cmd(
            self.log_age, feh_lo, phot_system, self.mist_path)
        mist_1 = fetch_mist_iso_cmd(
            self.log_age, feh_hi, phot_system, self.mist_path)

        y0, y1 = np.array(mist_0.tolist()), np.array(mist_1.tolist())

        x = self.feh
        x0, x1 = feh_lo, feh_hi
        weight = (x - x0) / (x1 - x0)

        len_0, len_1 = len(y0), len(y1)

        # if necessary, extrapolate using trend of the longer array
        if (len_0 < len_1):
            delta = y1[len_0:] - y1[len_0 - 1]
            y0 = np.append(y0, y0[-1] + delta, axis=0)
        elif (len_0 > len_1):
            delta = y0[len_1:] - y0[len_1 - 1]
            y1 = np.append(y1, y1[-1] + delta, axis=0)

        y = y0 * (1 - weight) + y1 * weight
        iso = np.core.records.fromarrays(y.transpose(), dtype=mist_0.dtype)

        return iso
