import abc
import six
import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.modeling.models import Sersic2D
from astropy.convolution import convolve_fft
from fast_histogram import histogram2d
from .isochrones import MistIsochrone
from .stellar_pops import SSP, MultiSSP
from .utils import check_random_state
from .log import logger
from . import MIST_PATH

__all__ = ['pixel_scales', 'ArtPopSersic', 'ArtPopBox', 'ArtPopPlummer']


pixel_scales = dict(HST=0.05, SDSS=0.4, HSC=0.168, LSST=0.2, HST_ACS=0.05)


@six.add_metaclass(abc.ABCMeta)
class ArtPopBase(object):
    """
    Base class for building artificial stellar pops.
    """

    def __init__(self, log_age=10.0, feh=-1.0, total_mass=None, 
                 phot_system='SDSS', imf='kroupa', 
                 num_stars=None, pixscale=None, random_state=None, 
                 mist_path=MIST_PATH, imf_kw={}, multi_ssp=None): 

        self.rng = check_random_state(random_state)

        if multi_ssp is not None:
            assert type(multi_ssp) == MultiSSP
            self.pop = multi_ssp 
            self.pop_type = 'MultiSSP'
            self.phot_system = multi_ssp.phot_system
        else:
            self.pop_type = 'SSP'
            self.pop = SSP(log_age, feh, phot_system, total_mass, 
                           num_stars, imf, mist_path, imf_kw=imf_kw, 
                           random_state=self.rng)
            self.feh = feh
            self.log_age = log_age
            self.phot_system = phot_system
            self.imf = imf
            self.imf_kw = imf_kw

        if pixscale is None:
            self.pixscale = pixel_scales[self.phot_system]
        else:
            self.pixscale = pixscale 

    def inject_into_image(self, band, zpt, psf=None, boundary='wrap', 
                          phase='all', A_lam=0):

        if phase != 'all':
            phase_mask = self.pop.get_phase_mask(phase)
            _x = self.star_x[phase_mask]
            _y = self.star_y[phase_mask]
            masses = self.pop.star_masses[phase_mask]
        else:
            _x = self.star_x
            _y = self.star_y
            masses = self.pop.star_masses

        if A_lam != 0:
            logger.warning('Applying extinction with A_{} = {}'.\
                   format(band, A_lam))

        mags = self.pop.star_mags(band) + A_lam
        mags += self.dist_mod

        flux = 10**(0.4 * (zpt - mags))
        bins = tuple(self.xy_image_dim.astype(int))
        hist_range = [[0, self.xy_image_dim[0]-1], [0, self.xy_image_dim[1]-1]]
        image = histogram2d(_x, _y, bins=self.xy_image_dim, 
                             weights=flux, range=hist_range).T

        if psf is not None:
            image = convolve_fft(image, psf, boundary=boundary,
                                 normalize_kernel=True)
                         
        return image

    @abc.abstractmethod
    def generate_star_positions(self):
        return NotImplementedError()


class ArtPopBox(ArtPopBase):
    """
    Build a stellar population with some surface brightness in a box.

    Notes
    -----
    For this class, the stellar population is sampled each time you create 
    a box, since the number of stars depends on the distance and surface 
    brightness.
    """

    def __init__(self,  mu, distance, xy_image_dim, ref_band, log_age=10.0, 
                 feh=-1.0, phot_system='SDSS', imf='kroupa', pixscale=None,  
                 random_state=None, mist_path=MIST_PATH, imf_kw={}, 
                 mean_mag=None, **kwargs):

        self.mu = mu
        self.distance = distance
        if type(xy_image_dim) == int:
            xy_image_dim = (xy_image_dim, xy_image_dim)
        self.xy_image_dim = np.asarray(xy_image_dim)
        self.dist_mod = 5 * np.log10(distance * 10**6) - 5
        npix = xy_image_dim[0] * xy_image_dim[1]

        if mean_mag is None:
            logger.warning('Calculating mean magnitude. You can pass as arg.')
            num_mass_bins = kwargs.pop('num_mass_bins', 5e7)
            mist = MistIsochrone(log_age, feh, phot_system, mist_path)
            self.mean_abs_mag =  mist.mean_mag(
                ref_band, imf=imf, imf_kw=imf_kw, num_mass_bins=num_mass_bins)
        else:
            self.mean_abs_mag = mean_mag

        self.pixscale = pixscale if pixscale else pixel_scales[phot_system]
        num_stars = self.nstar_per_pix() * npix

        super(ArtPopBox, self).__init__(log_age=log_age, 
                                        feh=feh, 
                                        phot_system=phot_system,
                                        imf=imf, 
                                        pixscale=self.pixscale, 
                                        random_state=random_state,
                                        mist_path=mist_path,
                                        num_stars=num_stars, 
                                        imf_kw=imf_kw)
                                        

    def nstar_per_pix(self):
        mean_mag = self.mean_abs_mag + self.dist_mod
        num_stars_per_arsec_sq = 10**(0.4 * (mean_mag -  self.mu))
        num_stars_per_pix = num_stars_per_arsec_sq * self.pixscale**2
        return num_stars_per_pix

    def generate_star_positions(self):
        x = self.rng.randint(0, self.xy_image_dim[1], 
                              len(self.pop.star_masses))
        y = self.rng.randint(0, self.xy_image_dim[0], 
                              len(self.pop.star_masses))
        self.star_x, self.star_y = x, y


class ArtPopSersic(ArtPopBase):
    """
    Build Sersic galaxies with Artificial stellar Populations.
    """

    def generate_star_positions(self, xy_image_dim=501, distance=20.0,    
                                sersic_n=0.8, ellip=0.3, 
                                theta=0.5, r_eff=2.5, num_r_eff=10):

        if type(xy_image_dim) == int:
            xy_image_dim = np.array([xy_image_dim, xy_image_dim])
        else:
            xy_image_dim = np.asarray(xy_image_dim)

        self.xy_image_dim = xy_image_dim

        assert np.all(self.xy_image_dim % 2 != 0), 'dimensions must be odd'

        self.distance = distance
        self.dist_mod = 5 * np.log10(distance * 10**6) - 5
        self.sersic_n = sersic_n
        self.ellip = ellip
        self.theta = theta * np.pi/180.0
        self.r_eff = np.arctan2(r_eff, distance*10**3) / self.pixscale
        self.r_eff *= u.radian.to('arcsec') 

        sample_side = 2 * np.ceil(self.r_eff * num_r_eff).astype(int) + 1
        self.sample_side = sample_side
        x_0, y_0 = sample_side//2, sample_side//2

        self.model = Sersic2D(
            amplitude=1, r_eff=self.r_eff, n=sersic_n, x_0=x_0, y_0=y_0, 
            ellip=ellip, theta=self.theta)

        yy, xx = np.meshgrid(np.arange(sample_side), 
                             np.arange(sample_side))

        image = self.model(xx, yy)
        sersic_weights = image.flatten()
        sersic_weights /= np.sum(sersic_weights)
        rand_num = self.rng.choice(len(sersic_weights), 
                                    size=self.pop.num_stars, 
                                    p=sersic_weights)
        x, y = np.unravel_index(
            rand_num, (sample_side, sample_side), order='C')

        shift = (sample_side - xy_image_dim) // 2
        x -= shift[0]
        y -= shift[1]
        x_0 -= shift[0]
        y_0 -= shift[1]

        if xy_image_dim[0] < sample_side or xy_image_dim[1] < sample_side:
            outside_image = x < 0
            outside_image |= x > xy_image_dim[0] - 1
            outside_image |= y < 0
            outside_image |= y > xy_image_dim[1] - 1 

            if outside_image.sum() > 0:
                msg = '{} stars outside the image'.format(outside_image.sum())
                logger.warning(msg)
                self.x_outside = x[outside_image]
                self.y_outside = y[outside_image]
                self.mass_outside = self.pop.star_masses[outside_image]

                x = x[~outside_image]
                y = y[~outside_image]
                self.pop.num_stars = len(x)
                self.pop.star_masses = self.pop.star_masses[~outside_image]
                self.pop.eep = self.pop.eep[~outside_image]

                if self.pop_type == 'MultiSSP':
                    for filt in self.pop.filters:
                        setattr(self.pop, filt, 
                                getattr(self.pop, filt)[~outside_image])
                else:
                    self.pop.log_L = self.pop.log_L[~outside_image]
                    self.pop.log_Teff = self.pop.log_Teff[~outside_image]

        self.x_0= x_0
        self.y_0= y_0
        self.star_x = x
        self.star_y = y


class ArtPopPlummer(ArtPopBase):
    """
    Build Plummer spheres (for GCs) with Artificial stellar Populations.
    """

    def generate_star_positions(self, xy_image_dim=501, distance=20.0, 
                                scale_radius=2.8, r_max=5):
            
        if type(xy_image_dim) == int:
            xy_image_dim = np.array([xy_image_dim, xy_image_dim])
        else:
            xy_image_dim = np.asarray(xy_image_dim)

        self.xy_image_dim = xy_image_dim

        assert np.all(self.xy_image_dim % 2 != 0), 'dimensions must be odd'

        self.distance = distance
        self.dist_mod = 5 * np.log10(distance * 10**6) - 5
        self.scale_radius = scale_radius        
        self.scale_radius = np.arctan2(scale_radius, distance*10**3) / self.pixscale
        self.scale_radius *= u.radian.to('arcsec') 
        self.r_max = r_max

        m = np.random.random(size=self.pop.num_stars)
        r = self.scale_radius / np.sqrt(m**(-2/3) - 1)

        phi = np.random.uniform(0, 2*np.pi, size=len(r))
        theta = np.arccos(2*np.random.random(size=len(r)) - 1)
        xyz = np.zeros((len(r), 3))
        xyz[:,0] = r * np.cos(phi) * np.sin(theta)
        xyz[:,1] = r * np.sin(phi) * np.sin(theta)
        xyz[:,2] = r * np.cos(theta)

        x = xyz[:,0]
        y = xyz[:,1]

        shift = xy_image_dim // 2
        x += shift[0]
        y += shift[1]

        x_outside = (x < 0) | (x >= xy_image_dim[0])
        y_outside = (y < 0) | (y >= xy_image_dim[0])

        if (x_outside.sum() > 0) or (y_outside.sum() > 0):
            outside = x_outside | y_outside
            x = x[~outside]
            y = y[~outside]
            self.pop.num_stars = len(x)
            self.pop.eep = self.pop.eep[~outside]
            self.pop.star_masses = self.pop.star_masses[~outside]
            self.pop.log_L = self.pop.log_L[~outside]
            self.pop.log_Teff = self.pop.log_Teff[~outside]

        self.star_x = x
        self.star_y = y
