import numpy as np
from copy import deepcopy
from astropy import units as u
from astropy.table import Table, vstack
from .utils import check_units, check_xy_dim
from .stars import SSP
from .space import sersic_xy, plummer_xy
from . import MIST_PATH


__all__ = ['Source', 'SersicSSP', 'PlummerSSP']


class Source(object):

    def __init__(self, xy, mags, xy_dim, pixel_scale=0.2, labels=None):
        if type(mags) == dict:
            self.mags = Table(mags)
        else:
            self.mags = mags.copy()
        if len(xy) != len(self.mags):
            raise Exception('numbers of magnitudes and positions must match')
        if type(xy) == np.ma.MaskedArray:
            mask = np.any(~xy.mask, axis=1)
            self.mags = self.mags[mask]
            self.xy = xy.data[mask]
        else:
            self.xy = np.asarray(xy)
        self.labels = labels
        self.xy_dim = check_xy_dim(xy_dim)
        self.pixel_scale = check_units(pixel_scale, u.arcsec / u.pixel)

    @property
    def x(self):
        return self.xy[:, 0]

    @property
    def y(self):
        return self.xy[:, 1]

    @property
    def num_stars(self):
        return len(self.x)

    def __add__(self, src):
        if Source not in src.__class__.__mro__:
            raise Exception(f'{type(src)} is not a valid Source object')
        new = deepcopy(self)
        if new.labels is None:
            new.labels = np.ones(new.num_stars, dtype=int)
        labels = np.ones(src.num_stars, dtype=int) * (new.labels.max() + 1)
        new_labels = np.concatenate([new.labels, labels])
        new.xy = np.concatenate([new.xy, src.xy])
        new.mags = vstack([new.mags, src.mags])
        return CompositeSource(new.xy, new.mags, new_labels)


class CompositeSource(Source):
    pass


class SersicSSP(Source):

    def __init__(self, log_age, feh, phot_system, r_eff, n, theta, ellip, 
                 distance, xy_dim, total_mass=None, num_stars=None, 
                 imf='kroupa', mist_path=MIST_PATH, imf_kw={}, pixel_scale=0.2, 
                 num_r_eff=10, random_state=None):

        self.ssp_kw = dict(log_age=log_age, feh=feh, phot_system=phot_system,
                           distance=distance, total_mass=total_mass, 
                           num_stars=num_stars, imf=imf, mist_path=mist_path,
                           imf_kw=imf_kw, random_state=random_state)
        self.ssp = SSP(**self.ssp_kw)

        self.xy_kw = dict(num_stars=self.ssp.num_stars, r_eff=r_eff, n=n,
                          theta=theta, ellip=ellip, distance=distance, 
                          xy_dim=xy_dim, num_r_eff=num_r_eff, 
                          pixel_scale=pixel_scale, random_state=self.ssp.rng)
        _xy = sersic_xy(**self.xy_kw)

        super(SersicSSP, self).__init__(_xy, self.ssp.mag_table, xy_dim)


class PlummerSSP(Source):

    def __init__(self, log_age, feh, phot_system, scale_radius, distance, 
                 xy_dim, total_mass=None, num_stars=None, imf='kroupa', 
                 mist_path=MIST_PATH, imf_kw={}, pixel_scale=0.2, 
                 random_state=None):

        self.ssp_kw = dict(log_age=log_age, feh=feh, phot_system=phot_system,
                           distance=distance, total_mass=total_mass, 
                           num_stars=num_stars, imf=imf, mist_path=mist_path,
                           imf_kw=imf_kw, random_state=random_state)
        self.ssp = SSP(**self.ssp_kw)

        self.xy_kw = dict(num_stars=self.ssp.num_stars, 
                          scale_radius=scale_radius, distance=distance, 
                          xy_dim=xy_dim, pixel_scale=pixel_scale, 
                          random_state=self.ssp.rng)
        _xy = plummer_xy(**self.xy_kw)

        super(PlummerSSP, self).__init__(_xy, self.ssp.mag_table, xy_dim)
