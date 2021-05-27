# Standard library
import os

# Third-party
import numpy as np
from astropy.table import Table
import pytest

# Project
from artpop.util import data_dir
from artpop.stars import Isochrone, SSP, imf
iso_fn = os.path.join(data_dir, 'feh_m1.00_vvcrit0.4_LSST_10gyr_test_iso')


def test_isochrone():
    iso_table = Table.read(iso_fn, format='ascii')
    mag_cols = ['LSST_u', 'LSST_g', 'LSST_r', 'LSST_i', 'LSST_z', 'LSST_y']
    iso = Isochrone(
        mini=iso_table['initial_mass'],
        mact=iso_table['star_mass'],
        mags=iso_table[mag_cols]
    )
    assert np.isclose(iso.ssp_mag('LSST_i'), 5.690581263723638)
    assert np.isclose(iso.ssp_surviving_mass('salpeter'), 0.7014040567731574)
    m_no_rem = iso.ssp_surviving_mass('salpeter', add_remnants=False)
    assert np.isclose(m_no_rem, 0.580936948329605)
    assert np.isclose(iso.ssp_color('LSST_u', 'LSST_y'), 2.232094935936278)
    assert np.isclose(iso.ssp_sbf_mag('LSST_i'), -1.4292710596351692)


def test_ssp():
    iso_table = Table.read(iso_fn, format='ascii')
    mag_cols = ['LSST_u', 'LSST_g', 'LSST_r', 'LSST_i', 'LSST_z', 'LSST_y']
    iso = Isochrone(
        mini=iso_table['initial_mass'],
        mact=iso_table['star_mass'],
        mags=iso_table[mag_cols]
    )

    ssp = SSP(iso, num_stars=1e5, random_state=1)
    assert np.allclose(ssp.total_mass.value, 42503.7698205372)
    assert np.allclose(ssp.total_initial_live_mass.value, 30038.3808820988)

    ssp_2 = SSP(iso, num_stars=1e5, random_state=11)
    csp = ssp + ssp_2
    assert csp.num_stars == 200000
    assert ssp.total_mass + ssp_2.total_mass == csp.total_mass
    assert np.isclose(csp.mean_mag('LSST_z'), 5.900790585790645)


def test_imf():
    masses = imf.sample_imf(1e5, imf='kroupa', random_state=10)
    assert np.isclose(masses.sum(), 57561.51401879226)
    assert np.isclose(masses.mean(), 0.5756151401879226)

    masses = imf.sample_imf(1e5, imf='salpeter', random_state=9)
    assert np.isclose(masses.sum(), 28755.51602397958)
    assert np.isclose(masses.mean(), 0.2875551602397958)

    masses = imf.sample_imf(1e5, imf='scalo', random_state=8)
    assert np.isclose(masses.sum(), 65983.67121017262)
    assert np.isclose(masses.mean(), 0.6598367121017261)

    assert 1e4 / imf.build_galaxy(1e4, 1e3).sum() >= 0.9
