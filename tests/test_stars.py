# Standard library
import os

# Third-party
import numpy as np
import pytest

# Project
from artpop.filters import phot_system_list, phot_system_lookup
from artpop.util import MIST_PATH
from artpop.stars import MistIsochrone, SSP, imf


if MIST_PATH is not None:
    p = phot_system_list
    mist_dir = os.listdir(MIST_PATH)
    grids = [g for g in mist_dir if g[:4]=='MIST' and g.split('_')[-1] in p]
    has_grids = len(grids) > 0
    if has_grids:
        phot_system = grids[0].split('_')[-1]
        version = float(grids[0].split('_')[1][1:])
        v = float(grids[0].split('_')[2][6:])
else:
    has_grids = False


@pytest.mark.skipif(not has_grids, reason='No MIST grids found')
def test_mist_isochrone():
    mist = MistIsochrone(10, -1, phot_system, version=version, v_over_vcrit=v)
    assert phot_system_lookup(mist.filters[0]) == phot_system
    assert mist.mass_min < mist.mass_max
    assert mist.filters[0] in mist.iso.dtype.names


@pytest.mark.skipif(not has_grids, reason='No MIST grids found')
def test_ssp():
    ssp_1 = SSP(10, -1.12, phot_system, num_stars=1e5, 
                version=version, v_over_vcrit=v)
    ssp_2 = SSP(9, -1.4, phot_system, num_stars=1e5, 
                version=version, v_over_vcrit=v)
    ssp = ssp_1 + ssp_2
    filt = ssp.filters[0]
    assert ssp.num_stars == 2e5
    assert ssp_1.num_stars + ssp_2.num_stars == 2e5
    assert ssp.total_mag(filt) < ssp_1.total_mag(filt)
    assert ssp.total_mag(filt) < ssp_2.total_mag(filt)
    assert ssp.get_phase_mask('MS').sum() > ssp.get_phase_mask('RGB').sum()


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
