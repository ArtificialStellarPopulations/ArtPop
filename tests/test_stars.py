# Standard library
import os
import pickle
from unittest import TestCase

# Third-party
from astropy.table import Table

# Project
from artpop import data_dir
from artpop.stars import Isochrone, SSP, imf
iso_fn = os.path.join(data_dir, 'feh_m1.00_vvcrit0.4_LSST_10gyr_test_iso.pkl')


class TestStars(TestCase):
    """Unit tests for the stars module."""

    def setUp(self):
        """Build single isochrone object for all test cases."""
        with open(iso_fn, 'rb') as f:
            iso_table = Table(pickle.load(f))
        self.filters = ['LSST_u', 'LSST_g', 'LSST_r',
                        'LSST_i', 'LSST_z', 'LSST_y']
        self.iso = Isochrone(
            mini=iso_table['initial_mass'],
            mact=iso_table['star_mass'],
            mags=iso_table[self.filters]
        )

    def test_imf(self):
        """Test sampling of initial mass functions."""
        masses = imf.sample_imf(1e5, imf='kroupa', random_state=10)
        self.assertAlmostEqual(57561.5140188, masses.sum())
        self.assertAlmostEqual(0.5756151, masses.mean())

        masses = imf.sample_imf(1e5, imf='salpeter', random_state=9)
        self.assertAlmostEqual(28755.5160240, masses.sum(),3)
        self.assertAlmostEqual(0.2875552, masses.mean())

        masses = imf.sample_imf(1e5, imf='scalo', random_state=8)
        self.assertAlmostEqual(65983.6712102, masses.sum(),3)
        self.assertAlmostEqual(0.6598367, masses.mean())

        self.assertLessEqual(0.9, 1e4 / imf.build_galaxy(1e4, 1e3).sum())

    def test_isochrone_init(self):
        """Test initialization of Isochrone object."""
        self.assertEqual(1467, len(self.iso.mag_table))
        self.assertEqual(0.1, self.iso.m_min)
        self.assertEqual(self.filters, self.iso.filters)

    def test_isochrone_calculations(self):
        """Test isochrone calculations of SSP integrated parameters."""
        self.assertAlmostEqual(4.7117543, self.iso.ssp_mag('LSST_i'))
        self.assertAlmostEqual(-1.4292711, self.iso.ssp_sbf_mag('LSST_i'))
        self.assertAlmostEqual(2.2320949,
            self.iso.ssp_color('LSST_u', 'LSST_y'))
        self.assertAlmostEqual(0.7014041,
            self.iso.ssp_surviving_mass('salpeter'))
        self.assertAlmostEqual(0.5809369,
            self.iso.ssp_surviving_mass('salpeter', add_remnants=False))

    def test_ssp(self):
        """Test SSP objects."""
        ssp = SSP(self.iso, num_stars=1e5, random_state=1)
        self.assertAlmostEqual(42503.7698205, ssp.total_mass.value, 3)
        self.assertAlmostEqual(
            30038.3808821, ssp.total_initial_live_mass.value, 3)
        ssp_2 = SSP(self.iso, num_stars=1e5, random_state=11)
        csp = ssp + ssp_2
        self.assertEqual(200000, csp.num_stars)
        self.assertEqual(ssp.total_mass + ssp_2.total_mass, csp.total_mass)
        self.assertAlmostEqual(5.9007906, csp.mean_mag('LSST_z'))

    def test_add_extinction(self):
        """Test that extinction is applied to SSPs correctly."""
        ssp_no_ext = SSP(self.iso, num_stars=1e6, random_state=1)

        ssp_with_ext = SSP(self.iso, num_stars=1e6, random_state=1, a_lam=1.0)
        self.assertDictEqual(ssp_with_ext.a_lam, {f'LSST_{f}': 1.0 for f in 'ugrizy'})

        m_no_ext = ssp_no_ext.total_mag('LSST_i')
        m_with_ext = ssp_with_ext.total_mag('LSST_i')
        self.assertAlmostEqual(1.0, m_with_ext - m_no_ext)

        a_lam = dict(LSST_i=2.0, LSST_g=1.0)
        ssp_with_ext = SSP(self.iso, num_stars=1e6, random_state=1, a_lam=a_lam.copy())
        a_lam.update({f'LSST_{f}': 0.0 for f in 'urzy'})
        self.assertDictEqual(ssp_with_ext.a_lam, a_lam)

        m_with_ext = ssp_with_ext.total_mag('LSST_i')
        self.assertAlmostEqual(2.0, m_with_ext - m_no_ext)

        m_no_ext_g = ssp_no_ext.total_mag('LSST_g')
        m_with_ext_g = ssp_with_ext.total_mag('LSST_g')
        self.assertAlmostEqual(1.0, m_with_ext_g - m_no_ext_g)
