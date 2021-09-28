# Standard library
from unittest import TestCase

# Third-party
import numpy as np

# Project
from artpop.stars import MISTIsochrone, MISTSSP


class TestMIST(TestCase):
    """Unit tests for the MIST isochrones."""

    def setUp(self):
        """Build single isochrone and ssp objects for all test cases."""
        self.ab = MISTIsochrone(10, -1.5, 'LSST', ab_or_vega='ab')
        self.vega = MISTIsochrone(10, -1.5, 'LSST', ab_or_vega='vega')
        self.rng = np.random.RandomState(1234)
        self.ssp = MISTSSP(10.1, -1, 'LSST',  num_stars=1e4,
                           random_state=self.rng)

    def test_mist_iso(self):
        """Test that the MIST ischrones loaded correctly."""
        self.assertEqual(1464, len(self.ab.eep))
        self.assertEqual('ab', self.ab.ab_or_vega)
        self.assertEqual('vega', self.vega.ab_or_vega)
        diff = self.ab.mag_table['LSST_i'] - self.vega.mag_table['LSST_i']
        self.assertTrue(all([abs(d - 0.363627) < 1e-6 for d in diff]))

    def test_mist_ssp(self):
        """Test MIST-specific SSP methods/attributes."""
        self.assertEqual(9950, self.ssp.select_phase('MS').sum())
        self.assertEqual(['PMS', 'MS', 'giants', 'RGB', 'CHeB',
                          'AGB', 'EAGB', 'TPAGB', 'postAGB', 'WDCS'],
                          self.ssp.phases)
        self.assertGreater(self.ssp.select_phase('RGB').sum(),
                           self.ssp.select_phase('AGB').sum())
        giants = self.ssp.select_phase('AGB').sum()
        giants += self.ssp.select_phase('RGB').sum()
        giants += self.ssp.select_phase('CHeB').sum()
        self.assertEqual(giants, self.ssp.select_phase('giants').sum())

