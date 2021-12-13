# Standard library
import os

import pickle
from unittest import TestCase

# Third-party
import numpy as np

# Project
from artpop.stars import imf


class TestIMF(TestCase):
    """Unit tests for the IMF module."""

    def test_populations(self):
        """Test sampling of initial mass functions."""
        masses = imf.sample_imf(1e5, imf='kroupa', random_state=10)
        self.assertAlmostEqual(57561.5140188, masses.sum())
        self.assertAlmostEqual(0.5756151, masses.mean())

        masses = imf.sample_imf(1e5, imf='salpeter', random_state=9)
        self.assertAlmostEqual(28755.5160241, masses.sum())
        self.assertAlmostEqual(0.2875552, masses.mean())

        masses = imf.sample_imf(1e5, imf='scalo', random_state=8)
        self.assertAlmostEqual(65983.6712102, masses.sum())
        self.assertAlmostEqual(0.6598367, masses.mean())

        self.assertLessEqual(0.9, 1e4 / imf.build_galaxy(1e4, 1e3).sum())

    def test_integration(self):
        """Test integrate with custom broken power law."""
        test_dict = {'a':[-3., -3., -3.], 'b':[1, 10]}
        mfint = imf.IMFIntegrator(test_dict)

        self.assertAlmostEqual(1.5, mfint.integrate(0.5, 1), 6)
        self.assertAlmostEqual(3./8., mfint.integrate(1, 2), 6)
        self.assertAlmostEqual(3./800., mfint.integrate(10, 20), 6)

        self.assertAlmostEqual(1., mfint.m_integrate(0.5,1), 6)
        self.assertAlmostEqual(1./2., mfint.m_integrate(1,2), 6)
        self.assertAlmostEqual(1./20., mfint.m_integrate(10,20), 6)

    def test_weights(self):
        """Test calculating weights with custom broken power law."""

        test_dict = {'a':[-3.,-4.,-5.],'b':[1,10]}

        mfint = imf.IMFIntegrator(test_dict)
        wtest = mfint.weights([0.5,2.,100.])

        self.assertAlmostEqual(8., wtest[0], 6)
        self.assertAlmostEqual(1./16., wtest[1], 6)
        # use log to compare effectively
        self.assertAlmostEqual(-9., np.log10(wtest[2]), 6)
