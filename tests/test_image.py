# Standard library
import os
import pickle
from unittest import TestCase

# Third-party
import numpy as np
from astropy.table import Table

# Project
import artpop


class TestImage(TestCase):
    """Unit tests for the image module."""

    def setUp(self):
        """Create psf and source object for tests."""
        self.psf = artpop.moffat_psf(0.6)
        self.src = artpop.Source(
            xy=np.array([[109.5,  71.7],
                         [ 36.6, 111.7],
                         [ 40.3, 107.7],
                         [166.7, 199.4],
                         [ 86.8, 160.9]]),
            mags=dict(LSST_i=np.array([10.1, 20.4, 30.2, 20.5, 8.2])),
            xy_dim=201
        )

    def test_psf(self):
        """Test the psf module."""
        self.assertIsNone(np.testing.assert_allclose(
            np.array([[0.04451139, 0.11380315, 0.04451139],
                      [0.11380315, 0.36674184, 0.11380315],
                      [0.04451139, 0.11380315, 0.04451139]]),
            artpop.moffat_psf(0.3, shape=3)
            )
        )
        self.assertIsNone(np.testing.assert_allclose(
            np.array([[0.03392846, 0.11633988, 0.03392846],
                      [0.11633988, 0.39892664, 0.11633988],
                      [0.03392846, 0.11633988, 0.03392846]]),
            artpop.gaussian_psf(0.3, shape=3)
            )
        )

    def test_ideal_imager(self):
        """Test the IdealImager class."""
        imager =  artpop.IdealImager()
        obs = imager.observe(self.src, 'LSST_i')
        self.assertEqual((201, 201), obs.image.shape)
        self.assertEqual('LSST_i', obs.bandpass)
        obs_smooth = imager.observe(self.src, 'LSST_i', psf=self.psf)
        self.assertLess((obs_smooth.image == 0).sum(),  (obs.image == 0).sum())

    def test_art_image_filters(self):
        """Test that we have all filter info for ArtImager class."""
        filter_dict = artpop.get_filter_names()
        for phot_system, filters in filter_dict.items():
            with self.subTest(phot_system):
                imager = artpop.ArtImager(phot_system)
                self.assertEqual(filters, imager.filters)

