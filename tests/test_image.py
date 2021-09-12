# Standard library
import os
import pickle
from unittest import TestCase

# Third-party
import numpy as np
from astropy import units as u
from astropy.table import Table

# Project
import artpop


class TestImage(TestCase):
    """Unit tests for the image module."""

    def setUp(self):
        """Create psf and source object for tests."""
        self.psf = artpop.moffat_psf(0.6, pixel_scale=0.2)
        self.src = artpop.Source(
            xy=np.array([[109.5,  71.7],
                         [ 36.6, 111.7],
                         [ 40.3, 107.7],
                         [166.7, 199.4],
                         [ 86.8, 160.9]]),
            mags=dict(LSST_i=np.array([10.1, 20.4, 30.2, 20.5, 8.2])),
            xy_dim=201,
            pixel_scale=0.2
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
        self.assertLess((obs_smooth.image == 0).sum(), (obs.image == 0).sum())

    def test_art_imager_filters(self):
        """Test that we have all filter info for ArtImager class."""
        filter_dict = artpop.get_filter_names()
        fn = os.path.join(artpop.data_dir, 'filter_properties.fits')
        filt_prop = Table.read(fn)
        for phot_system, filters in filter_dict.items():
            with self.subTest(phot_system):
                imager = artpop.ArtImager(phot_system)
                self.assertEqual(filters, imager.filters)
                for filt in imager.filters:
                    prop = filt_prop[filt_prop['bandpass'] == filt]
                    self.assertEqual(prop['dlam'], imager.dlam[filt])
                    self.assertEqual(prop['lam_eff'], imager.lam_eff[filt])

    def test_art_imager(self):
        """Test the ArtImager class."""
        imager = artpop.ArtImager(
            'LSST', diameter=2*u.m/np.sqrt(np.pi), random_state=1985)
        self.assertAlmostEqual(1.0, imager.area.value)

        obs = imager.observe(self.src, 'LSST_i', exptime=1*u.min, sky_sb=19)
        self.assertEqual((201, 201), obs.image.shape)
        self.assertGreater(
            0.02, abs(100 * (38869053 -  obs.image.sum()) / obs.image.sum()))
        self.assertGreater(
            0.03, abs(100 * (962 - obs.image.mean()) / obs.image.mean()))

        obs_smooth = imager.observe(
            self.src, 'LSST_i', exptime=1*u.min, sky_sb=19, psf=self.psf)
        self.assertLess(np.std(obs_smooth.image), np.std(obs.image))

        obs_long = imager.observe(
            self.src, 'LSST_i', exptime=1*u.hr, sky_sb=19)
        self.assertAlmostEqual(
            60.0, obs_long.raw_counts.mean() / obs.raw_counts.mean(), 1)
