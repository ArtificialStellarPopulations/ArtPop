# Standard library
import os
from functools import partial
from unittest import TestCase

# Third-party
import numpy as np
from astropy import units as u

# Project
from artpop import space


class TestSpace(TestCase):
    """Unit tests for the space module."""

    def setUp(self):
        """Create sampling functions for unit tests."""
        self.num_stars = 1e5
        self.d = 500 * u.kpc
        self.ps = 0.2 * u.arcsec / u.pixel
        self.functions = dict(
            uniform=partial(space.uniform_xy, xy_dim=201, random_state=1985),
            plummer=partial(
                space.plummer_xy, distance=self.d, pixel_scale=self.ps,
                xy_dim=201, random_state=1985),
            sersic=partial(
                space.sersic_xy, distance=self.d, pixel_scale=self.ps,
                xy_dim=201, random_state=1985, num_r_eff=1)
        )

    def test_sampling_functions(self):
        """Test sampling of spatial positions  using pre-defined functions."""
        for name in ['uniform', 'plummer', 'sersic']:
            with self.subTest(name=name):
                xy = self.functions[name](self.num_stars)
                self.assertEqual(self.num_stars, len(xy))
                self.assertLessEqual(abs(xy.mean() - 100), 1)

    def test_grid_sampling(self):
        """Test sampling of spatial postions using the grid sampler."""
        for model in [space.Plummer2D, space.Constant2D]:
            with self.subTest(model=model.__name__):
                xy = space.xy_from_grid(
                    self.num_stars, model(), 201, random_state=1985)
                self.assertEqual(self.num_stars, len(xy))
                self.assertTrue(isinstance(xy[10, 0], np.int64))

