#!/usr/bin/env python 
import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    PATH = '/home/docs/checkouts/readthedocs.org/user_builds/artpop/conda/latest/bin/'
    env = os.environ.copy()
    env['PATH'] = env.get('PATH', "") + ":" + PATH
else:
    env = None


setup(name='artpop',
      version='0.1',
      author='Johnny Greco & Shany Danieli',
      author_email='artpopcode@gmail.com',
      packages=['artpop'],
      url='https://github.com/ArtificialStellarPopulations/ArtPop')
