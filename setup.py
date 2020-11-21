#!/usr/bin/env python 
import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='artpop',
      version='0.1',
      author='Johnny Greco & Shany Danieli',
      author_email='artpopcode@gmail.com',
      packages=['artpop'],
      url='https://github.com/ArtificialStellarPopulations/ArtPop', 
      install_requires=[    
        'numpy>=1.17',
        'scipy>=1',
        'fast-histogram',
        'astropy>=4',
        'matplotlib>=3'
      ],
)

