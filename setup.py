import os
from setuptools import setup, find_packages


def readme():
    with open("README.rst") as f:
        return f.read()

setup(
    name='artpop',
    version='0.0.2',
    description='Building artificial galaxies one star at a time',
    long_description=readme(),
    author='Johnny Greco & Shany Danieli',
    author_email='artpopcode@gmail.com',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    url='https://github.com/ArtificialStellarPopulations/ArtPop', 
    install_requires=[    
        'numpy>=1.17',
        'scipy>=1',
        'astropy>=4',
        'matplotlib>=3',
        'fast-histogram'
     ],
     classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
      ],
    python_requires='>=3.6',
)
