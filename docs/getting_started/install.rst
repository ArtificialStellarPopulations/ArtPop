.. _artpop-install:

============
Installation
============


Installing with ``pip``
=======================

You can install the latest stable version of `ArtPop` using ``pip``::

    pip install artpop

If you are worried about breaking your Python environment, we recommend 
creating a new `conda environment 
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ 
for `ArtPop`. Assuming you have `conda 
<https://docs.conda.io/projects/conda/en/latest/index.html>`_ installed::

    conda create -n artpop pip
    conda activate artpop
    pip install artpop

Installing the Development Version
==================================

If you're feeling adventurous, you can download the development 
version of `ArtPop` from `GitHub <https://github.com/>`_ using ``git``::
    
    git clone git://github.com/ArtificialStellarPopulations/ArtPop.git

Then ``cd`` into the repo directory and install using::

    python setup.py install


MIST Isochrone Grids
====================

In order to generate stellar populations with `ArtPop`, you must download a set of 
MIST isochrone grids, as we describe :ref:`here <artpop-mist>`.


Python Dependencies  
===================

`ArtPop` has the following python dependencies, which are automatically 
installed by ``pip``:

- `python <https://www.python.org/>`_ >= 3.6
- `numpy <https://numpy.org/>`_ >= 1.17
- `scipy <https://www.scipy.org/>`_ > 1 
- `astropy`_ >= 4
- `matplotlib <https://matplotlib.org/>`_ >= 3
- `fast-histogram <https://github.com/astrofrog/fast-histogram>`_
