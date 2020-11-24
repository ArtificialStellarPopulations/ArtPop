.. _artpop-start:

===============
Getting Started
===============


Installing with ``pip``
=======================

You can install the latest stable version of `ArtPop` using ``pip``::

    pip install artpop

If you are worried about breaking your Python environment, we recommend 
creating its own `conda 
<https://docs.conda.io/en/latest/miniconda.html>`_
environment. Assuming you have ``conda`` installed::

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

`ArtPop` generates stellar populations by interpolating the `MIST model grids 
<http://waps.cfa.harvard.edu/MIST/model_grids.html#>`_. To simulate photometry 
in a given photometric system, you must first download the `Synthetic 
Photometry <http://waps.cfa.harvard.edu/MIST/model_grids.html#synthetic>`_ 
grid for that photometric system. For convenience, you should 
create an environment variable called ``MIST_PATH`` and save all your 
`MIST` grids in this directory. 

For example, in ``bash``, add this to your
``.bashrc`` file::

    export MIST_PATH='/path/to/your/MIST/grids'

`ArtPop` will use this path by default in all functions that use the 
`MIST` grids. Note, however, that you always have the option to pass a 
different path to these functions. 


Opening Issues
==============

If anything breaks and/or you find any documentation typos, 
please `open an issue 
<https://github.com/ArtificialStellarPopulations/ArtPop/issues>`_ or 
consider :ref:`making a code contribution <contribute>` to help us 
fix it |:smile:|. 


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
