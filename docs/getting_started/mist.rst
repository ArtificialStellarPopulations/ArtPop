.. _artpop-mist:

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
