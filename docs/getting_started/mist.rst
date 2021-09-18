.. _artpop-mist:

MIST Isochrone Grids
====================

`ArtPop` can work with synthetic photometry generated from any source, 
provided it is in the correct format. For convenience, we provide built-in tools for 
fetching and manipulating synthetic photometry from the 
`MIST isochrone models <http://waps.cfa.harvard.edu/MIST/>`_. 
See, for example, the :class:`~artpop.stars.MISTSSP` and :class:`~artpop.source.MISTSersicSSP`
classes.

The first time you need a particular 
`MIST synthetic photometry grid <http://waps.cfa.harvard.edu/MIST/model_grids.html>`_, 
it will be downloaded and saved to your ``MIST_PATH``. If this environment variable is not set, 
the grid(s) will be saved in ``~/.artpop/mist``. 

To change the default path location, create an environment variable called ``MIST_PATH``.

For example, in ``bash``, add this to your
``.bashrc`` file::

    export MIST_PATH='/path/to/your/MIST/grids'

`ArtPop` will use this path by default in all functions that use the
MIST grids. Note, however, that you always have the option to pass a
different path to any function that uses a MIST grid.
