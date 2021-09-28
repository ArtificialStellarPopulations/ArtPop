# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# sys.path.insert(0, os.path.abspath('.'))
import sys
try:
    from sphinx_astropy.conf.v1  import *
except ImportError:
    print('ERROR: the documentation requires the sphinx-astropy package to be installed')
    sys.exit(1)


# -- Project information -----------------------------------------------------

project = 'ArtPop'
copyright = '2021, Johnny Greco and Shany Danieli'
author = 'Johnny Greco and Shany Danieli'


# -- General configuration ---------------------------------------------------

highlight_language = 'python3'
needs_sphinx = '1.7'

check_sphinx_version("1.2.1")

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#extensions = ['sphinx_automodapi.automodapi']
#numpydoc_show_class_members = False

extensions.append('sphinxemoji.sphinxemoji')
extensions.append('nbsphinx')
extensions.append('IPython.sphinxext.ipython_console_highlighting')

plot_formats = [('png', 200)]
plot_include_source = False


# Custom setting for nbsphinx - timeout for executing one cell
nbsphinx_timeout = 300

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
exclude_patterns.append('_templates')

# -- Options for HTML output -------------------------------------------------

html_theme_options = {
    'logotext1': 'Art', # white,  semi-bold
    'logotext2': 'Pop',  # red, light
    'logotext3': ' docs',   # white,  light
}


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_style = 'artpop.css'
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '_static'))
html_favicon = os.path.join(path, 'omega-cen-png.ico')

automodsumm_inherited_members = True

this_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(this_dir)
sys.path.insert(0, top_dir)
