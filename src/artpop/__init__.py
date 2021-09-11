__version__ = "0.2"

try:
    __ARTPOP_SETUP__
except NameError:
    __ARTPOP_SETUP__ = False

if not __ARTPOP_SETUP__:
    import os
    project_dir = os.path.dirname(os.path.dirname(__file__))
    package_dir = os.path.join(project_dir, 'artpop')
    data_dir = os.path.join(package_dir, 'data')
    from .filters import *
    from .stars import *
    from .image import *
    from .space import *
    from .source import *
    from .visualization import *
