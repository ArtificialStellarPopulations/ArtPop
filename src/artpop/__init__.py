__version__ = "0.2"

try:
    __ARTPOP_SETUP__
except NameError:
    __ARTPOP_SETUP__ = False

if not __ARTPOP_SETUP__:
    from .filters import *
    from .stars import *
    from .image import *
    from .space import *
    from .source import *
    from .visualization import *
