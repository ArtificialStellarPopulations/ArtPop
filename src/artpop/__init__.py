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
    MIST_PATH = os.getenv('MIST_PATH')
    if MIST_PATH is None:
        MIST_PATH = os.path.join(os.path.expanduser('~'), '.artpop')
        if not os.path.exists(MIST_PATH):
            os.mkdir(MIST_PATH)
        MIST_PATH = os.path.join(MIST_PATH, 'mist')
        if not os.path.exists(MIST_PATH):
            print('\033[33mWARNING:\033[0m Environment variable MIST_PATH '
                  'does not exist. When you first use a MIST grid, it will be '
                 f'downloaded and saved in {MIST_PATH}. To change this '
                  'location, create a MIST_PATH environment variable.')
            os.mkdir(MIST_PATH)
    from .filters import *
    from .stars import *
    from .image import *
    from .space import *
    from .source import *
    from .visualization import *
