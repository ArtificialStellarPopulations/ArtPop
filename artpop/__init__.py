import os


on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    MIST_PATH = 'MIST_PATH'
else:
    MIST_PATH = os.getenv('MIST_PATH')
    if MIST_PATH is None:
        print('\033[33mWARNING:\033[0m Environment variable MIST_PATH does not '
              'exist. You will need to pass the path to the MIST grids to all '
              'functions that use them.')
project_dir = os.path.dirname(os.path.dirname(__file__))
package_dir = os.path.join(project_dir, 'artpop')
data_dir = os.path.join(package_dir, 'data')

from . import stars
from . import space
from . import image 
from . import util
from .filters import *
from .viz import *
from .image import *
from .stars import *
from .space import *
from .source import *
