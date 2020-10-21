import os
MIST_PATH = os.getenv('MIST_PATH')
if MIST_PATH is None:
    raise Exception('Environment variable MIST_PATH does not exist.')
project_dir = os.path.dirname(os.path.dirname(__file__))
package_dir = os.path.join(project_dir, 'artpop')
data_dir = os.path.join(package_dir, 'data')

from . import stars
from . import space
from . import image 
from .filters import *
from .viz import *
from .image import *
from .stars import *
from .space import *
from .source import *
