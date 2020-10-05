import os
MIST_PATH = os.getenv('MIST_PATH')
if MIST_PATH is None:
    raise Exception('Environment variable MIST_PATH does not exist.')
project_dir = os.path.dirname(os.path.dirname(__file__))
package_dir = os.path.join(project_dir, 'artpop')
data_dir = os.path.join(package_dir, 'data')

from .isochrones import *
from .imf import *
from .spatial_distribution import *
from .read_mist_models import *
from .stellar_pops import *
from .observatory import *
from .filter_info import *
