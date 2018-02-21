""" Setup for Py-CAWS Subpackages. """

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

def configuration(parent_package='', top_path=None):
    """ Configuration of pycaw subpackages. """
    config = Configuration('pycaws', parent_package, top_path)
    config.add_subpackage('io')
    config.add_subpackage('calc')
    return config

if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
