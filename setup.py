#!/usr/bin/env python
"""Py-CAWS

A package for reading and analyzing data from the Chicago Area Waterways
System (CAWS) project.

"""


DOCLINES = __doc__.split("\n")

import glob

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration


NAME = 'pycaws'
MAINTAINER = 'Zach Sherman, Jules Cacho and Robert Jackson'
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
LICENSE = 'BSD'
PLATFORMS = "Linux"
MAJOR = 0
MINOR = 1
MICRO = 0
#SCRIPTS = glob.glob('scripts/*')
#TEST_SUITE = 'nose.collector'
#TESTS_REQUIRE = ['nose']
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


def configuration(parent_package='', top_path=None):
    """ Configuration of Py-CAWS package. """
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('pycaws')
    return config


def setup_package():
    """ Setup of Py-CAWS package. """
    setup(
        name=NAME,
        maintainer=MAINTAINER,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        version=VERSION,
        license=LICENSE,
        platforms=PLATFORMS,
        configuration=configuration,
        include_package_data=True,
        #test_suite=TEST_SUITE,
        #tests_require=TESTS_REQUIRE,
        #scripts=SCRIPTS,
    )

if __name__ == '__main__':
    setup_package()
