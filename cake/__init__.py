"""
CAKE
Constant Addition Kinetic Elucidation is a method for analyzing the kinetics of reactions performed under the constant
addition of a reactant.
"""

# Add imports here
from cake.cake_fitting import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions