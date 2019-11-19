"""
set of companion functions for the toolbox cat12
--------------------------------------------------

TO DO : #####Documentation is available in the docstrings and online at
http://nilearn.github.io.

Contents
--------
This module provide a set of useful functions to work with the spm toolbox cat
Submodules
---------
No submodule for the moment
"""

from pathlib import Path
import pandas as pd
from glob import glob
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import load_img, math_img, resample_to_img
import numpy as np
from nipype.interfaces.spm.preprocess import Normalize12, Coregister
from nibabel import Nifti1Image
from os.path import isfile, isdir
import nibabel as nib
from warnings import warn


# Boolean controlling the default globbing technique when using check_niimg
# and the os.path.expanduser usage in CacheMixin.
# Default value it True, set it to False to completely deactivate this
# behavior.
EXPAND_PATH_WILDCARDS = True

# Boolean controlling whether the joblib caches should be
# flushed if the version of certain modules changes (eg nibabel, as it
# does not respect the backward compatibility in some of its internal
# structures
# This  is used in nilearn._utils.cache_mixin
CHECK_CACHE_VERSION = True

