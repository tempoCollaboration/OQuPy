#!/usr/bin/env python
# Copyright 2022 The TEMPO Collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

from setuptools import find_packages
from setuptools import setup
import pkg_resources

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here)

# get the __version__ variable
with open(os.path.join(here,"oqupy/version.py"), "r") as f:
  exec(f.read(), globals())

# get short and long description
short_description = \
"A Python 3 package to efficiently compute non-Markovian open quantum systems."

with open(os.path.join(here,"README.md"), "r") as f:
  long_description = f.read()

# get requirements list
with open(os.path.join(here,"requirements.txt"), "r") as requirements_file:
    requirements = requirements_file.readlines()


setup(
    name='oqupy',
    version=__version__,
    url='http://github.com/tempoCollaboration/OQuPy',
    author='TEMPO Collaboration',
    author_email='tempo.collaboration@gmail.com',
    python_requires=('>=3.6.0'),
    install_requires=requirements,
    license='Apache 2.0',
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords = ['physics',
                'computational-physics',
                'quantum-physics',
                'quantum-information',
                'open-quantum-systems'],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]
)
