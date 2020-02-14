#!/usr/bin/env python
# Copyright 2020 The TimeEvolvingMPO Authors
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

from setuptools import find_packages
from setuptools import setup

# get the __version__ variable
with open('time_evolving_mpo/version.py') as f:
  exec(f.read(), globals())

# get short and long description
short_description = "A python3 library to efficiently compute non-markovian open quantum systems."
with open("README.md", "r") as f:
  long_description = f.read()

# get requirements list
requirements = [
    requirement.strip() for requirement in open('requirements.txt').readlines()
]

setup(
    name='time_evolving_mpo',
    version=__version__,
    url='http://github.com/gefux/TimeEvolvingMPO',
    author='Gerald E. Fux',
    author_email='gerald.e.fux@gmail.com',
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
