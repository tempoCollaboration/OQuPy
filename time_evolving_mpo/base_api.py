# Copyright 2020 The TEMPO Collaboration
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
"""
Module for base classes of API objects.
"""

from typing import Dict, Optional, Text

from time_evolving_mpo.config import SEPERATOR


class BaseAPIClass:
    """
    Base class for API objects

    Parameters
    ----------
    name: str
        An optional name for the object.
    description: str
        An optional description of the object.
    description_dict: dict
        An optional dictionary with descriptive data.
    """
    def __init__(
            self,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a BaseAPIClass object. """
        self.name = name
        self.description = description
        self.description_dict = description_dict

    def __str__(self) -> Text:
        ret = []
        ret.append(SEPERATOR)
        ret.append("{} object: ".format(type(self).__name__)+self.name+"\n")
        ret.append(" {}\n".format(self.description))
        return "".join(ret)

    @property
    def name(self):
        """Name of the object. """
        return self._name

    @name.setter
    def name(self, new_name: Text = None):
        if new_name is None:
            new_name = "__unnamed__"
        else:
            assert isinstance(new_name, Text), "Name must be text."
        self._name = new_name

    @name.deleter
    def name(self):
        self.name = None

    @property
    def description(self):
        """Detailed description of the object. """
        return self._description

    @description.setter
    def description(self, new_description: Text = None):
        if new_description is None:
            new_description = "__no_description__"
        else:
            assert isinstance(new_description, Text), \
                "Description must be text."
        self._description = new_description

    @description.deleter
    def description(self):
        self.description = None

    @property
    def description_dict(self):
        """A dictionary for descriptive data. """
        return self._description_dict

    @description_dict.setter
    def description_dict(self, new_dict: Dict = None):
        if new_dict is None:
            new_dict = dict({})
        else:
            assert isinstance(new_dict, dict), \
                "Description dictionary must be a dictionary."
        self._description_dict = new_dict

    @description_dict.deleter
    def description_dict(self):
        self.description_dict = None
