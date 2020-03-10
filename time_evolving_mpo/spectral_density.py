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
ToDo
"""


class BaseSD:
    """
    ToDo
    """
    def __init__(self):
        """
        ToDo
        """
        self.name = "base spectral density"

    def correlation(self, tau, temperature=0.0):
        """
        ToDo
        """
        raise NotImplementedError(
            "{} has no correlation implementation.".format(self.name))

    def correlation_2d_integral(self, time_1, time_2, shape, temperature=0.0):
        """
        ToDo
        shape ... rectangle, upper_triangle, lower_triangle
        """
        raise NotImplementedError(
            "{} has no correlation_2d_integral implementation.".format(
                self.name))


class CustomFunctionSD(BaseSD):
    """
    ToDo
    """
    def __init__(self, j_function, cutoff, cut_exponent=1.0):
        """
        J(w) = j_function(w) exp( -(w/cutoff)^cut_exponent )
        """
        pass


class CustomDataSD(BaseSD):
    """
    ToDo
    """
    def __init__(self, w, j):
        """
        ToDo
        """
        pass


class StandardSD(BaseSD):
    """
    ToDo
    """
    def __init__(self, alpha, nu, cutoff, cut_exponent=1.0):
        """
        J(w) = 2 alpha w^nu w^(1-nu) exp( -(w/cutoff)^cut_exponent )
        """
        pass


class OhmicSD(StandardSD):
    """
    ToDo
    """
    def __init__(self, alpha, cutoff, cut_exponent=1.0):
        """
        J(w) = 2 alpha w exp( -(w/cutoff)^cut_exponent )
        """
        pass


class LorentzSD(BaseSD):
    """
    ToDo
    """
    def __init__(self, alpha, omega_0, gamma):
        """
        J(w) = alpha * gamma * (w0**2) * w /
              ((w0**2 - w**2)**2 + (G**2) * (w**2))
        """
        pass
