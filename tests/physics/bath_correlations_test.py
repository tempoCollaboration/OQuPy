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
Tests for the oqupy.bath_correlations module.
"""

from itertools import product
import pytest
import warnings
import numpy as np
from scipy.integrate import IntegrationWarning
from scipy.special import loggamma

from oqupy.bath_correlations import BaseCorrelations
from oqupy.bath_correlations import CustomCorrelations
from oqupy.bath_correlations import CustomSD
from oqupy.bath_correlations import PowerLawSD

def exact_2d_correlations(alpha, omega_cutoff, temperature, k, dt, shape):
    # Exact 2d correlations for an Ohmic spectral density and exponential cutoff
    if temperature == 0.0:
        eta_func = lambda k: np.log(1/omega_cutoff+1.j*k*dt)-1.j*omega_cutoff*dt*k
    else:
        eta_func = lambda k: -(1/2*np.log(1+(k*dt*omega_cutoff)**2)+
                              loggamma(temperature*(1/omega_cutoff+1.j*k*dt))+
                              loggamma(temperature*(1/omega_cutoff-1.j*k*dt))-
                              2*loggamma(temperature/omega_cutoff)+
                              1j*(omega_cutoff*k*dt-np.arctan(omega_cutoff*k*dt)))
    if shape == "square":
        correl = eta_func(k-1)-2*eta_func(k)+eta_func(k+1)
    elif shape == "upper-triangle":
        correl = eta_func(k+1)-eta_func(k)
    return 2*alpha*correl

def test():
    kranges = [np.arange(1, 5), np.arange(395, 400)]

    omega_cutoff = 3.0
    temperature = 0.0
    alpha = 5e-2
    dt = 0.4

    for shape, temperature, krange in product(["square", "upper-triangle"],
                                              [0.0, 2.0],
                                              kranges):
        exact_correls_2d = exact_2d_correlations(alpha,
                                                 omega_cutoff,
                                                 temperature,
                                                 krange,
                                                 dt,
                                                 shape)
        for alt_integrator in [False, True]:
            correls = PowerLawSD(alpha=alpha,
                                 zeta=1.0,
                                 cutoff=omega_cutoff,
                                 cutoff_type="exponential",
                                 temperature=temperature,
                                 alt_integrator=alt_integrator)
            with warnings.catch_warnings():
                warnings.simplefilter(action="error",
                                      category=IntegrationWarning)
                try:
                    correls_2d = [correls.correlation_2d_integral(dt, k*dt, shape=shape)
                                  for k in krange]
                except IntegrationWarning:
                    pass
                else:
                    try:
                        np.testing.assert_allclose(exact_correls_2d,
                                                   correls_2d,
                                                   rtol=1e-3, atol=1e-6)
                    except AssertionError:
                        pytest.fail("correlation_2d_integral should "\
                                "either give precise results or at least "\
                                "warn the user that there could be some "\
                                f"numerical error, failed: shape={shape}, "\
                                f"temperature={temperature}, ",
                                f"ks={krange}, "\
                                f"alt_integrator={alt_integrator}")


