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
"""
Tests for the oqupy.bath_dynamics module.
"""
import numpy as np

from oqupy.bath_dynamics import TwoTimeBathCorrelations
from oqupy.correlations import PowerLawSD
from oqupy import TempoParameters, System, Bath, PtTempo
from oqupy.config import NpDtype

def exact_correlation(t_1, t_2, w_1, w_2, dagg,
               g_1, g_2, temp):
    #Exact solution of independent boson model mode correlation functions
    ph_1 = np.exp(1j * (2*dagg[1] - 1) * w_1 * t_1)
    ph_2 = np.exp(1j * (2*dagg[0] - 1) * w_2 * t_2)
    out = 0
    if dagg in ((0, 1), (1, 0)) and w_1 == w_2:
        out += 1
        if dagg == (1, 0) and temp > 0:
            out += (np.exp(w_1/temp) - 1) ** (-1)
    out *= ph_1 * ph_2
    out += (ph_1 * ph_2 - ph_1 - ph_2 + 1) * (g_1 * g_2)/(w_1 * w_2)
    return out

def exact_occupation(t, w, g, temp):
    #Exact solution of independent boson model mode occupation
    out = g**2/w**2 * (2 - 2*np.cos(w*t))
    if temp > 0:
        out += (np.exp(w/temp) - 1) ** (-1)
    return out

def test():
    #Set parameters
    initial_state = np.array([[1,0],[0,0]])
    wc = 10.0
    alpha = 0.1
    temperature = 2
    dt = .1
    dkmax = None
    final_time = 2
    epsrel = 1e-7
    system_correlations = np.ones((int(final_time/dt),                  #system correlations
                              int(final_time/dt)), dtype = NpDtype)     #are known exactly.
    correlations = PowerLawSD(alpha=alpha,
                              zeta=1.0,
                              cutoff=wc,
                              cutoff_type="exponential",
                              temperature=temperature)
    coupling_operator = np.array([[1,0],[0,-1]])
    system_hamiltonian = np.array([[1,0],[0,-1]])
    parameters = TempoParameters(dt, dkmax, epsrel)
    system = System(system_hamiltonian)
    bath = Bath(coupling_operator,correlations)
    pt = PtTempo(bath, 0.0, final_time, parameters)
    pt = pt.get_process_tensor()
    corr = TwoTimeBathCorrelations(system, bath, pt,
                                   initial_state = initial_state,
                                   system_correlations=system_correlations)
    #Test properties
    assert corr.system == system
    assert corr.bath == bath
    np.testing.assert_equal(corr.initial_state, initial_state)

    tlist, occ_0 = corr.occupation(0)

    np.testing.assert_equal(occ_0,np.ones(len(tlist),
                                        dtype=NpDtype) * (np.nan + 1.0j*np.nan))

    w0 = 1
    coup = correlations.spectral_density(w0)

    _, occ = corr.occupation(w0)
    corr_occ = corr.correlation(w0, tlist[-1], dagg = (0, 1))
    assert np.allclose(occ[-1],corr_occ - 1)
    _, occ_change = corr.occupation(w0, change_only = True)
    exact_occ = exact_occupation(tlist, w0, coup ** 0.5, temperature)
    exact_occ_T0 = exact_occupation(tlist, w0, coup ** 0.5, 0)

    assert np.allclose(occ, exact_occ)
    assert np.allclose(occ_change, exact_occ_T0)

    w02 = 3
    coup2 = correlations.spectral_density(w02)

    for dagg in [(0,0),(0,1),(1,0),(1,1)]:
        corrs = []
        int_corrs = []
        exact_corrs = []
        sel = 10
        for t in tlist[sel:]:
            corrs.append(corr.correlation(w0, tlist[sel], w02, t, dagg = dagg))
            int_corrs.append(corr.correlation(w0, tlist[sel], w02, t, dagg = dagg,
                                              interaction_picture=True))
            exact_corrs.append(exact_correlation(sel*dt, t, w0, w02, dagg,
                                                 coup**0.5, coup2**0.5,
                                                 temperature))
        int_phase = np.exp(-1j * ((2 * dagg[0] - 1) * w02 * tlist[sel:] + \
                                  (2 * dagg[1] - 1) * w0 * tlist[sel]))
        assert np.allclose(corrs,exact_corrs)
        assert np.allclose(int_corrs,exact_corrs * int_phase)
    corr2 = TwoTimeBathCorrelations(system, bath, pt,
                                    initial_state=initial_state)
    corr2.generate_system_correlations(1)
    assert np.allclose(corr2._system_correlations[0,:],
                       np.ones(int(len(tlist)/2)))