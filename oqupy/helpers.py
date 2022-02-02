# Copyright 2022 The TEMPO Collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file  in compliance with the License.
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
Handy helper functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from oqupy.correlations import BaseCorrelations
from oqupy.tempo.tempo import TempoParameters


def plot_correlations_with_parameters(
        correlations: BaseCorrelations,
        parameters: TempoParameters,
        ax: Axes = None) -> Axes:
    """Plot the correlation function on a grid that corresponds to some
    tempo parameters. For comparison, it also draws a solid line that is 10%
    longer and has two more sampling points per interval.

    Parameters
    ----------
    correlations: BaseCorrelations
        The correlation object we are interested in.
    parameters: TempoParameters
        The tempo parameters that determine the grid.
    """
    if parameters.add_correlation_time is None:
        add_time = 0.0
        infinity = False
    elif parameters.add_correlation_time == np.infty:
        add_time = 0.0
        infinity = True
    else:
        add_time = parameters.add_correlation_time
        infinity = False

    dt = parameters.dt
    dkmax = parameters.dkmax
    int(add_time/dt)


    times_infl = dt/3.0 * np.arange((dkmax+1)*3 - 2)
    times_add = np.hstack((dt * np.arange(dkmax, dkmax+int(add_time/dt)),
                           np.array([dt * dkmax + add_time])))
    times_extra = np.linspace(times_add[-1], times_add[-1]*1.5, 10)

    corr = np.vectorize(correlations.correlation)

    corr_infl = corr(times_infl)
    sample = [3*i for i in range(dkmax+1)]
    corr_add = corr(times_add)
    corr_extra = corr(times_extra)

    show = False
    if ax is None:
        fig, ax = plt.subplots()
        show = True
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$C(\tau)$")
    ax.plot(
        times_infl, np.real(corr_infl),
        color="C0", linestyle="-", label="real")
    ax.scatter(
        times_infl[sample], np.real(corr_infl[sample]),
        marker="d", color="C0")
    ax.plot(
        times_infl, np.imag(corr_infl),
        color="C1", linestyle="-", label="imag")
    ax.scatter(
        times_infl[sample], np.imag(corr_infl[sample]),
        marker="o", color="C1")
    ax.plot(times_extra, np.real(corr_extra), color="C0", linestyle="-")
    ax.plot(times_extra, np.imag(corr_extra), color="C1", linestyle="-")

    if infinity:
        ax.axvline(times_add[0], color="r", linestyle="dashed")
        ax.fill_between(
            times_extra, np.real(corr_extra),
            0.0, color="C0", alpha=0.30)
        ax.fill_between(
            times_extra, np.imag(corr_extra),
            0.0, color="C1", alpha=0.30)
    elif add_time != 0.0:
        ax.plot(times_add, np.real(corr_add), color="C0", linestyle="-")
        ax.plot(times_add, np.imag(corr_add), color="C1", linestyle="-")
        ax.fill_between(
            times_add, np.real(corr_add),
            0.0, color="C0", alpha=0.30)
        ax.fill_between(
            times_add, np.imag(corr_add),
            0.0, color="C1", alpha=0.30)
        ax.axvline(times_add[0], color="k", linestyle="dashed")
        ax.axvline(times_add[-1], color="k", linestyle="dotted")
    else:
        pass
    ax.legend()

    if show:
        fig.show()
    return ax
