# Copyright 2021 The TEMPO Collaboration
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

from time_evolving_mpo.correlations import BaseCorrelations
from time_evolving_mpo.tempo import TempoParameters


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
        The correlation obeject we are interested in.
    parameters: TempoParameters
        The tempo parameters that determine the grid.
    """
    times = parameters.dt/3.0 * np.arange(int(parameters.dkmax*3.3))
    corr_func = np.vectorize(correlations.correlation)
    corr_vals = corr_func(times)
    sample = [3*i for i in range(parameters.dkmax)]

    show = False
    if ax is None:
        fig, ax = plt.subplots()
        show = True
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$C(\tau)$")
    ax.plot(times, np.real(corr_vals), color="C0", linestyle="-", label="real")
    ax.scatter(times[sample], np.real(corr_vals[sample]), marker="d", color="C0")
    ax.plot(times, np.imag(corr_vals), color="C1", linestyle="-", label="imag")
    ax.scatter(times[sample], np.imag(corr_vals[sample]), marker="o", color="C1")
    ax.legend()

    if show:
        fig.show()
    return ax
