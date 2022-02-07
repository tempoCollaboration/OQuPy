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
Module for calculating bath dynamics as outlined in [Gribben2021].

**[Gribben2021]**
D. Gribben, A. Strathearn, G. E. Fux, P. Kirton, and B. W. Lovett,
*Using the Environment to Understand non-Markovian Open Quantum Systems*,
arXiv:2106.04212 [quant-ph] (2021).
"""

from typing import Optional, Text, Tuple
import numpy as np
from numpy import ndarray

from oqupy.base_api import BaseAPIClass
from oqupy.process_tensor import BaseProcessTensor
from oqupy.bath import Bath
from oqupy.system import BaseSystem
from oqupy.config import NpDtype
from oqupy.contractions import compute_correlations


class TwoTimeBathCorrelations(BaseAPIClass):
    """
    Class to facilitate calculation of two-time bath correlations.

    Parameters
    ----------
    system: BaseSystem
        The system.
    bath: Bath
        The bath object containing all coupling information and temperature.
    process_tensor: ProcessTensor
        The corresponding process tensor calculated for the given bath.
    initial_state: ndarray
        Initial state of the system.
    system_correlations: ndarray
        Optional previously calculated system correlations. This must
        be an upper triangular array with all ordered correlations up to a
        certain time.
    name: str
        An optional name for the bath dynamics object.
    description: str
        An optional description of the bath dynamics object.
    """
    def __init__(
            self,
            system: BaseSystem,
            bath: Bath,
            process_tensor: BaseProcessTensor,
            initial_state: Optional[ndarray] = None,
            system_correlations: Optional[ndarray] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None
            ) -> None:
        """Create a TwoTimeBathCorrelations object."""
        self._system = system
        self._bath = bath

        initial_tensor = process_tensor.get_initial_tensor()
        assert (initial_state is None) ^ (initial_tensor is None), \
            "Initial state must be either (exclusively) encoded in the " \
            + "process tensor or given as an argument."

        self._process_tensor = process_tensor
        self._initial_state = initial_state

        if system_correlations is None:
            self._system_correlations = np.array([[]], dtype=NpDtype)
        else:
            self._system_correlations = system_correlations
        self._temp = bath.correlations.temperature
        self._bath_correlations = {}
        super().__init__(name, description)

    @property
    def system(self) -> BaseSystem:
        """The system. """
        return self._system

    @property
    def bath(self) -> Bath:
        """The bath. """
        return self._bath

    @property
    def initial_state(self) -> ndarray:
        """The initial system state. """
        return self._initial_state

    def generate_system_correlations(
            self,
            final_time: float
            ) -> None:
        r"""
        Function to generate all ordered system correlations up to a given time
        using the process tensor.

        Parameters
        ----------
        final_time: float
            The latest time appearing in the generated system correlation
            functions.
        """
        dt = self._process_tensor.dt
        corr_mat_dim = int(np.round(final_time/dt))
        current_corr_dim = self._system_correlations.shape[0]
        times_a = slice(corr_mat_dim)
        if self._system_correlations.size == 0:
            times_b = slice(corr_mat_dim)
        else:
            times_b = slice(current_corr_dim, corr_mat_dim)
        dim_diff = corr_mat_dim - current_corr_dim
        if dim_diff > 0:
            coup_op = self.bath.unitary_transform \
                @ self.bath.coupling_operator \
                @ self.bath.unitary_transform.conjugate().T
            _,_,_new_sys_correlations = \
                compute_correlations(self.system,
                                     self._process_tensor,
                                     coup_op, coup_op,
                                     times_a, times_b,
                                     initial_state = self.initial_state)

            self._system_correlations = np.pad(self._system_correlations,
                                               ((0, dim_diff), (0, 0)),
                                               'constant',
                                               constant_values = np.nan)
            self._system_correlations = np.append(self._system_correlations,
                                                  _new_sys_correlations,
                                                  axis = 1)

    def occupation(
            self,
            freq: float,
            dw: Optional[float] = 1.0,
            change_only: Optional[bool] = False
            ) -> Tuple[ndarray, ndarray]:
        r"""
        Function to calculate the change in bath occupation in a particular
        bandwidth.

        Parameters
        ----------
        freq: float
            Central frequency of the frequency band.
        dw: float
            Width of the the frequency band. By default this method returns a
            a *density* by setting the frequency band `dw=1.0`.
        change_only: bool
            Option to include the initial occupation (density) in the result.

        Returns
        -------
        times: ndarray
            Times of the occupation dynamics.
        bath_occupation: ndarray
            Occupation (density) (difference) of the bath in the specified
            frequency band.
        """
        corr_mat_dim = len(self._process_tensor)
        dt = self._process_tensor.dt
        last_time = corr_mat_dim * dt
        tlist = np.arange(0, last_time+dt, dt)
        if freq == 0:
            return tlist, np.ones(len(tlist),
                                  dtype=NpDtype) * (np.nan + 1.0j*np.nan)
        self.generate_system_correlations(last_time)
        _sys_correlations = self._system_correlations[:corr_mat_dim,
                                                      :corr_mat_dim]
        _sys_correlations = np.nan_to_num(_sys_correlations)
        last_time = len(self._process_tensor) * self._process_tensor.dt
        re_kernel, im_kernel = self._calc_kernel(freq, last_time,
                                                freq, last_time, (1, 0))
        coup = self._bath.correlations.spectral_density(freq) * dw
        bath_occupation = np.cumsum(
            np.sum(_sys_correlations.real*re_kernel \
                          + 1j*_sys_correlations.imag*im_kernel, axis = 0)
                ).real * coup
        bath_occupation = np.append([0], bath_occupation)
        if not change_only and self._temp > 0:
            bath_occupation += np.exp(-freq/self._temp) \
                / (1 - np.exp(-freq/self._temp))
        return tlist, bath_occupation

    def correlation(self,
                    freq_1: float,
                    time_1: float,
                    freq_2: Optional[float] = None,
                    time_2: Optional[float] = None,
                    dw: Optional[tuple] = (1.0, 1.0),
                    dagg: Optional[tuple] = (1, 0),
                    interaction_picture: Optional[bool] = False,
                    change_only: Optional[bool] = False
                    ) -> complex:
        r"""
        Function to calculate two-time correlation function between two
        frequency bands of a bath.

        The calculation consists of a double integral of the form:

        .. math::

            \int_0^t \int_0^{t'} \left\{
                \mathrm{Re} \langle O(t')O(t'') \rangle \, K_R(t',t'')
                + i \,\mathrm{Im} \langle O(t')O(t'') \rangle \, K_I(t',t'')
                \right\} dt'' dt'

        where :math:`O` is the system operator coupled to the bath and
        :math:`K_R` and :math:`K_I` are generally piecewise kernels which
        depend on the exact bath correlation function desired.

        Parameters
        ----------
        freq_1: float
            Frequency of the earlier time operator.
        time_1: float
            Time the earlier operator acts.
        freq_2: float
            Frequency of the later time operator. If set to None will default
            to freq_2=freq_1.
        time_2: float
            Time the later operator acts. If set to None will default to
            time_2=time_1.
        dw: tuple
            Width of the the frequency bands. By default this method returns a
            correlation *density* by setting the frequency bands to
            `dw=(1.0, 1.0)`.
        dagg: tuple
            Determines whether each operator is daggered or not e.g. (1,0)
            would correspond to :math:`< a^\dagger a >`.
        interaction_picture: bool
            Option whether to generate the result within the bath interaction
            picture.
        change_only: bool
            Option to include the initial occupation in the result.

        Returns
        -------
        correlation : complex
            Bath correlation function
            <a^{dagg[0]}_{freq_2} (time_2) a^{dagg[1]}_{freq_1} (time_1)>
        """
        dt = self._process_tensor.dt
        if time_2 is None:
            time_2 = time_1
        if freq_2 is None:
            freq_2 = freq_1
        self.generate_system_correlations(time_2)
        corr_mat_dim = int(np.round(time_2/dt))
        _sys_correlations = self._system_correlations[:corr_mat_dim,
                                                      :corr_mat_dim]
        _sys_correlations = np.nan_to_num(_sys_correlations)
        re_kernel,im_kernel = self._calc_kernel(freq_1, time_1,
                                                freq_2, time_2, dagg)
        coup_1 = dw[0] * self._bath.correlations.spectral_density(freq_1)**0.5
        coup_2 = dw[1] * self._bath.correlations.spectral_density(freq_2)**0.5
        correlation = np.sum(_sys_correlations.real*re_kernel + \
                             1j*_sys_correlations.imag*im_kernel) * \
            coup_1 * coup_2
        if (not change_only) and (freq_1 == freq_2) \
            and (dagg in ((1, 0), (0, 1))):
            if self._temp > 0:
                correlation += np.exp(-freq_1/self._temp) \
                    / (1 - np.exp(-freq_1/self._temp))
            if dagg == (0, 1):
                correlation += 1

        if not interaction_picture:
            correlation *= np.exp(1j * ((2*dagg[0] - 1) * freq_2 * time_2 + \
                                       (2*dagg[1] - 1) * freq_1 * time_1))
        return correlation

    def _calc_kernel(self,
                    freq_1: float,
                    time_1: float,
                    freq_2: float,
                    time_2: float,
                    dagg: tuple
                    ) -> Tuple[ndarray, ndarray]:
        r"""
        Function to calculate the corresponding kernel for the desired
        correlation function.
        Parameters
        ----------
        freq_1 : float
            Frequency of the earlier time operator.
        time_1 : float
            Time the earlier operator acts.
        freq_2 : float
            Frequency of the later time operator.
        time_2 : float
            Time the later operator acts.
        dagg : tuple
            Determines whether each operator is daggered or not e.g. (1,0)
            would correspond to :math:`< a^\dagger a >`

        Returns
        -------
        re_kernel : ndarray
            An array that multiplies the real part of the system correlation
            functions before being summed.
        im_kernel : ndarray
            An array that multiplies the imaginary part of the system
            correlation functions before being summed.

        The general structure of the kernel is piecewise and different for the
        real and imaginary parts of the correlation function. To accommodate
        the most general case we split the integrals up in the following way:

        .. math::
            \int_0^t \int_0^t' = \int_0^{t_1} \int_0^{t'}+
                                 \int_{t_1}^{t} \int_0^{t_1}+
                                 \int_{t_1}^{t} \int_{t_1}^{t'}

        where :math:`t_1` is the time the earlier operator acts. We will refer
        to these as regions `a`, `b` and `c` in the code. In the actual
        implementation we build the kernel for the full square integration
        region and then simply keep the upper triangular portion of the matrix.
        """
        dt = self._process_tensor.dt
        #pieces of kernel consist of some combination of phases and
        #Bose-Einstein factors
        n_1, n_2 = 0, 0
        if self._temp > 0:
            n_1 += np.exp(-freq_1/self._temp) / (1 - np.exp(-freq_1/self._temp))
            n_2 += np.exp(-freq_2/self._temp) / (1 - np.exp(-freq_2/self._temp))

        ker_dim = int(np.round(time_2 / dt))
        # calculate index corresponding to t_1
        switch = int(np.round(time_1 / dt))
        re_kernel = np.zeros((ker_dim, ker_dim), dtype = NpDtype)
        im_kernel = np.zeros((ker_dim, ker_dim), dtype = NpDtype)

        tpp_index, tp_index = np.meshgrid(
            np.arange(ker_dim), np.arange(ker_dim),
            indexing='ij') #array of indices for each array element
        regions = {
            'a': (slice(switch), slice(switch)),             #(0->t_1, 0->t_1)
            'b': (slice(switch), slice(switch, None)),       #(0->t_1, t_1->t)
            'c': (slice(switch, None), slice(switch, None))} #(t_1->t, t_1->t)

        def phase(region, swap_ts = False):
            tk = tp_index[regions[region]]
            tkp = tpp_index[regions[region]]
            if tk.size == 0 or tkp.size == 0:
                return 0
            a = -1j * ((2*dagg[0] - 1)) * freq_2
            b = -1j * ((2*dagg[1] - 1)) * freq_1
            if swap_ts:
                a, b = b, a
            if region in ('a','c'):
                ph = np.triu(
                    np.exp(a * (tk+1)*dt + b * (tkp+1)*dt) / (a * b), k = 1)
                ph -= np.triu(
                    np.exp(a * (tk+1)*dt + b * tkp*dt) / (a * b), k = 1)
                ph -= np.triu(
                    np.exp(a * tk*dt + b * (tkp+1)*dt) / (a * b), k = 1)
                ph += np.triu(
                    np.exp(a * tk*dt + b * tkp*dt) / (a * b), k = 1)
                sel = np.diag(tk)
                di = -np.exp((a * (sel + 1) + b * sel) * dt) / (a * b)
                if a + b != 0:
                    di += np.exp((a + b) * (sel + 1) * dt) / (b * (a+b))
                    di += np.exp((a + b) * sel * dt) / (a * (a+b))
                else:
                    di += (1 + a * sel * dt + b * (sel + 1) * dt) / (a * b)
                ph += np.diag(di)
            else:
                ph = np.exp(a * (tk+1)*dt + b * (tkp+1)*dt) / (a * b)
                ph -= np.exp(a * (tk+1)*dt + b * tkp*dt) / (a * b)
                ph -= np.exp(a * tk*dt + b * (tkp+1)*dt) / (a * b)
                ph += np.exp(a * tk*dt + b * tkp*dt) / (a * b)
            return ph


        if dagg == (0, 1):
            re_kernel[regions['a']] = phase('a') + phase('a', 1)

            re_kernel[regions['b']] = phase('b')

            im_kernel[regions['a']] = ((2*n_1 + 1) * phase('a') -
                                       (2*n_2 + 1) * phase('a', 1))

            im_kernel[regions['b']] = (2*n_1 + 1) * phase('b')

            im_kernel[regions['c']] = -2 * (n_1 + 1) * phase('c')

        elif dagg == (1, 0):
            re_kernel[regions['a']] = phase('a') + phase('a', 1)

            re_kernel[regions['b']] = phase('b')

            im_kernel[regions['a']] = ((2*n_1 + 1) * phase('a') -
                                       (2*n_2 + 1) * phase('a', 1))

            im_kernel[regions['b']] = (2*n_1 + 1) * phase('b')

            im_kernel[regions['c']] = 2 * n_1 * phase('c')

        elif dagg == (1, 1):
            re_kernel[regions['a']] = -(phase('a') + phase('a', 1))

            re_kernel[regions['b']] = -phase('b')

            im_kernel[regions['a']] = ((2*n_1 + 1) * phase('a') +
                                       (2*n_2 + 1) * phase('a', 1))

            im_kernel[regions['b']] = (2*n_1 + 1) * phase('b')

            im_kernel[regions['c']] = 2 * (n_1 + 1) * phase('c')

        elif dagg == (0, 0):
            re_kernel[regions['a']] = -(phase('a') + phase('a', 1))

            re_kernel[regions['b']] = -phase('b')

            im_kernel[regions['a']] = -((2*n_2 + 1) * phase('a', 1) +
                                        (2*n_1 + 1) * phase('a'))

            im_kernel[regions['b']] = -(2*n_1 + 1) * phase('b')

            im_kernel[regions['c']] = -2 * n_1 * phase('c')

        re_kernel = np.triu(re_kernel) #only keep triangular region
        im_kernel = np.triu(im_kernel)
        return re_kernel, im_kernel
