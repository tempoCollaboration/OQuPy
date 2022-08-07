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
Module on physical information of the system.
"""

from typing import Callable, List, Optional, Text, Tuple
from copy import copy
from functools import lru_cache

import numpy as np
from numpy import ndarray

from oqupy.base_api import BaseAPIClass
from oqupy.config import NpDtype
import oqupy.operators as opr

class BaseSystem(BaseAPIClass):
    """Base class for systems. """
    def __init__(
            self,
            dimension: int,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a BaseSystem object."""
        self._dimension = dimension
        super().__init__(name, description)

    @property
    def dimension(self) -> ndarray:
        """Hilbert space dimension of the system. """
        return self._dimension

class System(BaseSystem):
    r"""
    Represents a system (without any coupling to a non-Markovian bath).
    It is possible to include Lindblad terms in the master equation.
    The equations of motion for a system density matrix (without any coupling
    to a non-Markovian bath) is then:

    .. math::

        \frac{d}{dt}\rho(t) = &-i [\hat{H}, \rho(t)] \\
            &+ \sum_n^N \gamma_n \left(
                \hat{A}_n \rho(t) \hat{A}_n^\dagger
                - \frac{1}{2} \hat{A}_n^\dagger \hat{A}_n \rho(t)
                - \frac{1}{2} \rho(t) \hat{A}_n^\dagger \hat{A}_n \right)

    with `hamiltionian` :math:`\hat{H}`, the rates `gammas` :math:`\gamma_n` and
    `linblad_operators` :math:`\hat{A}_n`.

    Parameters
    ----------
    hamiltonian: ndarray
        System-only Hamiltonian :math:`\hat{H}`.
    gammas: List(float)
        The rates :math:`\gamma_n`.
    lindblad_operators: list(ndarray)
        The Lindblad operators :math:`\hat{A}_n`.
    name: str
        An optional name for the system.
    description: str
        An optional description of the system.
    """

    def __init__(
            self,
            hamiltonian: ndarray,
            gammas: Optional[List[float]] = None,
            lindblad_operators: Optional[List[ndarray]] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a System object. """
        # input check for Hamiltonian.
        self._hamiltonian = _check_hamiltonian(hamiltonian)
        tmp_dimension = self._hamiltonian.shape[0]
        # input check gammas and lindblad_operators
        self._gammas, self._lindblad_operators = \
                _check_gammas_lindblad_operators(gammas,
                        lindblad_operators)
        super().__init__(tmp_dimension, name, description)

    @lru_cache(4)
    def liouvillian(self) -> ndarray:
        r"""
        Returns the Liouvillian super-operator :math:`\mathcal{L}` with

        .. math::

            \mathcal{L}\rho = -i [\hat{H}, \rho]
                + \sum_n^N \gamma_n \left(
                    \hat{A}_n \rho \hat{A}_n^\dagger
                    - \frac{1}{2} \hat{A}_n^\dagger \hat{A}_n \rho
                    - \frac{1}{2} \rho \hat{A}_n^\dagger \hat{A}_n
                  \right) .

        Returns
        -------
        liouvillian : ndarray
            Liouvillian :math:`\mathcal{L}`.
        """
        return _liouvillian(self._hamiltonian,
                            self._gammas,
                            self._lindblad_operators)

    @property
    def hamiltonian(self) -> ndarray:
        """The system Hamiltonian."""
        return copy(self._hamiltonian)

    @property
    def gammas(self) -> List[float]:
        """List of gammas."""
        return copy(self._gammas)

    @property
    def lindblad_operators(self) -> List[ndarray]:
        """List of lindblad operators."""
        return copy(self._lindblad_operators)

class TimeDependentSystem(BaseSystem):
    r"""
    Represents an explicitly time dependent system (without any coupling to a
    non-Markovian bath). It is possible to include (also explicitly
    time dependent) Lindblad terms in the master equation.
    The equations of motion for a system density matrix (without any coupling
    to a non-Markovian bath) is then:

    .. math::

        \frac{d}{dt}\rho(t) = &-i [\hat{H}(t), \rho(t)] \\
            &+ \sum_n^N \gamma_n(t) \left(
                \hat{A}_n(t) \rho(t) \hat{A}_n(t)^\dagger
                - \frac{1}{2} \hat{A}_n^\dagger(t) \hat{A}_n(t) \rho(t)
                - \frac{1}{2} \rho(t) \hat{A}_n^\dagger(t) \hat{A}_n(t) \right)

    with the time dependent `hamiltionian` :math:`\hat{H}(t)`, the  time
    dependent rates `gammas` :math:`\gamma_n(t)` and the time dependent
    `linblad_operators` :math:`\hat{A}_n(t)`.

    Parameters
    ----------
    hamiltonian: callable
        System-only Hamiltonian :math:`\hat{H}(t)`.
    gammas: List(callable)
        The rates :math:`\gamma_n(t)`.
    lindblad_operators: list(callable)
        The Lindblad operators :math:`\hat{A}_n(t)`.
    name: str
        An optional name for the system.
    description: str
        An optional description of the system.
    """
    def __init__(
            self,
            hamiltonian: Callable[[float], ndarray],
            gammas: \
                Optional[List[Callable[[float], float]]] = None,
            lindblad_operators: \
                Optional[List[Callable[[float], ndarray]]] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a TimeDependentSystem object."""
        # input check for Hamiltonian.
        self._hamiltonian = _check_tdependent_hamiltonian(hamiltonian)
        tmp_dimension = self._hamiltonian(1.0).shape[0]
        # input check gammas and lindblad_operators
        self._gammas, self._lindblad_operators = \
            _check_tdependent_gammas_lindblad_operators(
                    gammas,
                    lindblad_operators)

        super().__init__(tmp_dimension, name, description)

    def liouvillian(self, t: float) -> ndarray:
        r"""
        Returns the Liouvillian super-operator :math:`\mathcal{L}(t)` with

        .. math::

            \mathcal{L}(t)\rho = -i [\hat{H}(t), \rho]
                + \sum_n^N \gamma_n \left(
                    \hat{A}_n(t) \rho \hat{A}_n^\dagger(t)
                    - \frac{1}{2} \hat{A}_n^\dagger(t) \hat{A}_n(t) \rho
                    - \frac{1}{2} \rho \hat{A}_n^\dagger(t) \hat{A}_n(t)
                  \right),

        with time :math:`t`.

        Parameters
        ----------
        t: float (default = None)
            time :math:`t`.

        Returns
        -------
        liouvillian : ndarray
            Liouvillian :math:`\mathcal{L}(t)` at time :math:`t`.
        """
        hamiltonian = self._hamiltonian(t)
        gammas = [gamma(t) for gamma in self._gammas]
        lindblad_operators = [l_op(t) for l_op in self._lindblad_operators]
        return _liouvillian(hamiltonian, gammas, lindblad_operators)

    @property
    def hamiltonian(self) -> Callable[[float], ndarray]:
        """The system Hamiltonian. """
        return copy(self._hamiltonian)

    @property
    def gammas(self) -> List[Callable[[float], float]]:
        """List of gammas. """
        return copy(self._gammas)

    @property
    def lindblad_operators(self) -> List[Callable[[float], ndarray]]:
        """List of lindblad operators. """
        return copy(self._lindblad_operators)

class TimeDependentSystemWithField(BaseSystem):
    r"""
    Represents an explicitly time dependent system (without any coupling
    to a non-Markovian bath) with a coherent field (a complex scalar)
    :math:`\langle a \rangle` that evolves according to a specified
    equation of motion :math:`\partial_t\langle a \rangle`.

    It is possible to include (also explicitly time dependent) Lindblad
    terms in the master equation.  The equations of motion for a system
    density matrix (without any coupling to a non-Markovian bath) is
    then:

    .. math::

        \frac{d}{dt}\rho(t) = &-i [\hat{H}(t, \langle a \rangle), \rho(t)] \\
            &+ \sum_n^N \gamma_n(t) \left(
                \hat{A}_n(t) \rho(t) \hat{A}_n(t)^\dagger
                - \frac{1}{2} \hat{A}_n^\dagger(t) \hat{A}_n(t) \rho(t)
                - \frac{1}{2} \rho(t) \hat{A}_n^\dagger(t) \hat{A}_n(t) \right)

    with the  `hamiltionian` :math:`\hat{H}(t, \langle a \rangle)`
    depending on both time :math:`t` and `field` :math:`\langle
    a \rangle` (in general), the  time dependent rates `gammas`
    :math:`\gamma_n(t)` and the time dependent `linblad_operators`
    :math:`\hat{A}_n(t)`.

    Parameters
    ----------
    hamiltonian: callable
        System-only Hamiltonian :math:`\hat{H}(t, \langle a \rangle)`
        where :math:`\langle a \rangle` is the field at time :math:`t`.
    field_eom: callable
        Field equation of motion :math:`\partial_t
        \langle a \rangle(t, \rho, \langle a \rangle)`
        where :math:`\rho` and :math:`\langle a \rangle` are the system
        density matrix (a square matrix) and field at time :math:`t`.
    gammas: list(callable)
        The rates :math:`\gamma_n(t)`.
    lindblad_operators: list(callable)
        The Lindblad operators :math:`\hat{A}_n(t)`.
    name: str
        An optional name for the system.
    description: str
        An optional description of the system.
    """
    def __init__(
            self,
            hamiltonian: Callable[[float, complex], ndarray],
            field_eom: Callable[[float, ndarray, complex], complex],
            gammas: \
                Optional[List[Callable[[float], float]]] = None,
            lindblad_operators: \
                Optional[List[Callable[[float], ndarray]]] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a TimeDependentSystemWithField object."""
        # input check for Hamiltonian
        self._hamiltonian = _check_tfielddependent_hamiltonian(hamiltonian)
        tmp_dimension = self._hamiltonian(1.0, 1.0+1.0j).shape[0]
        # input check for field equation of motion
        tmp_field_eom = _check_field_eom(tmp_dimension, field_eom)
        self._field_eom = tmp_field_eom
        # input check gammas and lindblad_operators
        self._gammas, self._lindblad_operators = \
             _check_tdependent_gammas_lindblad_operators(
                     gammas,
                     lindblad_operators)

        super().__init__(tmp_dimension, name, description)

    def _linearised_hamiltonian(self, t0: float, t: float, state: ndarray,
            field: complex) -> complex:
        r"""
        Return value of the system Hamiltonian at time `t` using a linearisation
        of the field from its value at time `t0`.
        """
        return self._hamiltonian(t, self._linearised_field(t0, t, state, field))

    def _linearised_field(self, t0: float, t: float, state: ndarray,
            field: complex) -> complex:
        r"""
        Return the value of the field a time `(t-t0)` given the value at `t0`
        in a linear approximation using the field equation of motion.
        """
        # field_eom expects a square matrix
        state = state.reshape((self._dimension, self._dimension))
        return field + self._field_eom(t0, state, field) * (t-t0)

    def liouvillian(self,
            t0: float,
            t: float,
            state: ndarray,
            field: complex) -> ndarray:
        r"""
        Returns the Liouvillian super-operator :math:`\mathcal{L}(t)` such that

        .. math::

            \mathcal{L}(t)\rho = -i [\hat{H}(t, \langle a \rangle), \rho]
                + \sum_n^N \gamma_n \left(
                    \hat{A}_n(t) \rho \hat{A}_n^\dagger(t)
                    - \frac{1}{2} \hat{A}_n^\dagger(t) \hat{A}_n(t) \rho
                    - \frac{1}{2} \rho \hat{A}_n^\dagger(t) \hat{A}_n(t)
                  \right),

        with time :math:`t`.

        Parameters
        ----------
        t0: float
            Start time of the current step
        t: float
            Current time :math:`t`
        state: ndarray
            Flattened system density matrix (vector) at time :math:`t`
        field: complex
            Field value at time :math:`t` obtained from the
            linearisation of the field at :math:`t` using the field
            equation of motion.

        Returns
        -------
        liouvillian : ndarray
            Liouvillian :math:`\mathcal{L}(t)` at time :math:`t`.
        """
        try:
            t0 = float(t0)
        except Exception as e:
            raise AssertionError("Argument t0 must be float") from e
        try:
            t = float(t)
        except Exception as e:
            raise AssertionError("Argument t must be float") from e
        assert t >= t0, "Argument t must exceed t0"
        try:
            field = complex(field)
        except Exception as e:
            raise AssertionError("Argument field must be complex") from e
        assert isinstance(state, ndarray), "Argument state must be ndarray"
        assert state.size ==  self._dimension**2, "State size must match "\
                "Hilbert space dimension {}".format(self._dimension**2)
        # flatten if given as a square matrix
        state = state.reshape((self._dimension**2,))
        hamiltonian = self._linearised_hamiltonian(t0, t, state, field)
        gammas = [gamma(t) for gamma in self._gammas]
        lindblad_operators = [l_op(t) for l_op in self._lindblad_operators]
        return _liouvillian(hamiltonian, gammas, lindblad_operators)

    @property
    def hamiltonian(self) -> Callable[[float, complex], ndarray]:
        """The system Hamiltonian. """
        return copy(self._hamiltonian)

    @property
    def gammas(self) -> List[Callable[[float], float]]:
        """List of gammas. """
        return copy(self._gammas)

    @property
    def field_eom(self) -> Callable[[float, float, ndarray, complex], complex]:
        """The field equation of motion. """
        return copy(self._field_eom)

    @property
    def lindblad_operators(self) -> List[Callable[[float], ndarray]]:
        """List of lindblad operators. """
        return copy(self._lindblad_operators)

class SystemChain(BaseAPIClass):
    """
    Represents a 1D chain of systems with nearest neighbor interactions.

    Parameters
    ----------
    hilbert_space_dimensions: List[int]
        Hilbert space dimension for each chain site.
    name: str
        An optional name for the system chain.
    description: str
        An optional description of the system chain.
    """
    def __init__(
            self,
            hilbert_space_dimensions: List[int],
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a SystemChain object. """
        tmp_hs_dims = np.array(hilbert_space_dimensions, int)
        assert len(tmp_hs_dims.shape) == 1
        assert len(hilbert_space_dimensions) >= 1
        assert np.all(tmp_hs_dims > 0)
        self._hs_dims = tmp_hs_dims

        self._site_liouvillians = []
        for hs_dim in self._hs_dims:
            self._site_liouvillians.append(
                np.zeros((hs_dim**2, hs_dim**2), dtype=NpDtype))

        self._nn_liouvillians = []
        for hs_dim_l, hs_dim_r in zip(self._hs_dims[:-1], self._hs_dims[1:]):
            self._nn_liouvillians.append(
                np.zeros((hs_dim_l**2 * hs_dim_r**2, hs_dim_l**2 * hs_dim_r**2),
                dtype=NpDtype))

        super().__init__(name, description)

    def __len__(self):
        """Chain length. """
        return len(self._hs_dims)

    @property
    def hs_dims(self):
        """Hilbert space dimension for each chain site. """
        return self._hs_dims

    @property
    def site_liouvillians(self):
        """The single site Liouvillians. """
        return self._site_liouvillians

    @property
    def nn_liouvillians(self):
        """The nearest neighbor Liouvillians. """
        return self._nn_liouvillians

    def add_site_hamiltonian(
            self,
            site: int,
            hamiltonian: ndarray) -> None:
        r"""
        Add a hamiltonian term to a single site Liouvillian

        .. math::

            \mathcal{L} \rho_n = -i [\hat{H}, \rho_n]

        with `site` :math:`n` and `hamiltonian` :math:`\hat{H}`.

        Parameters
        ----------
        site: int
            Index of the site.
        hamiltonian: ndarray
            Hamiltonian acting on the single site.
        """
        assert isinstance(site, int)
        assert site >= 0
        assert site < len(self)
        op = np.array(hamiltonian, dtype=NpDtype)
        assert len(op.shape) == 2
        assert op.shape[0] == op.shape[1]
        assert self._hs_dims[site] == op.shape[0]

        self._site_liouvillians[site] += (0.0-1.0j) * opr.commutator(op)

    def add_site_liouvillian(
            self,
            site: int,
            liouvillian: ndarray) -> None:
        """
        Add a single site Liouvillian.

        Parameters
        ----------
        site: int
            Index of the site.
        liouvillian: ndarray
            Liouvillian acting on the single site.
        """
        self._site_liouvillians[site] += np.array(liouvillian, dtype=NpDtype)

    def add_site_dissipation(
            self,
            site: int,
            lindblad_operator: ndarray,
            gamma: Optional[float] = 1.0) -> None:
        r"""
        Add single site lindblad dissipator

        .. math::

            \mathcal{L} \rho_n = \gamma \left(
                    \hat{A} \rho_n \hat{A}^\dagger
                    - \frac{1}{2} \hat{A}^\dagger \hat{A} \rho_n
                    - \frac{1}{2} \rho_n \hat{A}^\dagger \hat{A} \right)

        with `site` :math:`n`, `lindblad_operator` :math:`\hat{A}`,
        and `gamma` :math:`\gamma`.

        Parameters
        ----------
        site: int
            Index of the site.
        lindblad_operator: ndarray
            Lindblad dissipator acting on the single site.
        gamma: float
            Optional multiplicative factor :math:`\gamma`.
        """
        op = np.array(lindblad_operator, dtype=NpDtype)
        op_dagger = op.conjugate().T
        self._site_liouvillians[site] += \
            gamma * (opr.left_right_super(op, op_dagger) \
                      - 0.5 * opr.acommutator(np.dot(op_dagger, op)))


    def add_nn_hamiltonian(
            self,
            site: int,
            hamiltonian_l: ndarray,
            hamiltonian_r: ndarray) -> None:
        r"""
        Add a hamiltonian term to the Liouvillian of two neighboring sites:

        .. math::

            \mathcal{L} \rho_{n,n+1} =
                -i [\hat{H}_l \otimes \hat{H}_r, \rho_{n,n+1}]

        with `site` :math:`n`, `hamiltonian_l` :math:`\hat{H}_l` and
        `hamiltonian_r` :math:`\hat{H}_r`.

        Parameters
        ----------
        site: int
            Index of the left site :math:`n`.
        hamiltonian_l: ndarray
            Hamiltonian acting on the left site :math:`n`.
        hamiltonian_r: ndarray
            Hamiltonian acting on the right site :math:`n+1`.
        """
        assert isinstance(site, int)
        assert site >= 0
        assert site < len(self) - 1
        op_l = np.array(hamiltonian_l, dtype=NpDtype)
        op_r = np.array(hamiltonian_r, dtype=NpDtype)
        assert len(op_l.shape) == 2
        assert len(op_r.shape) == 2
        assert op_l.shape[0] == op_l.shape[1]
        assert op_r.shape[0] == op_r.shape[1]
        assert self._hs_dims[site] == op_l.shape[0]
        assert self._hs_dims[site+1] == op_r.shape[0]

        self._nn_liouvillians[site] += (0.0-1.0j) \
                                       * opr.cross_commutator(op_l, op_r)

    def add_nn_liouvillian(
            self,
            site: int,
            liouvillian_l_r: ndarray) -> None:
        """
        Add Liouvillian of for the two neighboring sites `site` and `site` +1.

        Parameters
        ----------
        site: int
            Index of the left site :math:`n`.
        liouvillian_l_r: ndarray
            Liouvillian acting on sites :math:`n` and :math:`n+1`.
        """
        self._nn_liouvillians[site] += np.array(liouvillian_l_r, dtype=NpDtype)

    def add_nn_dissipation(
            self,
            site: int,
            lindblad_operator_l: ndarray,
            lindblad_operator_r: ndarray,
            gamma: Optional[float] = 1.0) -> None:
        r"""
        Add two site lindblad dissipator

        .. math::

            \mathcal{L} \rho_{n,n+1} = \gamma \left(
                    \hat{A} \rho_{n,n+1} \hat{A}^\dagger
                    - \frac{1}{2} \hat{A}^\dagger \hat{A} \rho_{n,n+1}
                    - \frac{1}{2} \rho_{n,n+1} \hat{A}^\dagger \hat{A} \right)

        where :math:`\hat{A}=\hat{A}_l\otimes\hat{A}_r`, with `site` :math:`n`,
        `lindblad_operator_l` :math:`\hat{A}_l`,
        `lindblad_operator_r` :math:`\hat{A}_r`, and `gamma` :math:`\gamma`.

        Parameters
        ----------
        site: int
            Index of the left site :math:`n`.
        lindblad_operator_l: ndarray
            Lindblad dissipator acting on the left site :math:`n`.
        lindblad_operator_r: ndarray
            Lindblad dissipator acting on the right site :math:`n+1`.
        gamma: float
            Optional multiplicative factor :math:`\gamma`.
        """
        assert isinstance(site, int)
        assert site >= 0
        assert site < len(self) - 1
        op_l = np.array(lindblad_operator_l, dtype=NpDtype)
        op_r = np.array(lindblad_operator_r, dtype=NpDtype)
        assert len(op_l.shape) == 2
        assert len(op_r.shape) == 2
        assert op_l.shape[0] == op_l.shape[1]
        assert op_r.shape[0] == op_r.shape[1]
        assert self._hs_dims[site] == op_l.shape[0]
        assert self._hs_dims[site+1] == op_r.shape[0]

        cross_lr = opr.cross_left_right_super(
            operator_1_l=op_l,
            operator_1_r=op_l.T.conjugate(),
            operator_2_l=op_r,
            operator_2_r=op_r.T.conjugate())
        cross_acomm = opr.cross_acommutator(
            operator_1=op_l.T.conjugate() @ op_l,
            operator_2=op_r.T.conjugate() @ op_r)

        self._nn_liouvillians[site] += \
            gamma * (cross_lr - 0.5 * cross_acomm)

    def get_nn_full_liouvillians(self) -> List[ndarray]:
        """
        Return the list of nearest neighbor Liouvillians
        (incorporating single site terms).
        """
        assert len(self) >= 2, \
            "To return a full set of nearest neighbor liouvillians, " \
            + "the chain has to be at least two sites long."

        nn_full_liouvillians = []
        for i in range(len(self)-1):
            factor_l = 1 if i == 0 else 0.5
            factor_r = 1 if i == len(self)-2 else 0.5

            liouv_l = self._site_liouvillians[i]
            id_l = np.identity(self._hs_dims[i]**2)
            liouv_r = self._site_liouvillians[i+1]
            id_r = np.identity(self._hs_dims[i+1]**2)
            liouv_nn = self._nn_liouvillians[i]

            nn_full_liouvillian = \
                factor_l * np.kron(liouv_l, id_r) \
                + factor_r * np.kron(id_l, liouv_r) \
                + liouv_nn

            nn_full_liouvillians.append(nn_full_liouvillian)

        return nn_full_liouvillians

def _check_hamiltonian(hamiltonian) -> ndarray:
    """Input checking for a single Hamiltonian. """
    try:
        tmp_hamiltonian = np.array(hamiltonian, dtype=NpDtype)
        tmp_hamiltonian.setflags(write=False)
    except Exception as e:
        raise AssertionError("Coupling operator must be numpy array") from e
    assert len(tmp_hamiltonian.shape) == 2, \
        "Coupling operator is not a matrix."
    assert tmp_hamiltonian.shape[0] == \
        tmp_hamiltonian.shape[1], \
        "Coupling operator is not a square matrix."
    return tmp_hamiltonian

def _check_tdependent_hamiltonian(hamiltonian) -> Callable[[float],
        ndarray]:
    """Input checking for a time-dependent Hamiltonian. """
    try:
        tmp_hamiltonian = np.vectorize(hamiltonian)
        _check_hamiltonian(tmp_hamiltonian(1.0))
    except Exception as e:
        raise AssertionError(
            "Time dependent Hamiltonian must be vectorizable callable.") \
                from e
    return tmp_hamiltonian

def _check_tfielddependent_hamiltonian(hamiltonian) -> Callable[[float,
    complex], ndarray]:
    try:
        tmp_hamiltonian = np.vectorize(hamiltonian)
        _check_hamiltonian(tmp_hamiltonian(1.0, 1.0+1.0j))
    except Exception as e:
        raise AssertionError(
                "Time and field dependent Hamiltonian must be vectorizable "\
                        "callable.") from e
    return tmp_hamiltonian

def _check_dissipator_lists(gammas, lindblad_operators) -> Tuple[List, List]:
    """Check gammas and lindblad operators are lists of equal length."""
    if gammas is None:
        gammas = []
    if lindblad_operators is None:
        lindblad_operators = []
    assert isinstance(gammas, list), \
        "Argument `gammas` must be a list)]."
    assert isinstance(lindblad_operators, list), \
        "Argument `lindblad_operators` must be a list."
    assert len(gammas) == len(lindblad_operators), \
        "Lists `gammas` and `lindblad_operators` must have the same length."
    return gammas, lindblad_operators

def _check_gammas_lindblad_operators(gammas, lindblad_operators) -> Tuple[
        List[float], List[ndarray]]:
    """Input check for time-independent gammas and lindblad_operators"""
    # firstly check both are lists of the same length
    gammas, lindblad_operators = _check_dissipator_lists(gammas,
            lindblad_operators)
    try:
        tmp_gammas = []
        for gamma in gammas:
            tmp_gammas.append(float(gamma))
    except Exception as e:
        raise AssertionError("All elements of `gammas` must be floats.") \
            from e
    try:
        tmp_lindblad_operators = []
        for lindblad_operator in lindblad_operators:
            tmp_lindblad_operators.append(
                np.array(lindblad_operator, dtype=NpDtype))
    except Exception as e:
        raise AssertionError(
            "All elements of `lindblad_operators` must be numpy arrays.") \
                from e
    return tmp_gammas, tmp_lindblad_operators

def _check_tdependent_gammas_lindblad_operators(
        gammas,
        lindblad_operators) -> Tuple[List[Callable[[float], float]],
                List[Callable[[float], ndarray]]]:
    """Input check for time-dependent gammas and lindblad_operators"""
    # firstly check both are lists of the same length
    gammas, lindblad_operators = _check_dissipator_lists(
            gammas,
            lindblad_operators)
    try:
        tmp_gammas = []
        for gamma in gammas:
            float(gamma(1.0))
            tmp_gamma = np.vectorize(gamma)
            tmp_gammas.append(tmp_gamma)
    except Exception as e:
        raise AssertionError(
            "All elements of `gammas` must be vectorizable " \
             + "callables returning floats.") from e
    try:
        tmp_lindblad_operators = []
        for lindblad_operator in lindblad_operators:
            tmp_lindblad_operator = np.vectorize(lindblad_operator)
            np.array(tmp_lindblad_operator(1.0))
            tmp_lindblad_operators.append(tmp_lindblad_operator)
    except Exception as e:
        raise AssertionError(
            "All elements of `lindblad_operators` must be vectorizable " \
            + "callables returning numpy arrays.") from e
    return tmp_gammas, tmp_lindblad_operators

def _check_field_eom(dim, field_eom):
    """Input check a field equation of motion"""
    test_matrix = _create_density_matrix(dim)
    test_field = 1.0+1.0j
    test_time = 1.0
    try:
        value = field_eom(test_time, test_matrix, test_field)
        complex(value)
    except Exception as e:
        raise AssertionError("Field equation of motion must "\
                "take a time, (dim,dim) matrix and field value "\
                "and return a (complex) scalar.") from e
    return field_eom

def _liouvillian(hamiltonian, gammas, lindblad_operators):
    """Lindbladian for a specific Hamiltonian, gammas and lindblad_operators.
    """
    liouvillian = -1j * opr.commutator(hamiltonian)
    for gamma, op in zip(gammas, lindblad_operators):
        op_dagger = op.conjugate().T
        liouvillian += gamma * (opr.left_right_super(op, op_dagger) \
                                - 0.5 * opr.acommutator(np.dot(op_dagger, op)))
    return liouvillian

def _create_density_matrix(dim, seed=1):
    r"""Create a repeatable (dim,dim) matrix that represents
    a valid density matrix :math:`rho`"""
    rng = np.random.default_rng(seed)
    a = rng.random((dim, dim)) + 1j*rng.random((dim,dim))
    b = np.matmul(a, a.conj().T)
    rho = b / b.trace()
    return rho
