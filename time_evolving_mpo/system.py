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
Module on physical information of the system.
"""

from typing import Callable, Dict, List, Optional, Text
from typing import Any as ArrayLike
from copy import copy
from functools import lru_cache

from numpy import array, dot, ndarray, vectorize

from time_evolving_mpo.config import NpDtype
from time_evolving_mpo.base_api import BaseAPIClass
from time_evolving_mpo.util import commutator, acommutator, left_right_super


def _check_hamiltonian(hamiltonian):
    """Input checking for a single hamiltonian. """
    try:
        __hamiltonian = array(hamiltonian, dtype=NpDtype)
        __hamiltonian.setflags(write=False)
    except Exception as e:
        raise AssertionError("Coupling operator must be numpy array") from e
    assert len(__hamiltonian.shape) == 2, \
        "Coupling operator is not a matrix."
    assert __hamiltonian.shape[0] == \
        __hamiltonian.shape[1], \
        "Coupling operator is not a square matrix."
    return __hamiltonian


def _liouvillian(hamiltonian, gammas, lindblad_operators):
    """Lindbladian for a specific hamiltonian, gammas and lindblad_operators.
    """
    liouvillian = -1j * commutator(hamiltonian)
    for gamma, op in zip(gammas, lindblad_operators):
        op_dagger = op.conjugate().T
        liouvillian += gamma * (left_right_super(op, op_dagger) \
                                - 0.5 * acommutator(dot(op_dagger, op)))
    return liouvillian


class BaseSystem(BaseAPIClass):
    """Base class for systems. """
    def __init__(
            self,
            dimension: int,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a BaseSystem object."""
        self._dimension = dimension
        super().__init__(name, description, description_dict)

    @property
    def dimension(self) -> ndarray:
        """Hilbert space dimension of the system. """
        return self._dimension

    def liouvillian(self, t: Optional[float] = None) -> ndarray:
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
        raise NotImplementedError(
            "Class {} has no liouvillian implementation.".format(
                type(self).__name__))

class System(BaseSystem):
    r"""
    Represents the system only (without any coupling to a non-markovian bath).
    It is, however possible, to include Lindblad terms in the master equation.
    The equations of motion for a system density matrix (without any coupling
    to a non-markovian bath) is then:

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
    """
    def __init__(
            self,
            hamiltonian: ArrayLike,
            gammas: Optional[List[float]] = None,
            lindblad_operators: Optional[List[ArrayLike]] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a System object. """
        # input check for hamiltonian.
        self._hamiltonian = _check_hamiltonian(hamiltonian)
        __dimension = self._hamiltonian.shape[0]

        # input check gammas and lindblad_operators
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
        try:
            __gammas = []
            for gamma in gammas:
                __gammas.append(float(gamma))
        except Exception as e:
            raise AssertionError("All elements of `gamma` must be floats.") \
                from e
        try:
            __lindblad_operators = []
            for lindblad_operator in lindblad_operators:
                __lindblad_operators.append(
                    array(lindblad_operator, dtype=NpDtype))
        except Exception as e:
            raise AssertionError(
                "All elements of `lindblad_operators` must be numpy arrays.") \
                    from e
        self._gammas = __gammas
        self._lindblad_operators = __lindblad_operators

        super().__init__(__dimension, name, description, description_dict)

    @lru_cache(2**0)
    def liouvillian(self, t: Optional[float] = None) -> ndarray:
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
        """The system hamiltonian."""
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
    non-markovian bath). It is, however possible, to include (also explicitly
    time dependent) Lindblad terms in the master equation.
    The equations of motion for a system density matrix (without any coupling
    to a non-markovian bath) is then:

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
    """
    def __init__(
            self,
            hamiltonian: ArrayLike,
            gammas: \
                Optional[List[Callable[[float], float]]] = None,
            lindblad_operators: \
                Optional[List[Callable[[float], ArrayLike]]] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a System object."""
        # input check for hamiltonian.
        try:
            __hamiltonian = vectorize(hamiltonian)
            _check_hamiltonian(__hamiltonian(1.0))
        except Exception as e:
            raise AssertionError(
                "Time dependent hamiltonian must be vectorizable callable.") \
                    from e
        self._hamiltonian = __hamiltonian
        __dimension = self._hamiltonian(1.0)

        # input check gammas and lindblad_operators
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
        try:
            __gammas = []
            for gamma in gammas:
                float(gamma(1.0))
                __gamma = vectorize(gamma)
                __gammas.append(__gamma)
        except Exception as e:
            raise AssertionError(
                "All elements of `gammas` must be vectorizable " \
                 + "callables returning floats.") from e
        try:
            __lindblad_operators = []
            for lindblad_operator in lindblad_operators:
                __lindblad_operator = vectorize(lindblad_operator)
                array(__lindblad_operator(1.0))
                __lindblad_operators.append(__lindblad_operator)
        except Exception as e:
            raise AssertionError(
                "All elements of `lindblad_operators` must be vectorizable " \
                + "callables returning numpy arrays.") from e
        self._gammas = __gammas
        self._lindblad_operators = __lindblad_operators

        super().__init__(__dimension, name, description, description_dict)

    def liouvillian(self, t: Optional[float] = None) -> ndarray:
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

        Raises
        ------
        ValueError
            If `t = None`
        """
        if t is None:
            raise ValueError("Liouvillian depends on time: Argument `t` "
                             + "must be float.")
        hamiltonian = self._hamiltonian(t)
        gammas = [gamma(t) for gamma in self._gammas]
        lindblad_operators = [l_op(t) for l_op in self._lindblad_operators]
        return _liouvillian(hamiltonian, gammas, lindblad_operators)

    @property
    def hamiltonian(self) -> Callable[[float], ndarray]:
        """The system hamiltonian. """
        return copy(self._hamiltonian)

    @property
    def gammas(self) -> List[Callable[[float], float]]:
        """List of gammas. """
        return copy(self._gammas)

    @property
    def lindblad_operators(self) -> List[Callable[[float], ndarray]]:
        """List of lindblad operators. """
        return copy(self._lindblad_operators)
