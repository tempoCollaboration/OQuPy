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
Module for tensor network process tensor backend.
"""

from typing import Optional, Callable, Dict, Tuple, List

import numpy as np
import tensornetwork as tn

from time_evolving_mpo.backends.tensor_network.util import create_delta
from time_evolving_mpo.backends.tensor_network.util import add_singleton
from time_evolving_mpo.dynamics import Dynamics


class TensorNetworkProcessTensorBackend:
    """See BaseProcessTensorBackend for docstring. """
    def __init__(
            self,
            tensors: List[np.ndarray],
            initial_tensor: np.ndarray,
            trace: np.ndarray,
            config: Dict):
        """Create a TensorNetworkProcessTensorBackend object. """
        if "backend" in config:
            self._backend = config["backend"]
        else:
            self._backend = None

        if len(tensors) > 0 and len(tensors[0].shape) == 3:
            _tensors = [ create_delta(t, [0, 1, 2, 2]) for t in tensors ]
        elif len(tensors) > 0 and len(tensors[0].shape) == 4:
            _tensors = tensors
        else:
            _tensors = []

        self._tensors = [ tn.Node(t, backend=self._backend) for t in _tensors ]

        if initial_tensor is not None:
            self._initial_tensor = tn.Node(initial_tensor,
                                           backend=self._backend)
        else:
            self._initial_tensor = None

        self._trace = tn.Node(trace, backend=self._backend)

        self._caps = None

    def _compute_caps(self) -> None:
        """Compute trace caps of process tensor. """
        self._caps  = [tn.Node(np.array([1.0]), backend=self._backend)]

        for tensor in reversed(self._tensors):
            old_cap = self._caps[-1].copy()
            trace_out = self._trace.copy()
            trace_in = self._trace.copy()
            ten = tensor.copy()

            ten[1] ^ old_cap[0]
            ten[2] ^ trace_in[0]
            ten[3] ^ trace_out[0]

            new_cap = ten @ trace_in @ trace_out @ old_cap
            self._caps.append(new_cap)

    def get_bond_dimensions(self) -> np.ndarray:
        """See BaseProcessTensorBackend.get_bond_dimensions() for docstring. """
        if len(self._tensors) == 0:
            return None
        dims = [self._tensors[0].shape[0]]
        for ten in self._tensors:
            dims.append(ten.shape[1])
        return np.array(dims, dtype=int)

    def export_tensors(self) -> Tuple[np.ndarray, np.ndarray]:
        """See BaseProcessTensorBackend.export_tensors() for docstring. """
        tensors = [ t.get_tensor() for t in self._tensors ]
        if self._initial_tensor is not None:
            initial_tensor = self._initial_tensor.get_tensor()
        else:
            initial_tensor = None

        return tensors, initial_tensor

    def compute_dynamics(
            self,
            controls: Callable[[int], Tuple[np.ndarray, np.ndarray]],
            initial_state: Optional[np.ndarray] = None) -> Dynamics:
        """See BaseProcessTensorBackend.compute_dynamics() for docstring. """
        assert (initial_state is None) ^ (self._initial_tensor is None), \
            "Initial state must be either (exclusively) encoded in the " \
            + "process tensor or given as an argument."

        if self._caps is None:
            self._compute_caps()

        if self._initial_tensor is None:
            initial_tensor = add_singleton(initial_state, 0)
            current = tn.Node(initial_tensor, backend=self._backend)
        else:
            current = self._initial_tensor.copy()

        current_bond_leg = current[0]
        current_state_leg = current[1]

        states = []

        for i, tensor in enumerate(self._tensors):
            pre, post = controls(i)
            cap = self._caps[-1-i].copy()
            pre_node = tn.Node(pre, backend=self._backend)
            post_node = tn.Node(post, backend=self._backend)

            node_dict, edge_dict = tn.copy([current])
            edge_dict[current_bond_leg] ^ cap[0]
            state_node = node_dict[current] @ cap
            states.append(state_node.get_tensor())

            current_bond_leg ^ tensor[0]
            current_state_leg ^ pre_node[0]
            pre_node[1] ^ tensor[2]
            tensor[3] ^ post_node[0]

            current_bond_leg = tensor[1]
            current_state_leg = post_node[1]
            current = current @ pre_node @ tensor @ post_node

        cap = self._caps[0]
        current_bond_leg ^ cap[0]
        final_state_node = current @ cap
        states.append(final_state_node.get_tensor())

        return states


    def compute_final_state(
            self,
            controls: Callable[[int], Tuple[np.ndarray, np.ndarray]],
            initial_state: Optional[np.ndarray] = None) -> Dynamics:
        """See BaseProcessTensorBackend.compute_dynamics() for docstring. """
        assert (initial_state is None) ^ (self._initial_tensor is None), \
            "Initial state must be either (exclusively) encoded in the " \
            + "process tensor or given as an argument."

        if self._initial_tensor is None:
            initial_tensor = add_singleton(initial_state, 0)
            current = tn.Node(initial_tensor, backend=self._backend)
        else:
            current = self._initial_tensor.copy()

        current_bond_leg = current[0]
        current_state_leg = current[1]

        for i, tensor in enumerate(self._tensors):
            pre, post = controls(i)
            pre_node = tn.Node(pre, backend=self._backend)
            post_node = tn.Node(post, backend=self._backend)

            current_bond_leg ^ tensor[0]
            current_state_leg ^ pre_node[0]
            pre_node[1] ^ tensor[2]
            tensor[3] ^ post_node[0]

            current_bond_leg = tensor[1]
            current_state_leg = post_node[1]

            current = current @ pre_node @ tensor @ post_node

        one = tn.Node(np.array([1.0]), backend=self._backend)
        current_bond_leg ^ one[0]

        final_tensor = current @ one
        return final_tensor.get_tensor()
