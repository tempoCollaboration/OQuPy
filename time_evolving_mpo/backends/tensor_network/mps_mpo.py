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
Module containing various MPS/MPO extensions to the TensorNetwork package.

.. warning::
    This is in parts a computationally inefficent implementation to perform
    the tasks.

.. todo::
    Make it better.
"""

from typing import Any, List, Optional, Text, Tuple, Union
from copy import copy

from tensornetwork.network_components import Node, BaseNode
from tensornetwork.network_operations import split_node_full_svd
from tensornetwork.backends.base_backend import BaseBackend


class SimpleNodeArray:
    """
    ToDo
    """
    def __init__(
            self,
            tensors: List[Union[BaseNode, Any]],
            middle_names: List[Text],
            name: Optional[Text] = None,
            backend: Optional[Union[Text, BaseBackend]] = None) -> None:
        """Create a SimpleNodeArray object."""
        self.middle_names = middle_names
        self.backend = backend

        assert "left" not in middle_names
        assert "right" not in middle_names

        self.nodes = [
            Node(tensor,
                 backend=backend,
                 name='node{}'.format(n),
                 axis_names=["left"]+middle_names+["right"])
            for n, tensor in enumerate(tensors)
        ]
        self.name = name

        for i in range(len(self)-1):
            self.nodes[i]["right"] ^ self.nodes[i+1]["left"]


    def __len__(self) -> int:
        return len(self.nodes)

    def __str__(self) -> Text:
        ret = self.name+":\n  "
        for node in self.nodes:
            middle_names = node.axis_names[1:-1]
            middle_dims = [node.get_dimension(name) for name in middle_names]
            middle_str = ["{}/".format(dim) for dim in middle_dims]
            middle_str = "".join(middle_str)[:-1]
            ret += "-{}[{}]{}-".format(node.get_dimension("left"),
                                       middle_str,
                                       node.get_dimension("right"))
        return ret

    @property
    def name(self):
        """
        ToDo
        """
        return self._name

    @name.setter
    def name(self, name: Optional[Text] = None) -> None:
        if name is None:
            self._name = "__unnamed__"
        else:
            self._name = name

    @name.deleter
    def name(self) -> None:
        self.name = None

    def append(
            self,
            tensor: Any,
            left: bool = False) -> None:
        """
        ToDo
        """
        node = Node(tensor,
                    backend=self.backend,
                    axis_names=["left"]+self.middle_names+["right"])

        if len(self) == 0:
            self.nodes.append(node)
            return

        if left:
            self.nodes.insert(0, node)
            self.nodes[0]["right"] ^ self.nodes[1]["left"]
        else:
            self.nodes.append(node)
            self.nodes[-2]["right"] ^ self.nodes[-1]["left"]

        self.update_node_names("appended")
        return

    def append_left(
            self,
            tensor: Any) -> None:
        """
        ToDo
        """
        self.append(tensor, left=True)

    def append_right(
            self,
            tensor: Any) -> None:
        """
        ToDo
        """
        self.append(tensor, left=False)

    def is_standard_form(self) -> bool:
        """
        ToDo
        """
        if len(self) == 0:
            return True
        dim_l_end = self.nodes[0].get_dimension("left")
        dim_r_end = self.nodes[-1].get_dimension("right")
        if dim_l_end != 1 or dim_r_end != 1:
            return False
        return True

    def svd_sweep(
            self,
            from_index: int,
            to_index: int,
            max_singular_values: Optional[int] = None,
            max_truncation_err: Optional[float] = None,
            relative: Optional[bool] = False) -> None:
        """
        ToDo
        """
        full_length = len(self)
        assert from_index >= 0 and from_index < full_length
        assert to_index >= 0 and to_index < full_length

        if from_index <= to_index:
            direction = 1 # direction from left to right
        else:
            direction = -1 # direction from rith to left

        for i in range(from_index, to_index, direction):

            middle_names = copy(self.nodes[i].axis_names)[1:-1]
            u_node, s_node, v_node, __ = split_node_full_svd(
                self.nodes[i],
                self.nodes[i].edges[:-direction],
                self.nodes[i].edges[-direction:],
                max_singular_values=max_singular_values,
                max_truncation_err=max_truncation_err,
                relative=relative,
                left_name="u_node{}".format(i),
                middle_name="s_node{}".format(i),
                right_name="v_node{}".format(i))

            if direction == 1:
                u_node.add_axis_names(["left"] + middle_names + ["right"])
                s_node.add_axis_names(["left", "right"])
                v_node.add_axis_names(["left", "right"])

                s_v_node = s_node @ v_node
                s_v_node.name = "s_v_node{}".format(i)
                s_v_node.add_axis_names(["left", "right"])

                s_v_next_node = s_v_node @ self.nodes[i+1]
                s_v_next_node.name = "s_v_next_node{}".format(i)
                s_v_next_node.add_axis_names(["left"] + middle_names + ["right"])

                self.nodes[i] = u_node
                self.nodes[i].name = "node{}".format(i)
                self.nodes[i+1] = s_v_next_node
                self.nodes[i+1].name = "node{}".format(i+1)

            else:
                u_node.add_axis_names(["left", "right"])
                s_node.add_axis_names(["left", "right"])
                v_node.add_axis_names(["left"] + middle_names + ["right"])

                u_s_node = u_node @ s_node
                u_s_node.name = "u_s_node{}".format(i)
                u_s_node.add_axis_names(["left", "right"])

                prev_u_s_node = self.nodes[i-1] @ u_s_node
                prev_u_s_node.name = "prev_u_s_node{}".format(i)
                prev_u_s_node.add_axis_names(["left"] + middle_names + ["right"])

                self.nodes[i] = v_node
                self.nodes[i].name = "node{}".format(i)
                self.nodes[i-1] = prev_u_s_node
                self.nodes[i-1].name = "node{}".format(i-1)

    def update_node_names(self, name: Text = "node") -> None:
        """contract_with_MPO
        ToDo
        """
        for i, node in enumerate(self.nodes):
            node.name = "{}{}".format(name, i)

    def get_single(self) -> Any:
        """
        ToDo
        """
        assert len(self) == 1
        assert self.is_standard_form()
        tensor = copy(self.nodes[0].tensor)
        tensor.shape = tensor.shape[1:-1]
        return tensor


class SimpleAgnosticMPO(SimpleNodeArray):
    """
    ToDo
    """
    def __init__(
            self,
            tensors: List[Union[BaseNode, Any]],
            name: Optional[Text] = None,
            backend: Optional[Union[Text, BaseBackend]] = None) -> None:
        """Create a SimpleAgnosticMPO object."""
        super().__init__(tensors,
                         middle_names=["d1", "d2"],
                         name=name,
                         backend=backend)

    def copy(self) -> Any:
        """
        ToDo
        """
        self_type = type(self)
        tensors = [node.tensor for node in self.nodes]
        name = self.name
        backend = self.backend
        return self_type(tensors=tensors,
                         name=name,
                         backend=backend)

class SimpleAgnosticMPS(SimpleNodeArray):
    """
    ToDo
    """

    def __init__(
            self,
            tensors: List[Union[BaseNode, Any]],
            name: Optional[Text] = None,
            backend: Optional[Union[Text, BaseBackend]] = None) -> None:
        """Create a SimpleAgnosticMPS object."""
        super().__init__(tensors,
                         middle_names=["d"],
                         name=name,
                         backend=backend)

    def copy(self) -> Any:
        """
        ToDo
        """
        self_type = type(self)
        tensors = [node.tensor for node in self.nodes]
        name = self.name
        backend = self.backend
        return self_type(tensors=tensors,
                         name=name,
                         backend=backend)

    def contract_with_matrix(
            self,
            matrix: Any,
            index: int) -> None:
        """
        ToDo
        """
        m_node = Node(matrix,
                      backend=self.backend,
                      name='matrix',
                      axis_names=["d1", "d"])
        self.nodes[index]["d"] ^ m_node["d1"]
        tmp_node = self.nodes[index] @ m_node
        tmp_node.reorder_edges([tmp_node[0],
                                tmp_node[2],
                                tmp_node[1]])
        tmp_node.add_axis_names(["left", "d", "right"])
        tmp_node.name = "node{}".format(index)
        self.nodes[index] = tmp_node


    def contract_with_mpo(
            self,
            mpo: SimpleAgnosticMPO,
            from_index: int,
            to_index: int,
            max_singular_values: Optional[int] = None,
            max_truncation_err: Optional[float] = None,
            relative: Optional[bool] = False) -> None:
        """
        ToDo

        .. note::

            - updates mps with mpo by zipping with it.
            - does not change the mpo.
        """

        full_length = len(self)
        assert from_index >= 0 and from_index < full_length
        assert to_index >= 0 and to_index < full_length
        assert abs(to_index-from_index)+1 == len(mpo)
        assert mpo.is_standard_form()

        cp_mpo = mpo

        min_index = min([from_index, to_index])
        max_index = max([from_index, to_index])


        if len(cp_mpo) == 1:
            assert min_index == max_index
            tensor = cp_mpo.nodes[0].tensor
            t_shape = tensor.shape
            matrix = copy(tensor)
            matrix.shape = t_shape[1:-1]
            self.contract_with_matrix(matrix, min_index)
            return

        for node in cp_mpo.nodes:
            node.add_axis_names(["mpo_left", "d1", "d2", "mpo_right"])

        cp_mpo.nodes[0]["mpo_right"].disconnect()
        start_tensor = cp_mpo.nodes[0].tensor
        start_tensor_shape = start_tensor.shape
        new_start_tensor = copy(start_tensor)
        new_start_tensor.shape = start_tensor_shape[1:]
        new_start_node = Node(new_start_tensor,
                              backend=self.backend,
                              name="start_node",
                              axis_names=["d1", "d2", "mpo_right"])
        cp_mpo.nodes[0] = new_start_node
        cp_mpo.nodes[0]["mpo_right"] ^ cp_mpo.nodes[1]["mpo_left"]

        cp_mpo.nodes[-1]["mpo_left"].disconnect()
        end_tensor = cp_mpo.nodes[-1].tensor
        end_tensor_shape = end_tensor.shape
        new_end_tensor = copy(end_tensor)
        new_end_tensor.shape = end_tensor_shape[:-1]
        new_end_node = Node(new_end_tensor,
                            backend=self.backend,
                            name="end_node",
                            axis_names=["mpo_left", "d1", "d2"])
        cp_mpo.nodes[-1] = new_end_node
        cp_mpo.nodes[-2]["mpo_right"] ^ cp_mpo.nodes[-1]["mpo_left"]


        for i_mpo, i_mps in enumerate(range(min_index, max_index+1)):
            self.nodes[i_mps]["d"] ^ cp_mpo.nodes[i_mpo]["d1"]

        if from_index < to_index:
            for i_mpo, i_mps in enumerate(range(from_index, to_index)):
                tmp_node = self.nodes[i_mps] @ cp_mpo.nodes[i_mpo]
                tmp_node.add_axis_names(["left",
                                         "right",
                                         "d",
                                         "mpo_right"])
                u_node, s_node, v_node, __ = split_node_full_svd(
                    tmp_node,
                    [tmp_node["left"], tmp_node["d"]],
                    [tmp_node["right"], tmp_node["mpo_right"]],
                    max_singular_values=max_singular_values,
                    max_truncation_err=max_truncation_err,
                    relative=relative,
                    left_name="u_node{}".format(i_mps),
                    middle_name="s_node{}".format(i_mps),
                    right_name="v_node{}".format(i_mps))
                u_node.add_axis_names(["left", "d", "right"])
                s_node.add_axis_names(["left", "right"])
                v_node.add_axis_names(["left", "right", "mpo_right"])

                s_v_node = s_node @ v_node
                s_v_node.name = "s_v_node{}".format(i_mps)
                s_v_node.add_axis_names(["left", "right", "mpo_right"])

                s_v_next_node = s_v_node @ self.nodes[i_mps+1]
                s_v_next_node.name = "s_v_next_node{}".format(i_mps)
                s_v_next_node.add_axis_names(["left",
                                              "mpo_right",
                                              "d",
                                              "right"])

                self.nodes[i_mps] = u_node
                self.nodes[i_mps].name = "node{}".format(i_mps)
                self.nodes[i_mps+1] = s_v_next_node
                self.nodes[i_mps+1].name = "node{}".format(i_mps+1)

            tmp_node = self.nodes[to_index] @ cp_mpo.nodes[-1]
            tmp_node.reorder_edges([tmp_node[0],
                                    tmp_node[2],
                                    tmp_node[1]])
            tmp_node.add_axis_names(["left", "d", "right"])
            self.nodes[to_index] = tmp_node
            self.nodes[to_index].name = "node{}".format(to_index)

        else:
            for i_mpo, i_mps in enumerate(range(from_index, to_index, -1)):
                i_mpo = -1-i_mpo
                tmp_node = cp_mpo.nodes[i_mpo] @ self.nodes[i_mps]
                tmp_node.add_axis_names(["mpo_left",
                                         "d",
                                         "left",
                                         "right"])
                u_node, s_node, v_node, __ = split_node_full_svd(
                    tmp_node,
                    [tmp_node["mpo_left"], tmp_node["left"]],
                    [tmp_node["d"], tmp_node["right"]],
                    max_singular_values=max_singular_values,
                    max_truncation_err=max_truncation_err,
                    relative=relative,
                    left_name="u_node{}".format(i_mps),
                    middle_name="s_node{}".format(i_mps),
                    right_name="v_node{}".format(i_mps))
                u_node.add_axis_names(["mpo_left", "left", "right"])
                s_node.add_axis_names(["left", "right"])
                v_node.add_axis_names(["left", "d", "right"])

                u_s_node = u_node @ s_node
                u_s_node.name = "u_s_node{}".format(i_mps)
                u_s_node.add_axis_names(["mpo_left", "left", "right"])

                prev_u_s_node = self.nodes[i_mps-1] @ u_s_node
                prev_u_s_node.name = "prev_u_s_node{}".format(i_mps)
                prev_u_s_node.add_axis_names(["left",
                                              "d",
                                              "mpo_left",
                                              "right"])

                self.nodes[i_mps] = v_node
                self.nodes[i_mps].name = "node{}".format(i_mps)
                self.nodes[i_mps-1] = prev_u_s_node
                self.nodes[i_mps-1].name = "node{}".format(i_mps+1)

            tmp_node = cp_mpo.nodes[0] @ self.nodes[to_index]
            tmp_node.reorder_edges([tmp_node[1],
                                    tmp_node[0],
                                    tmp_node[2]])
            tmp_node.add_axis_names(["left", "d", "right"])
            self.nodes[to_index] = tmp_node
            self.nodes[to_index].name = "node{}".format(to_index)

    def contract_with_vectors(
            self,
            vectors: List[Any],
            from_index: int,
            to_index: int) -> None:
        """
        ToDo
        """
        full_length = len(self)
        assert from_index >= 0 and from_index < full_length
        assert to_index >= 0 and to_index < full_length
        assert abs(to_index-from_index) == len(vectors)

        vector_nodes = [
            Node(vector,
                 backend=self.backend,
                 name='node{}'.format(n),
                 axis_names=["d"])
            for n, vector in enumerate(vectors)
        ]

        if from_index < to_index:
            for vector_node in vector_nodes:
                self.nodes[from_index]["d"] ^ vector_node["d"]
                tmp_node = self.nodes[from_index] @ vector_node
                tmp_node.add_axis_names(["left", "right"])
                tmp_node.name = "interm_matrix"
                self.nodes[from_index+1] = tmp_node \
                                           @ self.nodes[from_index+1]
                self.nodes[from_index+1].add_axis_names(["left", "d", "right"])
                del self.nodes[from_index]
            self.update_node_names()
        else:
            for i, vector_node in enumerate(reversed(vector_nodes)):
                self.nodes[from_index-i]["d"] ^ vector_node["d"]
                tmp_node = self.nodes[from_index-i] @ vector_node
                tmp_node.add_axis_names(["left", "right"])
                tmp_node.name = "interm_matrix"
                self.nodes[from_index-i-1] = self.nodes[from_index-i-1] \
                                           @ tmp_node
                self.nodes[from_index-i-1].add_axis_names(["left",
                                                           "d",
                                                           "right"])
                del self.nodes[from_index-i]
            self.update_node_names()

def split(
        sna: SimpleNodeArray,
        n: int,
        name_1: Optional[Text] = None,
        name_2: Optional[Text] = None,
        ) -> Tuple[SimpleNodeArray, SimpleNodeArray]:
    """
    ToDo
    """
    assert n > 0 and n < len(sna)-1
    sna_type = type(sna)
    sna.nodes[n-1]["right"].disconnect()
    first = sna_type([], name=name_1)
    second = sna_type([], name=name_2)
    first.nodes = sna.nodes[:n]
    first.update_node_names()
    second.nodes = sna.nodes[n:]
    second.update_node_names()
    return first, second


def join(
        sna_1: SimpleNodeArray,
        sna_2: SimpleNodeArray,
        name: Optional[Text] = None
        ) -> SimpleNodeArray:
    """
    ToDo
    """
    sna_1_type = type(sna_1)
    sna_2_type = type(sna_2)
    assert sna_1_type == sna_2_type, \
        "Can't join simple node arrays of different type!"
    sna_1.nodes[-1]["right"] ^ sna_2.nodes[0]["left"]
    joined = sna_1_type([], name=name)
    joined.nodes = sna_1.nodes + sna_2.nodes
    joined.update_node_names()
    return joined
