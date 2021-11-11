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
Module containing a NodeArray extension to the TensorNetwork package.
"""

from typing import Any, List, Optional, Text, Tuple, Union

import numpy as np
import tensornetwork as tn
from tensornetwork import Node
from tensornetwork.backends.base_backend import BaseBackend


class NodeArray:
    """NodeArray class. """
    def __init__(
            self,
            tensors: List[Union[Node, Any]],
            left: Optional[bool] = True,
            right: Optional[bool] = True,
            name: Optional[Text] = None,
            backend: Optional[Union[Text, BaseBackend]] = None) -> None:
        """Create a SimpleNodeArray object."""
        self.backend = backend
        self.name = name

        if len(tensors) == 0:
            self.nodes = []
            self.left_edge = None
            self.right_edge = None
            self.bond_edges = []
            self.array_edges = []
            return

        self.nodes = [ Node(tensor, backend=backend) for tensor in tensors ]

        if left:
            self.left_edge = self.nodes[0][0]
        else:
            self.left_edge = None

        if right:
            self.right_edge = self.nodes[-1][-1]
        else:
            self.right_edge = None

        self.bond_edges = []
        for node_l, node_r in zip(self.nodes[:-1], self.nodes[1:]):
            self.bond_edges.append(node_l[-1] ^ node_r[0])

        array_edges = []
        for i, node in enumerate(self.nodes):
            edges = []
            if i == 0 and not self.left:
                edges = [ node[r] for r in range(self.rank)]
            else:
                edges = [ node[r+1] for r in range(self.rank)]
            array_edges.append(edges)
        self.array_edges = array_edges


    @property
    def left(self):
        """Returns True if array has a left edge. """
        return not self.left_edge is None

    @property
    def right(self):
        """Returns True if array has a right edge. """
        return not self.right_edge is None

    def __len__(self) -> int:
        return len(self.nodes)

    def __str__(self) -> Text:
        ret = [self.name+":\n  "]
        for i in range(len(self)):
            dims = [ edge.dimension for edge in self.array_edges[i]]
            if i == 0:
                if self.left:
                    left_leg = "~~"
                    dims.insert(0,self.left_edge.dimension)
                else:
                    left_leg = "  "
            else:
                left_leg = "~~"
                dims.insert(0,self.bond_edges[i-1].dimension)
            if i == len(self)-1:
                if self.right:
                    right_leg = "~~"
                    dims.append(self.right_edge.dimension)
                else:
                    right_leg = "  "
            else:
                right_leg = "~~"
                dims.append(self.bond_edges[i].dimension)
            ret.extend([left_leg,str(dims),right_leg])

        return "".join(ret)

    def get_verbose_string(self):
        """Returns a verbose desciption of the NodeArray. """
        ret = [self.__str__()]
        ret.append("\n")
        ret.append(f" rank = {self.rank}\n")
        ret.append(f" len = {len(self)}\n")
        ret.append(f" bond_dimensions = {self.bond_dimensions}\n")
        ret.append(f" left = {self.left}\n")
        ret.append(f" right = {self.right}\n")
        return "".join(ret)

    @property
    def bond_dimensions(self):
        """Returns the list of bond dimensions. """
        return [edge.dimension for edge in self.bond_edges]

    @property
    def rank(self) -> int:
        """Returns the rank of the NodeArray. (1 ... MPS, 2 ... MPO, etc.) """
        if len(self) == 0:
            return None
        ranks = [ node.get_rank() for node in self.nodes]
        if not self.left:
            ranks[0] += 1 # add a thought-dummy leg left
        if not self.right:
            ranks[-1] += 1 # add a thought-dummy leg right
        ranks = np.array(ranks)
        assert np.all(ranks == ranks[0]), \
            "Number of array edges are not the same for all nodes!"
        return int(ranks[0]-2)

    @property
    def name(self):
        """The name of the NodeArray. """
        if self._name is None:
            return "__no_name__"
        return self._name

    @name.setter
    def name(self, name: Optional[Text] = None) -> None:
        if name is None:
            self._name = "__unnamed__"
        else:
            self._name = name

    @name.deleter
    def name(self) -> None:
        self._name = None

    def copy(self) -> Any:
        """Return a copy of the NodeArray. """
        ret = NodeArray([], name=self.name, backend=self.backend)
        node_dict, edge_dict = tn.copy(self.nodes)
        ret.nodes = [node_dict[node] for node in self.nodes]
        ret.left_edge = edge_dict[self.left_edge] if self.left else None
        ret.right_edge = edge_dict[self.right_edge] if self.right else None
        ret.bond_edges = [edge_dict[edge] for edge in self.bond_edges]
        array_edges = []
        for edges in self.array_edges:
            array_edges.append([edge_dict[edge] for edge in edges])
        ret.array_edges = array_edges
        return  ret

    def apply_vector(
            self,
            vector: List[Union[Node, Any]],
            left: Optional[bool] = False) -> None:
        """
        ...................................................................

          |   |   |                     |   |   |
        ~~A1~~A2~~A3~~ , ~~v     ==>  ~~A1~~A2~~a3

           self        , vector  ==>    new self

        ...................................................................
        """
        v = tn.Node(vector, backend=self.backend)
        if left:
            assert self.left, "NodeArray has no left dangling leg."
            edge = self.left_edge ^ v[0]
            self.nodes[0] = tn.contract(edge)
            self.left_edge = None
        else:
            assert self.right, "NodeArray has no right dangling leg."
            edge = self.right_edge ^ v[0]
            self.nodes[-1] = tn.contract(edge)
            self.right_edge = None

    def apply_matrix(
            self,
            matrix: List[Union[Node, Any]],
            left: Optional[bool] = False) -> None:
        """
        ...................................................................

          |   |   |                     |   |   |
        --A1~~A2~~A3~~ , ~~M~~   ==>  --A1~~A2~~a3~~

           self        , matrix  ==>    new self

        ...................................................................
        """
        m = tn.Node(matrix, backend=self.backend)
        if left:
            assert self.left, "NodeArray has no left dangling leg."
            edge = self.left_edge ^ m[0]
            self.nodes[0] = tn.contract(edge)
            self.left_edge = m[1]
        else:
            assert self.right, "NodeArray has no right dangling leg."
            edge = self.right_edge ^ m[0]
            self.nodes[-1] = tn.contract(edge)
            self.right_edge = m[1]

    def svd_sweep(
            self,
            from_index: Optional[int] = 0,
            to_index: Optional[int] = -1,
            max_singular_values: Optional[int] = None,
            max_truncation_err: Optional[float] = None,
            relative: Optional[bool] = False) -> None:
        """
        ...................................................................

          |   |   |   |   |   |            |   |   |   |   |   |
        --A1~~A2~~A3~~A4~~A5~~A6--  ==>  --A1~~A2~~a3~~a4~~a5~~A6--

              self                  ==>          new self

        ...................................................................
        """
        _from_index = len(self) + from_index if from_index<0 else from_index
        _to_index = len(self) + to_index if to_index<0 else to_index
        if _from_index < 0 or _from_index >= len(self):
            raise IndexError("Index out of range.")
        if _to_index < 0 or _to_index >= len(self):
            raise IndexError("Index out of range.")

        singular_values = []
        if _from_index < _to_index:
            for i in range(_from_index, _to_index):
                u_edges = []
                if i == 0:
                    if self.left:
                        u_edges.append(self.left_edge)
                else:
                    u_edges.append(self.bond_edges[i-1])
                u_edges.extend(self.array_edges[i])
                v_edges = [self.bond_edges[i]]

                u, s, vh, trun_vals = tn.split_node_full_svd(
                    node=self.nodes[i],
                    left_edges=u_edges,
                    right_edges=v_edges,
                    max_singular_values=max_singular_values,
                    max_truncation_err=max_truncation_err,
                    relative=relative)
                singular_values.append((s.tensor.diagonal(),trun_vals))
                self.nodes[i] = u
                self.bond_edges[i] = s[0]
                svh = s @ vh
                self.nodes[i+1] = svh @ self.nodes[i+1]
        elif  _to_index < _from_index:
            for i in range(_from_index, _to_index, -1):
                u_edges = []
                u_edges.extend(self.array_edges[i])
                if i == len(self)-1:
                    if self.right:
                        u_edges.append(self.right_edge)
                else:
                    u_edges.append(self.bond_edges[i])
                v_edges = [self.bond_edges[i-1]]

                u, s, vh, trun_vals = tn.split_node_full_svd(
                    node=self.nodes[i],
                    left_edges=u_edges,
                    right_edges=v_edges,
                    max_singular_values=max_singular_values,
                    max_truncation_err=max_truncation_err,
                    relative=relative)
                singular_values.append((s.tensor.diagonal(),trun_vals))
                self.nodes[i] = u
                self.bond_edges[i-1] = s[0]
                svh = s @ vh
                self.nodes[i-1] = svh @ self.nodes[i-1]


        return singular_values

    def contract(
            self,
            array: Any,
            axes: Optional[List[Tuple]] = None,
            left_index: Optional[int] = None,
            right_index: Optional[int] = None,
            direction: Optional[Text] = "right",
            copy: Optional[bool] = True) -> None:
        """
        ...................................................................


          |  |  |                       B~~B                |
        ~~A~~A~~A~~ ,   B~~B    ==>     |  |  |      =    ~~C~~
                        |  |          ~~A~~A~~A~~

           self     ,   array   ==>              new self

        ...................................................................
        """
        # -- input parsing ----------------
        if axes is None:
            axes = [(0,0)]

        # assert len(axes) == self.rank and len(axes) == array.rank, \
        #     "This contraction would lead to nodes with dangling legs. " \
        #     + "To contract with dangling legs use NodeArray.zip_up()."

        _left_index, _right_index = _parse_left_right_index(self,
                                                            array,
                                                            left_index,
                                                            right_index)

        if direction == "right":
            if _left_index != 0:
                assert _right_index != len(self)-1, \
                    "Can't contract to the right. Swich direction to `left`!"
        elif direction == "left":
            if _right_index != 0:
                assert _left_index != 0, \
                    "Can't contract to the left. Swich direction to `right`!"
        else:
            raise ValueError()

        b = array.copy() if copy else array

        # -- handle left and right edges --
        if b.left:
            assert _left_index == 0
            assert not self.left
            self.left_edge = b.left_edge
        if b.right:
            assert _right_index == len(self) - 1
            assert not self.right
            self.right_edge = b.right_edge

        # -- set variables for directions --
        # Variables reappear in comments below.
        if direction == "right":
            from_index = _left_index
            to_index = _right_index
            sign = +1
            reverse = False
            inner_bond = 0
        elif  direction == "left":
            reverse = False
            from_index = _right_index
            to_index = _left_index
            sign = -1
            reverse = True
            inner_bond = -1

        # -- contraction --
        carry_node = None
        # sign ( 1 / -1 )
        ias = range(from_index, to_index + sign, sign)
        # reverse ( False / True )
        ibs = reversed(range(len(b))) if reverse else range(len(b))

        for ia, ib in zip(ias, ibs):
            ax_a_list = []
            ax_b_list = []
            for ax_a, ax_b in axes:
                self.array_edges[ia][ax_a] ^ b.array_edges[ib][ax_b]
                ax_a_list.append(ax_a)
                ax_b_list.append(ax_b)
            for ax_a in sorted(ax_a_list, reverse=True):
                del self.array_edges[ia][ax_a]
            for ax_b in sorted(ax_b_list, reverse=True):
                del b.array_edges[ib][ax_b]

            contraction_nodes = [self.nodes[ia], b.nodes[ib]]
            if carry_node is not None:
                contraction_nodes.append(carry_node)
            carry_node = tn.contractors.greedy(contraction_nodes,
                                               ignore_edge_order=True)
            if ia == to_index:
                self.nodes[ia] = carry_node

        # -- remove old nodes and edges --
        for ia in sorted(range(from_index, to_index, sign), reverse=True):
            del self.array_edges[ia]
            del self.bond_edges[ia+inner_bond]
            del self.nodes[ia]

        if len(self) > 1:
            new_n = self.nodes[_left_index] @ self.nodes[_left_index + sign]
            self.nodes[_left_index + sign] = new_n
            del self.array_edges[_left_index]
            del self.bond_edges[_left_index+inner_bond]
            del self.nodes[_left_index]

    def zip_up(
            self,
            array: Any,
            axes: Optional[List[Tuple]] = None,
            left_index: Optional[int] = None,
            right_index: Optional[int] = None,
            direction: Optional[Text] = "right",
            max_singular_values: Optional[int] = None,
            max_truncation_err: Optional[float] = None,
            relative: Optional[bool] = False,
            copy: Optional[bool] = True) -> List[Tuple]:
        """
        ...................................................................

                                        |  |
          |  |  |       |  |            B~~B              |  |  |
        ~~A~~A~~A~~ ,   B~~B    ==>     |  |  |      =  ~~C~~C~~C~~
                        |  |          ~~A~~A~~A~~

           self     ,   array   ==>              new self

        ...................................................................

                                           |  |
          |  |  |       |  |               B~~B~~         |  |  |
        ~~A~~A~~A   ,   B~~B~~  ==>     |  |  |      =  ~~C~~C~~C~~
                        |  |          ~~A~~A~~A

           self     ,   array   ==>              new self

        ...................................................................
        """
        # -- input parsing --
        if axes is None:
            axes = [(0,0)]

        assert self.rank + array.rank - 2*len(axes) > 0, \
            "This contraction would lead to nodes with no legs. " \
            + "To fully contract node use NodeArray.contract()."

        _left_index, _right_index = _parse_left_right_index(self,
                                                            array,
                                                            left_index,
                                                            right_index)

        b = array.copy() if copy else array

        # -- handle left and right edges --
        if b.left:
            assert _left_index == 0
            assert not self.left
            self.left_edge = b.left_edge
        if b.right:
            assert _right_index == len(self) - 1
            assert not self.right
            self.right_edge = b.right_edge

        # -- set variables for directions --
        # Variables reappear in comments below.
        if direction == "right":
            from_index = _left_index
            to_index = _right_index
            sign = +1
            reverse = False
            ia_start_border = 0
            outer_start = self.left
            outer_start_edge = self.left_edge
            outer_bond = -1
            inner_bond = 0
        elif  direction == "left":
            reverse = False
            from_index = _right_index
            to_index = _left_index
            sign = -1
            reverse = True
            ia_start_border = len(self)-1
            outer_start = self.right
            outer_start_edge = self.right_edge
            outer_bond = 0
            inner_bond = -1
        else:
            raise ValueError()

        carry_node = None
        singular_values = []
        # sign ( 1 / -1 )
        ias = range(from_index, to_index + sign, sign)
        # reverse ( False / True )
        ibs = reversed(range(len(b))) if reverse else range(len(b))

        for ia, ib in zip(ias, ibs):
            ax_a_list = []
            ax_b_list = []
            for ax_a, ax_b in axes:
                self.array_edges[ia][ax_a] ^ b.array_edges[ib][ax_b]
                ax_a_list.append(ax_a)
                ax_b_list.append(ax_b)
            for ax_a in sorted(ax_a_list, reverse=True):
                del self.array_edges[ia][ax_a]
            for ax_b in sorted(ax_b_list, reverse=True):
                del b.array_edges[ib][ax_b]
            self.array_edges[ia].extend(b.array_edges[ib])

            contraction_nodes = [self.nodes[ia], b.nodes[ib]]
            if carry_node is not None:
                contraction_nodes.append(carry_node)
            contracted_node = tn.contractors.greedy(contraction_nodes,
                                                    ignore_edge_order=True)

            # to_index (_right_index / _left_index)
            if ia == to_index:
                self.nodes[ia] = contracted_node
            else:
                u_edges = []
                # ia_start_border (0 / len(self)-1)
                if ia == ia_start_border:
                    # outer_start (self.left / self.right)
                    if outer_start:
                        # outer_start_edge (self.left_edge / self.right_edge)
                        u_edges.append(outer_start_edge)
                else:
                    # outer_bond (-1 / 0)
                    u_edges.append(self.bond_edges[ia + outer_bond])
                u_edges.extend(self.array_edges[ia])
                # inner_bond (0 / -1)
                v_edges = [self.bond_edges[ia + inner_bond],
                           b.bond_edges[ib + inner_bond]]

                u, s, vh, trun_vals = tn.split_node_full_svd(
                    node=contracted_node,
                    left_edges=u_edges,
                    right_edges=v_edges,
                    max_singular_values=max_singular_values,
                    max_truncation_err=max_truncation_err,
                    relative=relative)
                singular_values.append((s.tensor.diagonal(),trun_vals))
                self.nodes[ia] = u
                # inner_bond (0 / -1)
                self.bond_edges[ia + inner_bond] = s[0]
                carry_node = s @ vh

        return singular_values


def join(
        left_array: NodeArray,
        right_array: NodeArray,
        copy: Optional[bool] = True,
        name: Optional[Text] = None) -> None:
    """
    ...................................................................

      |   |   |        |   |             |   |   |   |   |
    --A1~~A2~~A3~~ , ~~B1~~B2--   ==>  --A1~~A2~~A3~~B1~~B2--

      left_array   , right_array  ==>        return

    ...................................................................
    """
    a = left_array.copy() if copy else left_array
    b = right_array.copy() if copy else right_array

    assert a.right, "Left array has no right-dangling leg."
    assert b.left, "Right array has no left-dangling leg."
    assert a.rank == b.rank, "The ranks of left/right array are not equal."

    ret = NodeArray([], name=name, backend=a.backend)

    ret.nodes = a.nodes + b.nodes
    ret.left_edge = a.left_edge
    ret.right_edge = b.right_edge
    ret.bond_edges = a.bond_edges \
                      + [a.right_edge ^ b.left_edge] \
                      + b.bond_edges
    ret.array_edges = a.array_edges + b.array_edges

    return ret

def split(
        array: NodeArray,
        index: int,
        copy: Optional[bool] = True,
        name_left: Optional[Text] = None,
        name_right: Optional[Text] = None) -> Tuple[NodeArray, NodeArray]:
    """
    ...................................................................

      |   |   |   |   |            |   |   |        |   |
    --A1~~A2~~A3~~A4~~A5--  ==>  --A1~~A2~~A3~~ , ~~A4~~A5--

            array           ==>            return

    ...................................................................
    """
    _index = len(array) + index if index < 0 else index
    if _index <= 0 or _index >= len(array):
        raise IndexError("Index out of range.")

    a = array.copy() if copy else array

    new_r_edge, new_l_edge = a.bond_edges[_index-1].disconnect()

    ret_l = NodeArray([], name=name_left, backend=array.backend)
    ret_l.nodes = a.nodes[:_index]
    ret_l.left_edge = a.left_edge
    ret_l.right_edge = new_r_edge
    ret_l.bond_edges = a.bond_edges[:_index-1]
    ret_l.array_edges = a.array_edges[:_index]

    ret_r = NodeArray([], name=name_right, backend=array.backend)
    ret_r.nodes = a.nodes[_index:]
    ret_r.left_edge = new_l_edge
    ret_r.right_edge = a.right_edge
    ret_r.bond_edges = a.bond_edges[_index:]
    ret_r.array_edges = a.array_edges[_index:]

    return ret_l, ret_r


def _parse_left_right_index(
        a: Any,
        b: Any,
        left_index: int,
        right_index: int) -> Tuple[int]:
    """Parse left and right index."""
    a_indices = list(range(len(a)))

    if left_index is None and right_index is None:
        assert len(b) == len(a), \
            "Length of arrays must mach when no indices are given."
        _left_index = a_indices[0]
        _right_index = a_indices[-1]

    elif left_index is None and right_index is not None:
        _right_index = a_indices[right_index]
        _left_index = _right_index - len(b) + 1
        assert _left_index >= 0, \
            "Array is too long. It extends to the left when zipped " \
            + f"to index {right_index}."

    elif left_index is not None and right_index is None:
        _left_index = a_indices[left_index]
        _right_index = _left_index + len(b) - 1
        assert _right_index < len(a), \
            "Array is too long. It extends to the rigth when zipped " \
            + f"from index {left_index}."

    elif left_index is not None and right_index is not None:
        _left_index = a_indices[left_index]
        _right_index = a_indices[right_index]
        assert len(b) == _right_index - _left_index + 1, \
            "Length of array must match index span"

    return _left_index, _right_index
