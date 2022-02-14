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
Module for the process tensor (PT) object. This code is based on [Pollock2018].

**[Pollock2018]**
F.  A.  Pollock,  C.  Rodriguez-Rosario,  T.  Frauenheim,
M. Paternostro, and K. Modi, *Non-Markovian quantumprocesses: Complete
framework and efficient characterization*, Phys. Rev. A 97, 012127 (2018).
"""


import os
import tempfile
from typing import Optional, Text

import numpy as np
from numpy import ndarray
import tensornetwork as tn
import h5py

from oqupy.base_api import BaseAPIClass
from oqupy.config import NpDtype
from oqupy import util

class BaseProcessTensor(BaseAPIClass):
    """
    ToDo
    """
    def __init__(
            self,
            hilbert_space_dimension: int,
            dt: Optional[float] = None,
            transform_in: Optional[ndarray] = None,
            transform_out: Optional[ndarray] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """ToDo. """
        self._hs_dim = hilbert_space_dimension
        self._dt = dt
        self._rho_dim = self._hs_dim**2
        self._trace = (np.identity(self._hs_dim, dtype=NpDtype) \
                       / np.sqrt(float(self._hs_dim))).flatten()
        self._trace_square = self._trace**2

        if transform_in is not None:
            tmp_transform_in = np.array(transform_in, dtype=NpDtype)
            assert len(tmp_transform_in.shape) == 2
            assert tmp_transform_in.shape[0] == self._rho_dim
            self._in_dim = tmp_transform_in.shape[1]
            self._transform_in = tmp_transform_in
            self._trace_in = self._trace @ self._transform_in
        else:
            self._in_dim = self._rho_dim
            self._transform_in = None
            self._trace_in = self._trace

        if transform_out is not None:
            tmp_transform_out = np.array(transform_out, dtype=NpDtype)
            assert len(tmp_transform_out.shape) == 2
            assert tmp_transform_out.shape[1] == self._rho_dim
            self._out_dim = tmp_transform_out.shape[0]
            self._transform_out = tmp_transform_out
            self._trace_out = self._transform_out @ self._trace
        else:
            self._out_dim = self._rho_dim
            self._transform_out = None
            self._trace_out = self._trace

        super().__init__(name, description)

    @property
    def hilbert_space_dimension(self):
        """ToDo. """
        return self._hs_dim

    @property
    def dt(self):
        """ToDo. """
        return self._dt

    @property
    def transform_in(self):
        """ToDo. """
        return self._transform_in

    @property
    def transform_out(self):
        """ToDo. """
        return self._transform_out

    def __len__(self) -> int:
        """Length of process tensor."""
        raise NotImplementedError(
            "Class {} has no __len__() implementation.".format(
                type(self).__name__))

    def set_initial_tensor(
            self,
            initial_tensor: Optional[ndarray] = None) -> None:
        """ToDo. """
        raise NotImplementedError(
            "Class {} has no set_initial_tensor() implementation.".format(
                type(self).__name__))

    def get_initial_tensor(self) -> ndarray:
        """ToDo. """
        raise NotImplementedError(
            "Class {} has no get_initial_tensor() implementation.".format(
                type(self).__name__))

    def set_mpo_tensor(
            self,
            step: int,
            tensor: Optional[ndarray] = None) -> None:
        """ToDo. """
        raise NotImplementedError(
            "Class {} has no set_mpo_tensor() implementation.".format(
                type(self).__name__))

    def get_mpo_tensor(
            self,
            step: int,
            transformed: Optional[bool] = True) -> ndarray:
        """ToDo. """
        raise NotImplementedError(
            "Class {} has no get_mpo_tensor() implementation.".format(
                type(self).__name__))

    def set_cap_tensor(
            self,
            step: int,
            tensor: Optional[ndarray] = None) -> None:
        """ToDo. """
        raise NotImplementedError(
            "Class {} has no set_cap_tensor() implementation.".format(
                type(self).__name__))

    def get_cap_tensor(self, step: int) -> ndarray:
        """ToDo. """
        raise NotImplementedError(
            "Class {} has no get_cap_tensor() implementation.".format(
                type(self).__name__))

    def set_lam_tensor(
            self,
            step: int,
            tensor: Optional[ndarray] = None) -> None:
        """ToDo. """
        raise NotImplementedError(
            "Class {} has no set_lam_tensor() implementation.".format(
                type(self).__name__))

    def get_lam_tensor(self, step: int) -> ndarray:
        """ToDo. """
        raise NotImplementedError(
            "Class {} has no get_lam_tensor() implementation.".format(
                type(self).__name__))

    def get_bond_dimensions(self) -> ndarray:
        """Return the bond dimensions of the MPS form of the process tensor."""
        raise NotImplementedError(
            "Class {} has no get_bond_dimensions() implementation.".format(
                type(self).__name__))


class TrivialProcessTensor(BaseProcessTensor):
    """
    ToDo
    """
    def __init__(
            self,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """ToDo. """
        super().__init__(
            hilbert_space_dimension=1,
            name=name,
            description=description)

    def __len__(self) -> int:
        """Length of process tensor. """
        return 0

    def get_initial_tensor(self) -> ndarray:
        """ToDo. """
        return None

    def get_mpo_tensor(
            self,
            step: int,
            transformed: Optional[bool] = True) -> ndarray:
        """ToDo. """
        return None

    def get_cap_tensor(self, step: int) -> ndarray:
        """ToDo. """
        return np.array([1.0], dtype=NpDtype)

    def get_lam_tensor(self, step: int) -> ndarray:
        """ToDo. """
        return None

    def get_bond_dimensions(self) -> ndarray:
        """Return the bond dimensions of the MPS form of the process tensor."""
        return None

class SimpleProcessTensor(BaseProcessTensor):
    """
    ToDo
    """
    def __init__(
            self,
            hilbert_space_dimension: int,
            dt: Optional[float] = None,
            transform_in: Optional[ndarray] = None,
            transform_out: Optional[ndarray] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """ToDo. """
        self._initial_tensor = None
        self._mpo_tensors = []
        self._cap_tensors = []
        self._lam_tensors = []
        super().__init__(
            hilbert_space_dimension,
            dt,
            transform_in,
            transform_out,
            name,
            description)

    def __len__(self) -> int:
        """Length of process tensor. """
        return len(self._mpo_tensors)

    def set_initial_tensor(
            self,
            initial_tensor: Optional[ndarray] = None) -> None:
        """ToDo. """
        if initial_tensor is None:
            self._initial_tensor = None
            self._initial_tensor = np.array(initial_tensor, dtype=NpDtype)

    def get_initial_tensor(self) -> ndarray:
        """ToDo. """
        self._initial_tensor

    def set_mpo_tensor(
            self,
            step: int,
            tensor: Optional[ndarray] = None) -> None:
        """ToDo. """
        length = len(self._mpo_tensors)
        if step >= length:
            self._mpo_tensors.extend([None] * (step - length + 1))
        self._mpo_tensors[step] = np.array(tensor, dtype=NpDtype)

    def get_mpo_tensor(
            self,
            step: int,
            transformed: Optional[bool] = True) -> ndarray:
        """ToDo. """
        length = len(self._mpo_tensors)
        if step >= length or step < 0:
            raise IndexError("Process tensor index out of bound. ")
        tensor = self._mpo_tensors[step]
        if len(tensor.shape) == 3:
            tensor = util.create_delta(tensor, [0, 1, 2, 2])
        if self._transform_in is not None:
            tensor = np.dot(np.moveaxis(tensor, -2, -1),
                            self._transform_in.T)
            tensor = np.moveaxis(tensor, -1, -2)
        if self._transform_out is not None:
            tensor = np.dot(tensor, self._transform_out)
        return tensor

    def set_cap_tensor(
            self,
            step: int,
            tensor: Optional[ndarray] = None) -> None:
        """ToDo. """
        length = len(self._cap_tensors)
        if step >= length:
            self._cap_tensors.extend([None] * (step - length + 1))
        self._cap_tensors[step] = np.array(tensor, dtype=NpDtype)

    def get_cap_tensor(self, step: int) -> ndarray:
        """ToDo. """
        length = len(self._cap_tensors)
        if step >= length or step < 0:
            return None
        return self._cap_tensors[step]

    def set_lam_tensor(
            self,
            step: int,
            tensor: Optional[ndarray] = None) -> None:
        """ToDo. """
        length = len(self._lam_tensors)
        if step >= length:
            self._lam_tensors.extend([None] * (step - length + 1))
        self._lam_tensors[step] = np.array(tensor, dtype=NpDtype)

    def get_lam_tensor(self, step: int) -> ndarray:
        """ToDo. """
        length = len(self._lam_tensors)
        if step >= length or step < 0:
            return None
        return self._lam_tensors[step]

    def get_bond_dimensions(self) -> ndarray:
        """Return the bond dimensions of the MPS form of the process tensor."""
        bond_dims = []
        for mpo in self._mpo_tensors:
            if mpo is None:
                bond_dims.append(0)
            else:
                bond_dims.append(mpo.shape[0])
        bond_dims.append(self._mpo_tensors[-1].shape[1])
        return np.array(bond_dims)

    def compute_caps(self) -> None:
        """ToDo. """
        length = len(self)

        caps = [np.array([1.0], dtype=NpDtype)]
        last_cap = tn.Node(caps[-1])

        for step in reversed(range(length)):
            trace_square = tn.Node(self._trace_square)
            trace_in = tn.Node(self._trace_in)
            trace_out = tn.Node(self._trace_out)
            ten = tn.Node(self._mpo_tensors[step])

            if len(ten.shape) == 3:
                ten[1] ^ last_cap[0]
                ten[2] ^ trace_square[0]
                new_cap = ten @ last_cap @ trace_square
            else:
                ten[1] ^ last_cap[0]
                ten[2] ^ trace_in[0]
                ten[3] ^ trace_out[0]
                new_cap = ten @ last_cap @ trace_in @ trace_out
            caps.insert(0, new_cap.get_tensor())
            last_cap = new_cap
        self._cap_tensors = caps

    def export(self, filename: Text, overwrite: bool = False):
        """ToDo. """
        if overwrite:
            mode = "overwrite"
        else:
            mode = "write"

        pt_file = FileProcessTensor(
            mode=mode,
            filename=filename,
            hilbert_space_dimension=self._hs_dim,
            dt=self._dt,
            transform_in=self._transform_in,
            transform_out=self._transform_out,
            name=self.name,
            description=self.description)

        pt_file.set_initial_tensor(self._initial_tensor)
        for step, mpo in enumerate(self._mpo_tensors):
            pt_file.set_mpo_tensor(step, mpo)
        for step, cap in enumerate(self._cap_tensors):
            pt_file.set_cap_tensor(step, cap)
        for step, lam in enumerate(self._lam_tensors):
            pt_file.set_lam_tensor(step, lam)
        pt_file.close()

class FileProcessTensor(BaseProcessTensor):
    """
    ToDo
    """
    def __init__(
            self,
            mode: Text,
            filename: Optional[Text] = None,
            hilbert_space_dimension: Optional[int] = None,
            dt: Optional[float] = None,
            transform_in: Optional[ndarray] = None,
            transform_out: Optional[ndarray] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """ToDo. """

        if mode == "read":
            write = False
            overwrite = False
        elif mode == "write":
            write = True
            overwrite = False
        elif mode == "overwrite":
            write = True
            overwrite = True
        else:
            raise ValueError("Parameter 'mode' must be one of 'read'/"\
                + "'write'/'overwrite'!")

        self._f = None
        self._initial_tensor_data = None
        self._initial_tensor_shape = None
        self._mpo_tensors_data = None
        self._mpo_tensors_shape = None
        self._cap_tensors_data = None
        self._cap_tensors_shape = None
        self._lam_tensors_data = None
        self._lam_tensors_shape = None

        if write:
            if filename is None:
                self._removeable = True
                tmp_filename = tempfile._get_default_tempdir() + "/pt_" \
                    + next(tempfile._get_candidate_names()) + ".hdf5"
            else:
                self._removeable = overwrite
                tmp_filename = filename
            assert isinstance(hilbert_space_dimension, int)
            super().__init__(
                hilbert_space_dimension,
                dt,
                transform_in,
                transform_out,
                name,
                description)
            self._filename = tmp_filename
            self._create_file(tmp_filename, overwrite)
        else:
            assert filename is not None
            self._filename = filename
            self._removeable = False
            dictionary = self._read_file(filename)
            super().__init__(**dictionary)

    def _create_file(self, filename: Text, overwrite: bool):
        """ToDo."""
        if overwrite:
            self._f = h5py.File(filename, "w")
        else:
            self._f = h5py.File(filename, "x")
        data_type = h5py.vlen_dtype(np.dtype('complex128'))
        shape_type = h5py.vlen_dtype(np.dtype('i'))

        self._f.create_dataset("hs_dim", (1,), dtype='i', data=[self._hs_dim])

        if self._dt is None:
            self._f.create_dataset("dt", (1,), dtype='float64', data=[0.0])
        else:
            self._f.create_dataset(
                "dt", (1,), dtype='float64', data=[self._dt])

        if self._transform_in is None:
            self._f.create_dataset("transform_in",
                                   (1,),
                                   dtype='complex128',
                                   data=[0.0])
        else:
            self._f.create_dataset("transform_in",
                                   self._transform_in.shape,
                                   dtype='complex128',
                                   data=self._transform_in)

        if self._transform_out is None:
            self._f.create_dataset("transform_out",
                                   (1,),
                                   dtype='complex128',
                                   data=[0.0])
        else:
            self._f.create_dataset("transform_out",
                                   self._transform_out.shape,
                                   dtype='complex128',
                                   data=self._transform_out)

        self._initial_tensor_data = self._f.create_dataset(
            "initial_tensor_data", (1,), dtype=data_type)
        self._initial_tensor_shape = self._f.create_dataset(
            "initial_tensor_shape", (1,), dtype=shape_type)

        self._mpo_tensors_data = self._f.create_dataset(
            "mpo_tensors_data", (0,), dtype=data_type, maxshape=(None,))
        self._mpo_tensors_shape = self._f.create_dataset(
            "mpo_tensors_shape", (0,), dtype=shape_type, maxshape=(None,))

        self._cap_tensors_data = self._f.create_dataset(
            "cap_tensors_data", (0,), dtype=data_type, maxshape=(None,))
        self._cap_tensors_shape = self._f.create_dataset(
            "cap_tensors_shape", (0,), dtype=shape_type, maxshape=(None,))

        self._lam_tensors_data = self._f.create_dataset(
            "lam_tensors_data", (0,), dtype=data_type, maxshape=(None,))
        self._lam_tensors_shape = self._f.create_dataset(
            "lam_tensors_shape", (0,), dtype=shape_type, maxshape=(None,))

        self.set_initial_tensor(initial_tensor=None)

    def _read_file(self, filename: Text):
        """ToDo."""
        self._create = False
        self._f = h5py.File(filename, "r")

        # hilber space dimension
        hs_dim = int(self._f["hs_dim"][0])

        # time step dt
        dt = float(self._f["dt"][0])
        if dt == 0.0:
            dt = None

        # transforms
        transform_in = np.array(self._f["transform_in"])
        if transform_in == 0.0:
            transform_in = None
        transform_out = np.array(self._f["transform_out"])
        if transform_out == 0.0:
            transform_out = None

        # initial tensor and mpo/cap/lam tensors
        self._initial_tensor_data = self._f["initial_tensor_data"]
        self._initial_tensor_shape = self._f["initial_tensor_shape"]
        self._mpo_tensors_data = self._f["mpo_tensors_data"]
        self._mpo_tensors_shape = self._f["mpo_tensors_shape"]
        self._cap_tensors_data = self._f["cap_tensors_data"]
        self._cap_tensors_shape = self._f["cap_tensors_shape"]
        self._lam_tensors_data = self._f["lam_tensors_data"]
        self._lam_tensors_shape = self._f["lam_tensors_shape"]

        return {
            "hilbert_space_dimension":hs_dim,
            "dt":dt,
            "transform_in":transform_in,
            "transform_out":transform_out,
            "name":None,
            "description":None,
        }

    def __len__(self) -> int:
        """Length of process tensor."""
        return self._mpo_tensors_shape.shape[0]

    @property
    def filename(self):
        """ToDo. """
        return self._filename

    def close(self):
        """ToDo. """
        if self._f is not None:
            self._f.close()

    def remove(self):
        """ToDo. """
        self.close()
        if self._removeable:
            os.remove(self._filename)
        else:
            raise FileExistsError("This process tensor file cannot be removed.")

    def set_initial_tensor(
            self,
            initial_tensor: Optional[ndarray] = None) -> None:
        """ToDo. """
        _set_data_and_shape(step=0,
                            data=self._initial_tensor_data,
                            shape=self._initial_tensor_shape,
                            tensor=initial_tensor)

    def get_initial_tensor(self) -> ndarray:
        """ToDo. """
        return _get_data_and_shape(step=0,
                                   data=self._initial_tensor_data,
                                   shape=self._initial_tensor_shape)

    def set_mpo_tensor(
            self,
            step: int,
            tensor: Optional[ndarray]=None) -> None:
        """ToDo. """
        if tensor is None:
            tmp_tensor = np.array([0.0])
        else:
            tmp_tensor = tensor
        _set_data_and_shape(step,
                            data=self._mpo_tensors_data,
                            shape=self._mpo_tensors_shape,
                            tensor=tmp_tensor)

    def get_mpo_tensor(
            self,
            step: int,
            transformed: Optional[bool] = True) -> ndarray:
        """ToDo. """
        tensor = _get_data_and_shape(step,
                                     data=self._mpo_tensors_data,
                                     shape=self._mpo_tensors_shape)
        if transformed:
            if len(tensor.shape) == 3:
                tensor = util.create_delta(tensor, [0, 1, 2, 2])
            if self._transform_in is not None:
                tensor = np.dot(np.moveaxis(tensor, -2, -1),
                                self._transform_in.T)
                tensor = np.moveaxis(tensor, -1, -2)
            if self._transform_out is not None:
                tensor = np.dot(tensor, self._transform_out)
        return tensor

    def set_cap_tensor(
            self,
            step: int,
            tensor: Optional[ndarray]=None) -> None:
        """ToDo. """
        _set_data_and_shape(step,
                            data=self._cap_tensors_data,
                            shape=self._cap_tensors_shape,
                            tensor=tensor)

    def get_cap_tensor(self, step: int) -> ndarray:
        """ToDo. """
        try:
            tensor = _get_data_and_shape(step,
                                         data=self._cap_tensors_data,
                                         shape=self._cap_tensors_shape)
        except IndexError:
            tensor = None
        return tensor

    def set_lam_tensor(
            self,
            step: int,
            tensor: Optional[ndarray]=None) -> None:
        """ToDo. """
        _set_data_and_shape(step,
                            data=self._lam_tensors_data,
                            shape=self._lam_tensors_shape,
                            tensor=tensor)

    def get_lam_tensor(self, step: int) -> ndarray:
        """ToDo. """
        try:
            tensor = _get_data_and_shape(step,
                                         data=self._lam_tensors_data,
                                         shape=self._lam_tensors_shape)
        except IndexError:
            tensor = None
        return tensor

    def get_bond_dimensions(self) -> ndarray:
        """Return the bond dimensions of the MPS form of the process tensor."""
        bond_dims = []
        for tensor_shape in self._mpo_tensors_shape:
            bond_dims.append(tensor_shape[0])
        bond_dims.append(self._mpo_tensors_shape[-1][1])
        return np.array(bond_dims)

    def compute_caps(self) -> None:
        """ToDo. """
        length = len(self)

        cap = np.array([1.0], dtype=NpDtype)
        self.set_cap_tensor(length, cap)
        last_cap = tn.Node(cap)

        for step in reversed(range(length)):
            trace_in = tn.Node(self._trace_in)
            trace_out = tn.Node(self._trace_out)
            ten = tn.Node(self.get_mpo_tensor(step))
            ten[1] ^ last_cap[0]
            ten[2] ^ trace_in[0]
            ten[3] ^ trace_out[0]
            new_cap = ten @ last_cap @ trace_in @ trace_out
            self.set_cap_tensor(step, new_cap.get_tensor())
            last_cap = new_cap

def _set_data_and_shape(step, data, shape, tensor):
    """ToDo."""
    if tensor is None:
        tensor = np.array([0.0])
    if step >= shape.shape[0]:
        shape.resize((step+1,))
    if step >= data.shape[0]:
        data.resize((step+1,))

    shape[step] = tensor.shape
    tensor = tensor.reshape(-1)
    data[step] = tensor

def _get_data_and_shape(step, data, shape) -> ndarray:
    """ToDo."""
    if step >= shape.shape[0]:
        raise IndexError("Process tensor index out of bound!")
    tensor_shape = shape[step]
    tensor = data[step]
    tensor = tensor.reshape(tensor_shape)
    if tensor.shape == (1,) and tensor == 0.0:
        tensor = None
    return tensor


def import_process_tensor(
        filename: Text,
        process_tensor_type: Text = None) -> BaseProcessTensor:
    """
    ToDo.
    """
    pt_file = FileProcessTensor(mode="read", filename=filename)

    if process_tensor_type is None or process_tensor_type == "file":
        pt = pt_file
    elif process_tensor_type == "simple":
        pt = SimpleProcessTensor(
            hilbert_space_dimension=pt_file.hilbert_space_dimension,
            dt=pt_file.dt,
            transform_in=pt_file.transform_in,
            transform_out=pt_file.transform_out,
            name=pt_file.name,
            description=pt_file.description)
        pt.set_initial_tensor(pt_file.get_initial_tensor())

        step = 0
        while True:
            try:
                mpo = pt_file.get_mpo_tensor(step, transformed=False)
            except IndexError:
                break
            pt.set_mpo_tensor(step, mpo)
            step += 1

        step = 0
        while True:
            cap = pt_file.get_cap_tensor(step)
            if cap is None:
                break
            pt.set_cap_tensor(step, cap)
            step += 1

        step = 0
        while True:
            lam = pt_file.get_lam_tensor(step)
            if lam is None:
                break
            pt.set_lam_tensor(step, lam)
            step += 1

    else:
        raise ValueError("Parameter 'process_tensor_type' must be "\
            + "'file' or 'simple'!")

    return pt
