# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Schema and helper models for schema-conformant Results."""

import copy
from enum import IntEnum, StrEnum

from qiskit.exceptions import QiskitError


class MeasReturnType(StrEnum):
    """meas_return allowed values defined by legacy PulseQobjConfig object but still used by Result."""

    AVERAGE = "avg"
    SINGLE = "single"


class MeasLevel(IntEnum):
    """MeasLevel allowed values defined by legacy PulseQobjConfig object but still used by Result."""

    RAW = 0
    KERNELED = 1
    CLASSIFIED = 2


class ExperimentResultData:
    """Class representing experiment result data"""

    def __init__(
        self, counts=None, snapshots=None, memory=None, statevector=None, unitary=None, **kwargs
    ):
        """Initialize an ExperimentResultData class

        Args:
            counts (dict): A dictionary where the keys are the result in
                hexadecimal as string of the format "0xff" and the value
                is the number of counts for that result
            snapshots (dict): A dictionary where the key is the snapshot
                slot and the value is a dictionary of the snapshots for
                that slot.
            memory (list): A list of results per shot if the run had
                memory enabled
            statevector (list or numpy.array): A list or numpy array of the
                statevector result
            unitary (list or numpy.array): A list or numpy array of the
                unitary result
            kwargs (any): additional data key-value pairs.
        """
        self._data_attributes = []
        if counts is not None:
            self._data_attributes.append("counts")
            self.counts = counts
        if snapshots is not None:
            self._data_attributes.append("snapshots")
            self.snapshots = snapshots
        if memory is not None:
            self._data_attributes.append("memory")
            self.memory = memory
        if statevector is not None:
            self._data_attributes.append("statevector")
            self.statevector = statevector
        if unitary is not None:
            self._data_attributes.append("unitary")
            self.unitary = unitary
        for key, value in kwargs.items():
            setattr(self, key, value)
            self._data_attributes.append(key)

    def __repr__(self):
        string_list = []
        for field in self._data_attributes:
            string_list.append(f"{field}={getattr(self, field)}")
        return f"ExperimentResultData({', '.join(string_list)})"

    def to_dict(self):
        """Return a dictionary format representation of the ExperimentResultData

        Returns:
            dict: The dictionary form of the ExperimentResultData
        """
        out_dict = {}
        for field in self._data_attributes:
            out_dict[field] = getattr(self, field)
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new ExperimentResultData object from a dictionary.

        Args:
            data (dict): A dictionary representing the ExperimentResultData to
                         create. It will be in the same format as output by
                         :meth:`to_dict`
        Returns:
            ExperimentResultData: The ``ExperimentResultData`` object from the
                                  input dictionary.
        """
        in_data = copy.copy(data)
        return cls(**in_data)


class ExperimentResult:
    """Class representing an Experiment Result.

    Attributes:
        shots (int or tuple): the starting and ending shot for this data.
        success (bool): if true, we can trust results for this experiment.
        data (ExperimentResultData): results information.
        meas_level (int): Measurement result level.
    """

    _metadata = {}

    def __init__(
        self,
        shots,
        success,
        data,
        meas_level=MeasLevel.CLASSIFIED,
        status=None,
        seed=None,
        meas_return=None,
        header=None,
        **kwargs,
    ):
        """Initialize an ExperimentResult object.

        Args:
            shots(int or tuple): if an integer the number of shots or if a
                tuple the starting and ending shot for this data
            success (bool): True if the experiment was successful
            data (ExperimentResultData): The data for the experiment's
                result
            meas_level (int): Measurement result level
            status (str): The status of the experiment
            seed (int): The seed used for simulation (if run on a simulator)
            meas_return (str): The type of measurement returned
            header (dict): A free form dictionary header for the experiment
            kwargs: Arbitrary extra fields

        Raises:
            QiskitError: If meas_return or meas_level are not valid values
        """
        self._metadata = {}
        self.shots = shots
        self.success = success
        self.data = data
        self.meas_level = meas_level
        if header is not None:
            self.header = header
        if status is not None:
            self.status = status
        if seed is not None:
            self.seed = seed
        if meas_return is not None:
            if meas_return not in list(MeasReturnType):
                raise QiskitError(f"{meas_return} not a valid meas_return value")
            self.meas_return = meas_return
        self._metadata.update(kwargs)

    def __repr__(self):
        out = (
            f"ExperimentResult(shots={self.shots}, success={self.success},"
            f" meas_level={self.meas_level}, data={self.data}"
        )
        if hasattr(self, "header"):
            out += f", header={self.header}"
        if hasattr(self, "status"):
            out += f", status={self.status}"
        if hasattr(self, "seed"):
            out += f", seed={self.seed}"
        if hasattr(self, "meas_return"):
            out += f", meas_return={self.meas_return}"
        for key, value in self._metadata.items():
            if isinstance(value, str):
                value_str = f"'{value}'"
            else:
                value_str = repr(value)
            out += f", {key}={value_str}"
        out += ")"
        return out

    def __getattr__(self, name):
        try:
            return self._metadata[name]
        except KeyError as ex:
            raise AttributeError(f"Attribute {name} is not defined") from ex

    def to_dict(self):
        """Return a dictionary format representation of the ExperimentResult

        Returns:
            dict: The dictionary form of the ExperimentResult
        """
        out_dict = {
            "shots": self.shots,
            "success": self.success,
            "data": self.data.to_dict(),
            "meas_level": self.meas_level,
        }
        if hasattr(self, "header"):
            out_dict["header"] = self.header
        if hasattr(self, "status"):
            out_dict["status"] = self.status
        if hasattr(self, "seed"):
            out_dict["seed"] = self.seed
        if hasattr(self, "meas_return"):
            out_dict["meas_return"] = self.meas_return
        out_dict.update(self._metadata)
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new ExperimentResult object from a dictionary.

        Args:
            data (dict): A dictionary representing the ExperimentResult to
                         create. It will be in the same format as output by
                         :meth:`to_dict`

        Returns:
            ExperimentResult: The ``ExperimentResult`` object from the input
                              dictionary.
        """

        in_data = copy.copy(data)
        data_obj = ExperimentResultData.from_dict(in_data.pop("data"))
        shots = in_data.pop("shots")
        success = in_data.pop("success")

        return cls(shots, success, data_obj, **in_data)
