# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Data-encoding circuits
======================

In machine learning, pattern recognition and image processing, a **data-encoding circuit**
starts from an initial set of measured data and builds derived values (also known as
**features**) intended to be informative and non-redundant, facilitating the subsequent
learning and generalization steps, and in some cases leading to better human
interpretations.

A feature map is related to **dimensionality reduction**; it involves reducing the amount of
resources required to describe a large set of data. When performing analysis of complex data,
one of the major problems stems from the number of variables involved. Analysis with a large
number of variables generally requires a large amount of memory and computation power, and may
even cause a classification algorithm to overfit to training samples and generalize poorly to new
samples.

When the input data to an algorithm is too large to be processed and is suspected to be redundant
(for example, the same measurement is provided in both pounds and kilograms), then it can be
transformed into a reduced set of features, named a **feature vector**.

The process of determining a subset of the initial features is called **feature selection**.
The selected features are expected to contain the relevant information from the input data,
so that the desired task can be performed by using the reduced representation instead
of the complete initial data.

"""

from .pauli_feature_map import PauliFeatureMap, pauli_feature_map, z_feature_map, zz_feature_map
from ._z_feature_map import ZFeatureMap
from ._zz_feature_map import ZZFeatureMap
from .state_preparation import StatePreparation, UniformSuperpositionGate
from .initializer import Initialize

__all__ = [
    "Initialize",
    "PauliFeatureMap",
    "StatePreparation",
    "UniformSuperpositionGate",
    "ZFeatureMap",
    "ZZFeatureMap",
    "pauli_feature_map",
    "z_feature_map",
    "zz_feature_map",
]
