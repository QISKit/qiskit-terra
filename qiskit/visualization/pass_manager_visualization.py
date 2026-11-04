# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Visualization function for a pass manager. Passes are grouped based on their
flow controller, and coloured based on the type of pass.
"""
from __future__ import annotations

import os
import inspect
import tempfile

from qiskit.utils import optionals as _optionals
from qiskit.passmanager.base_tasks import BaseController, GenericPass
from qiskit.passmanager.flow_controllers import FlowControllerLinear
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from .exceptions import VisualizationError

DEFAULT_STYLE = {AnalysisPass: "red", TransformationPass: "blue"}


@_optionals.HAS_GRAPHVIZ.require_in_call
@_optionals.HAS_PYDOT.require_in_call
def pass_manager_drawer(pass_manager, filename=None, style=None, raw=False):
    """
    Draws the pass manager.

    This function needs `pydot <https://github.com/pydot/pydot>`__, which in turn needs
    `Graphviz <https://www.graphviz.org/>`__ to be installed.

    .. warning::
        This function will call the system Graphviz tool on a file involving user-controllable
        strings (such as pass names).  It is recommended to only call this function on trusted
        input.

    Args:
        pass_manager (PassManager): the pass manager to be drawn
        filename (str): file path to save image to
        style (dict or OrderedDict): keys are the pass classes and the values are
            the colors to make them. An example can be seen in the DEFAULT_STYLE. An ordered
            dict can be used to ensure a priority coloring when pass falls into multiple
            categories. Any values not included in the provided dict will be filled in from
            the default dict
        raw (Bool) : True if you want to save the raw Dot output not an image. The
            default is False.
    Returns:
        PIL.Image or None: an in-memory representation of the pass manager. Or None if
        no image was generated or PIL is not installed.
    Raises:
        MissingOptionalLibraryError: when nxpd or pydot not installed.
        VisualizationError: If raw=True and filename=None.

    Example:
        .. plot::
            :include-source:
            :nofigs:

            from qiskit import QuantumCircuit
            from qiskit.transpiler import generate_preset_pass_manager
            from qiskit.visualization import pass_manager_drawer

            pm = generate_preset_pass_manager(optimization_level=0)
            pass_manager_drawer(pm)
    """
    import pydot

    if not style:
        style = DEFAULT_STYLE

    # create the overall graph
    graph = pydot.Dot()

    # identifiers for nodes need to be unique, so assign an id
    # can't just use python's id in case the exact same pass was
    # appended more than once
    component_id = 0

    prev_node = None

    for index, controller_group in enumerate(pass_manager.to_flow_controller().tasks):
        subgraph, component_id, prev_node = draw_subgraph(
            controller_group, component_id, style, prev_node, index
        )
        graph.add_subgraph(subgraph)

    output = make_output(graph, raw, filename)
    return output


def _get_node_color(pss, style):
    # look in the user provided dict first
    for typ, color in style.items():
        if isinstance(pss, typ):
            return color

    # failing that, look in the default
    for typ, color in DEFAULT_STYLE.items():
        if isinstance(pss, typ):
            return color

    return "black"


@_optionals.HAS_GRAPHVIZ.require_in_call
@_optionals.HAS_PYDOT.require_in_call
def staged_pass_manager_drawer(pass_manager, filename=None, style=None, raw=False):
    """
    Draws the staged pass manager.

        This function needs `pydot <https://github.com/erocarrera/pydot>`__, which in turn needs
    `Graphviz <https://www.graphviz.org/>`__ to be installed.

    .. warning::
        This function will call the system Graphviz tool on a file involving user-controllable
        strings (such as pass names).  It is recommended to only call this function on trusted
        input.

    Args:
        pass_manager (StagedPassManager): the staged pass manager to be drawn
        filename (str): file path to save image to
        style (dict or OrderedDict): keys are the pass classes and the values are
            the colors to make them. An example can be seen in the DEFAULT_STYLE. An ordered
            dict can be used to ensure a priority coloring when pass falls into multiple
            categories. Any values not included in the provided dict will be filled in from
            the default dict
        raw (Bool) : True if you want to save the raw Dot output not an image. The
            default is False.
    Returns:
        PIL.Image or None: an in-memory representation of the pass manager. Or None if
        no image was generated or PIL is not installed.
    Raises:
        MissingOptionalLibraryError: when nxpd or pydot not installed.
        VisualizationError: If raw=True and filename=None.

    Example:
        .. plot::
           :include-source:
           :nofigs:

            %matplotlib inline
            from qiskit.providers.fake_provider import GenericBackendV2
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            pass_manager = generate_preset_pass_manager(3, GenericBackendV2(num_qubits=5))
            pass_manager.draw()
    """
    import pydot

    # only include stages that have passes
    stages = list(filter(lambda s: s is not None, pass_manager.expanded_stages))

    if not style:
        style = DEFAULT_STYLE

    # create the overall graph
    graph = pydot.Dot()

    # identifiers for nodes need to be unique, so assign an id
    # can't just use python's id in case the exact same pass was
    # appended more than once
    component_id = 0

    # keep a running count of indexes across stages
    idx = 0

    prev_node = None

    for st in stages:
        stage = getattr(pass_manager, st)

        if stage is not None:
            stagegraph = pydot.Cluster(str(st), fontname="helvetica", label=str(st), labeljust="l")
            for controller_group in stage.to_flow_controller().tasks:
                subgraph, component_id, prev_node = draw_subgraph(
                    controller_group, component_id, style, prev_node, idx
                )
                stagegraph.add_subgraph(subgraph)
                idx += 1
            graph.add_subgraph(stagegraph)

    output = make_output(graph, raw, filename)
    return output


def draw_subgraph(controller_group, component_id, style, prev_node, idx):
    """Draw subgraph."""
    import pydot

    # label is the name of the flow controller parameter
    label = f"[{idx}] "
    if isinstance(controller_group, BaseController) and not isinstance(
        controller_group, FlowControllerLinear
    ):
        label += f"{controller_group.__class__.__name__}"

    # create the subgraph for this controller
    subgraph = pydot.Cluster(str(component_id), fontname="helvetica", label=label, labeljust="l")
    component_id += 1

    if isinstance(controller_group, BaseController):
        # Assume linear pipeline
        # TODO: support pipeline branching when such controller is introduced
        tasks = getattr(controller_group, "tasks", [])
    elif isinstance(controller_group, GenericPass):
        tasks = [controller_group]
    elif isinstance(controller_group, (list, tuple)):
        tasks = controller_group
    else:
        # Invalid data
        return subgraph, component_id, prev_node

    flatten_tasks = []
    for task in tasks:
        # Flatten nested linear flow controller.
        # This situation often occurs in the builtin pass managers because it constructs
        # some stages by appending other pass manager instance converted into a linear controller.
        # Flattening inner linear controller tasks doesn't change the execution.
        if isinstance(task, FlowControllerLinear):
            flatten_tasks.extend(task.tasks)
        else:
            flatten_tasks.append(task)

    for task in flatten_tasks:
        if isinstance(task, BaseController):
            # Partly nested flow controller
            # TODO recursively inject subgraph into subgraph
            node = pydot.Node(
                str(component_id),
                color="k",
                fontname="helvetica",
                label="Nested flow controller",
                shape="rectangle",
            )
        else:
            # label is the name of the pass
            node = pydot.Node(
                str(component_id),
                color=_get_node_color(task, style),
                fontname="helvetica",
                label=str(type(task).__name__),
                shape="rectangle",
            )

        subgraph.add_node(node)
        component_id += 1

        # the arguments that were provided to the pass when it was created
        arg_spec = inspect.getfullargspec(task.__init__)
        # 0 is the args, 1: to remove the self arg
        args = arg_spec[0][1:]

        num_optional = len(arg_spec[3]) if arg_spec[3] else 0

        # add in the inputs to the pass
        for arg_index, arg in enumerate(args):
            nd_style = "solid"
            # any optional args are dashed
            # the num of optional counts from the end towards the start of the list
            if arg_index >= (len(args) - num_optional):
                nd_style = "dashed"

            input_node = pydot.Node(
                component_id,
                color="black",
                fontname="helvetica",
                fontsize=10,
                label=arg,
                shape="ellipse",
                style=nd_style,
            )
            subgraph.add_node(input_node)
            component_id += 1
            subgraph.add_edge(pydot.Edge(input_node, node))

        # if there is a previous node, add an edge between them
        if prev_node:
            subgraph.add_edge(pydot.Edge(prev_node, node))

        prev_node = node

    return subgraph, component_id, prev_node


def make_output(graph, raw, filename):
    """Produce output for pass_manager."""
    if raw:
        if filename:
            graph.write(filename, format="raw")
            return None
        else:
            raise VisualizationError("if format=raw, then a filename is required.")

    if not _optionals.HAS_PIL and filename:
        # pylint says this isn't a method - it is
        graph.write_png(filename)
        return None

    _optionals.HAS_PIL.require_now("pass manager drawer")

    with tempfile.TemporaryDirectory() as tmpdirname:
        from PIL import Image

        tmppath = os.path.join(tmpdirname, "pass_manager.png")

        # pylint says this isn't a method - it is
        graph.write_png(tmppath)

        image = Image.open(tmppath)
        os.remove(tmppath)
        if filename:
            image.save(filename, "PNG")
        return image
