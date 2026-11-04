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

"""Common visualization utilities."""

from qiskit.utils import optionals as _optionals


@_optionals.HAS_PIL.require_in_call("the latex circuit drawer")
def _trim(image):
    """Trim a PIL image and remove white space."""
    import PIL

    background = PIL.Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = PIL.ImageChops.difference(image, background)
    diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        image = image.crop(bbox)
    return image


MATPLOTLIB_INLINE_BACKENDS = {
    "module://ipykernel.pylab.backend_inline",
    "module://matplotlib_inline.backend_inline",
    "nbAgg",
}


def matplotlib_close_if_inline(figure):
    """Close the given matplotlib figure if the backend in use draws figures inline.

    If the backend does not draw figures inline, this does nothing.  This function is to prevent
    duplicate images appearing; the inline backends will capture the figure in preparation and
    display it as well, whereas the drawers want to return the figure to be displayed."""
    # This can only called if figure has already been created, so matplotlib must exist.
    import matplotlib.pyplot

    if matplotlib.get_backend() in MATPLOTLIB_INLINE_BACKENDS:
        matplotlib.pyplot.close(figure)
