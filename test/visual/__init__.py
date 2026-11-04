# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Image comparison tests."""

import json
import os
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


@contextmanager
def cwd(path):
    """A context manager to run in a particular path"""
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


class VisualTestUtilities:
    """Utility methods for circuit and graph visual testing"""

    @staticmethod
    def _new_gray(size, color):
        img = Image.new("L", size)
        drawing = ImageDraw.Draw(img)
        drawing.rectangle((0, 0) + size, color)
        return img

    @staticmethod
    def _black_or_b(diff_image, image, reference, opacity=0.85):
        """Copied from https://stackoverflow.com/a/30307875"""
        thresholded_diff = diff_image
        for _ in range(3):
            thresholded_diff = ImageChops.add(thresholded_diff, thresholded_diff)
        size = diff_image.size
        mask = VisualTestUtilities._new_gray(size, int(255 * (opacity)))
        shade = VisualTestUtilities._new_gray(size, 0)
        new = reference.copy()
        new.paste(shade, mask=mask)
        if image.size != new.size:
            image = image.resize(new.size)
        if image.size != thresholded_diff.size:
            thresholded_diff = thresholded_diff.resize(image.size)
        new.paste(image, mask=thresholded_diff)
        return new

    @staticmethod
    def _get_black_pixels(image):
        black_and_white_version = image.convert("1")
        black_pixels = black_and_white_version.histogram()[0]
        return black_pixels

    @staticmethod
    def _save_diff(current, expected, image_name, failure_diff_dir, failure_prefix):
        diff_name = current.split(".")
        diff_name.insert(-1, "diff")
        diff_name = ".".join(diff_name)

        current = Image.open(current)
        expected = Image.open(expected)

        diff = ImageChops.difference(expected, current).convert("L")

        black_pixels = VisualTestUtilities._get_black_pixels(diff)
        total_pixels = diff.size[0] * diff.size[1]
        diff_ratio = black_pixels / total_pixels

        if diff_ratio != 1:
            VisualTestUtilities._black_or_b(diff, current, expected).save(
                str(Path(failure_diff_dir) / (failure_prefix + image_name)), "PNG"
            )
        else:
            VisualTestUtilities._black_or_b(diff, current, expected).save(diff_name, "PNG")
        return diff_ratio

    @staticmethod
    def save_data_wrap(func, testname, result_dir):
        """A wrapper to save the data a test"""

        def wrapper(*args, **kwargs):
            image_filename = kwargs["filename"]
            with cwd(result_dir):
                results = func(*args, **kwargs)
                VisualTestUtilities.save_data(image_filename, testname)
            return results

        return wrapper

    @staticmethod
    def save_data(image_filename, testname):
        """Saves result data of a test"""
        datafilename = "result_test.json"
        if os.path.exists(datafilename):
            with open(datafilename, encoding="UTF-8") as datafile:
                data = json.load(datafile)
        else:
            data = {}
        data[image_filename] = {"testname": testname}
        with open(datafilename, "w", encoding="UTF-8") as datafile:
            json.dump(data, datafile)
