import os
from math import floor, log10

# Find the image folder one level up from this file
IMAGE_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/../images"

DATA_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/../data"

# Find the data folder in the ~/kaggle-data/HAM10000
_home = os.path.expanduser("~")
HAM10000_DATA_FOLDER = _home + "/kaggle-data/HAM10000"

def bmatrix(a, format=None):
    """Returns a LaTeX bmatrix
    Source: https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if format is None:
        format = lambda x: x
    if len(a.shape) > 2:
        raise ValueError("bmatrix can at most display two dimensions")
    lines = str(a).replace("[", "").replace("]", "").splitlines()
    rv = [r"\begin{bmatrix}"]
    numbers = [[format(x) for x in l.split()] for l in lines]
    rv += ["  " + " & ".join(n) + r"\\" for n in numbers]
    rv += [r"\end{bmatrix}"]
    return "\n".join(rv)


def scientific_notation(x, precision=2):
    """Returns a string representation of a number in scientific notation.
    Source: https://stackoverflow.com/questions/3410976/how-to-format-numbers-in-python-with-significant-figures-and-without-exponential
    Modified to return LaTeX string.
    """
    if x == 0:
        return "0"
    elif x < 0:
        return "-" + scientific_notation(-x)
    else:
        exponent = int(floor(log10(x)))
        mantissa = x / 10**exponent
        return f"{round(mantissa, precision)} \cdot 10^{{{exponent}}}"
