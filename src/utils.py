import os

# Find the image folder one level up from this file
IMAGE_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/../images"

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
