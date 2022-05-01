#!/usr/bin/env python3

# This script takes care of building the dynamic content of the report.
# If a specific part is specified, it will only build that part.
# If no part is specified, it will build all parts.

import click
import os
import numpy as np
import matplotlib.pyplot as plt
from requests import options
from torch import convolution

parts = ["convolution_example"]

BUILD_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/build"

# Use the click library to define the command line interface
@click.command()
@click.option("--part", default=None, help="Part to build")
def main(part):
    # Make sure the build folder exists
    if not os.path.exists(BUILD_FOLDER):
        print("No build folder found. Creating...")
        os.makedirs(BUILD_FOLDER)


    # Build the report
    if part is None:
        for part in parts:
            build_part(part)
    else:
        build_part(part)

def bmatrix(a):
    """Returns a LaTeX bmatrix
    Source: https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)


def build_part(part):
    # Make sure that there exists a part folder
    part_folder = BUILD_FOLDER + "/" + part
    if not os.path.exists(part_folder):
        os.makedirs(part_folder)
    else:
        # Remove the old files
        for file in os.listdir(part_folder):
            os.remove(part_folder + "/" + file)


    # Change to the part folder
    os.chdir(part_folder)

    # Build the part
    if part == "convolution_example":
        build_convolution_example()


def build_convolution_example():
    from scipy import signal
    # Build the convolution example
    print("Building convolution example...")
    I = np.array(
        [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 7, 9],
            [1, 0, 0, 0, 0, 0, 8, 10],
        ]
    )
    K = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    )
    # Save I and K as latex files
    with open("I.tex", "w") as f:
        f.write(bmatrix(I))
    
    with open("K.tex", "w") as f:
        f.write(bmatrix(K))

    convolution = signal.convolve2d(I, K, mode="valid")

    # Save the convolution as a latex file
    with open("convolution.tex", "w") as f:
        f.write(bmatrix(convolution))

    # Plot the convolution
    plt.imshow(convolution)
    # Remove the axes
    plt.axis('off')
    # Show a legend
    plt.colorbar()
    plt.savefig("convolution.png")

    # Remove the old legend
    plt.clf()


    plt.imshow(I)
    plt.axis('off')
    plt.colorbar()
    plt.savefig("I.png")

    plt.clf()

    # Calculate a padded version of I with 1 pixel border
    I_padded = np.pad(I, 1, 'constant')

    with open("I_padded.tex", "w") as f:
        f.write(bmatrix(I_padded))

    convolution_padded = signal.convolve2d(I_padded, K, mode="valid")

    with open("convolution_padded.tex", "w") as f:
        f.write(bmatrix(convolution_padded))
    
    plt.imshow(convolution_padded)
    plt.axis('off')
    plt.colorbar()
    plt.savefig("convolution_padded.png")


    
    
if __name__ == "__main__":
    main()
