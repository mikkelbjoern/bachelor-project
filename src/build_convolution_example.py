import numpy as np
from src.utils import bmatrix
from src.convolve import convolve
import matplotlib.pyplot as plt


def build_convolution_example():
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
    K = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    # Save I and K as latex files
    with open("I.tex", "w") as f:
        f.write(bmatrix(I))

    with open("K.tex", "w") as f:
        f.write(bmatrix(K))

    convolution = convolve(I, K)

    # Save the convolution as a latex file
    with open("convolution.tex", "w") as f:
        f.write(bmatrix(convolution))

    # Plot the convolution
    plt.imshow(convolution)
    # Remove the axes
    plt.axis("off")
    # Show a legend
    plt.colorbar()
    plt.savefig("convolution.png")

    # Remove the old legend
    plt.clf()

    plt.imshow(I)
    plt.axis("off")
    plt.colorbar()
    plt.savefig("I.png")

    plt.clf()

    # Calculate a padded version of I with 1 pixel border
    I_padded = np.pad(I, 1, "constant")

    with open("I_padded.tex", "w") as f:
        f.write(bmatrix(I_padded))

    # convolution_padded = signal.convolve2d(I_padded, K, mode="valid")
    convolution_padded = convolve(I, K, padding=1)

    with open("convolution_padded.tex", "w") as f:
        f.write(bmatrix(convolution_padded))

    plt.imshow(convolution_padded)
    plt.axis("off")
    plt.colorbar()
    plt.savefig("convolution_padded.png")

    # Calculate with stride
    # convolution_stride = strideConv(I_padded,K,2)
    convolution_stride = convolve(I_padded, K, stride=2)

    with open("convolution_stride.tex", "w") as f:

        def formatting(x):
            integer = float(x)
            integer = int(integer)
            return str(integer)

        f.write(bmatrix(convolution_stride, format=formatting))

    print()
    with open("final_convolution_size.tex", "w") as f:
        f.write(str(len(convolution_stride) * len(convolution_stride[0])))

    with open("I_size.tex", "w") as f:
        f.write(str(len(I) * len(I[0])))

    plt.clf()
    plt.imshow(convolution_stride)
    plt.axis("off")
    plt.colorbar()
    plt.savefig("convolution_stride.png")
