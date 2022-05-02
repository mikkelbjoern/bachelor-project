import numpy as np
from src.utils import bmatrix, IMAGE_FOLDER
import matplotlib.pyplot as plt
from src.convolve import convolve

def build_kernel_examples():
    image = np.array(plt.imread(IMAGE_FOLDER + "/temple.jpg"))
    image = image / 255.0
    # rescale the image
    image_gray = np.mean(image, axis=2)
    N = 3
    image_scaled = convolve(
        image_gray,
        np.array([[1 / N for _ in range(N)] for _ in range(N)]),
        stride=N,
        padding=0,
    )
    # Save the image
    plt.imsave("image_scaled.jpg", image_scaled, cmap="binary")
    plt.clf()

    K_edge = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # Write the kernel to a file
    with open("K_edge.txt", "w") as file:
        file.write(bmatrix(K_edge))

    edges = convolve(image_scaled, K_edge, stride=2, padding=1)
    plt.imsave("edges.jpg", edges, cmap="binary")
    plt.clf()

    K_horizontal = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    # Write the kernel to a file
    with open("K_horizontal.txt", "w") as file:
        file.write(bmatrix(K_horizontal))

    horizontal = convolve(image_scaled, K_horizontal, stride=2, padding=1)
    plt.imsave("horizontal.jpg", horizontal, cmap="binary")
    plt.clf()

    K_vertical = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    # Write the kernel to a file
    with open("K_vertical.txt", "w") as file:
        file.write(bmatrix(K_vertical))

    vertical = convolve(image_scaled, K_vertical, stride=2, padding=1)
    plt.imsave("vertical.jpg", vertical, cmap="binary")
    plt.clf()

    K_laplacian_of_gaussian = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0],
        ]
    )
    # Write the kernel to a file
    with open("K_laplacian_of_gaussian.txt", "w") as file:
        file.write(bmatrix(K_laplacian_of_gaussian))

    laplacian_of_gaussian = convolve(
        image_scaled, K_laplacian_of_gaussian, stride=2, padding=1
    )
    plt.imsave("laplacian_of_gaussian.jpg", laplacian_of_gaussian, cmap="binary")
    plt.clf()