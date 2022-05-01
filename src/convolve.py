import numpy as np


def convolve(I, K, stride=1, padding=0):
    if padding != 0:
        # Creating a padded version of I
        # First allocate memory for the padded version of I
        I_padded = np.zeros((I.shape[0] + 2 * padding, I.shape[1] + 2 * padding))
        # Then copy the original I into the middle of the padded version of I
        I_padded[padding : I.shape[0] + padding, padding : I.shape[1] + padding] = I
    else:
        I_padded = I

    h_out = (I.shape[0] + 2 * padding - K.shape[0] ) // stride + 1
    w_out = (I.shape[1] + 2 * padding - K.shape[1] ) // stride + 1

    # Initialize the output
    O = np.zeros((h_out, w_out))
    for y in range(h_out):
        for x in range(w_out):
            O[y, x] = np.sum(
                I_padded[
                    y * stride : y * stride + K.shape[0],
                    x * stride : x * stride + K.shape[1],
                ]
                * K
            )
    return O
