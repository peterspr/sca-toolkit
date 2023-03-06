import numpy as np
import h5py as h5
import sys


def create_random_dataset(filename, x_shape="1", y_shape="1"):
    """
    @param filename: file name to be created with traces data set in it.
    @param x_shape: is the number of x tiles to generate
    @param y_shape: is the number of y tiles to generate

    USAGE:
    In CLI: 'python3 fake_h5_model.py filename x_shape y_shape'
    """
    x_shape = int(x_shape)
    y_shape = int(y_shape)
    dset = [[np.random.randint(50, 200, 20000, dtype=np.uint8) for i in range(y_shape)] for j in range(x_shape)]
    for i in range(x_shape):
        print(f"x: {i}")
        for j in range(y_shape):
            print(f"   y: {j}: {dset[i][j]}")

    with h5.File(filename, "w") as f:
        f.create_dataset("traces", data=dset, dtype=np.uint8)


if __name__ == "__main__":

    create_random_dataset(sys.argv[1], sys.argv[2], sys.argv[3])
