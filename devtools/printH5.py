import h5py
import fire

"""
Usage: `python3 printH5.py --file=your_file_path`
"""


def print_hdf5_by_group_recursively(group, indent=0):
    # loop through group for other groups
    for key in group.keys():
        # get item in group
        item = group[key]
        print(" " * indent + key)
        # if a dataset print
        if isinstance(item, h5py.Dataset):
            print(" " * (indent + 2) + "Dataset: ", item.shape, item.dtype, "\n", group[key][:], "\n")
        # if a group then call on next group with higher print indent
        elif isinstance(item, h5py.Group):
            print_hdf5_by_group_recursively(item, indent + 2)


def print_hdf5_file_structure(hdf5_name):
    # get initial group from hdf5 file
    with h5py.File(hdf5_name, "r") as f:
        # call recursive print
        print_hdf5_by_group_recursively(f)


def printH5(file):
    print_hdf5_file_structure(file)


if __name__ == "__main__":
    fire.Fire(printH5)
