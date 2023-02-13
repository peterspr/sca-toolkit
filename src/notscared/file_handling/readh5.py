import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

"""
    Usage:
        - Class:
            my_h5 = ReadH5(filename, 0, 99)
            my_h5.print_file()
            ...
        - CLI:
            `python3 readh5.py --file=filename --start=arg --stop=arg COMMAND arg1 ...`
            ex:
                `python3 readh5.py --file=my_data.h5 --start=0 --stop=4 get_key_from_index 1`

            `python3 readh5.py --file=filename` -- would read the whole file in, but with CLI you must pass a COMMAND! So...
            `python3 readh5.py --file=filename print_file`

"""


class ReadH5:
    def __init__(self, file, batch_size=10):
        self._file = file
        self._ptxt = None
        self._k = None
        self._samples = None
        self._cursor = 0
        self._file = h5.File(file, "r")
        self._batch_size = batch_size

    def next(self):
        self._k = self._file["traces/k"][self._cursor:self._cursor + self._batch_size]
        self._ptxt = self._file["traces/ptxt"][self._cursor:self._cursor + self._batch_size]
        self._samples = self._file["traces/samples"][self._cursor:self._cursor + self._batch_size]
        self._cursor += self._batch_size

        if len(self._k) == 0:
            print("End of Array")
            self.close_file()
            return False

        return True

    def close_file(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def print_batch(self):
        print("KEYS\n", self._k)
        print("PLAINTEXTS\n", self._ptxt)
        print("SAMPLES\n", self._samples)

    def get_batch_keys(self):
        return self._k

    def get_batch_ptxts(self):
        return self._ptxt

    def get_batch_samples(self):
        return self._samples

    # def get_key_from_index(self, index):
    #     return self._k[index]

    # def get_ptxt_from_index(self, index):
    #     return self._ptxt[index]

    # def get_sample_from_index(self, index):
    #     return self._sample[index]

    def plot_samples(self):
        plt.style.use("_mpl-gallery")
        for sample in self._samples:
            plt.plot(sample, color="b")
        plt.show()
