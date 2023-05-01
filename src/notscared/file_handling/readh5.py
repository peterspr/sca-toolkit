import h5py as h5
import matplotlib.pyplot as plt


class ReadH5:
    def __init__(self, file, tile=(0,0), batch_size=10):
        self._file = file
        self._ptxt = None
        self._k = None
        self._samples = None
        self._cursor = 0
        self._file = h5.File(file, "r")
        self._batch_size = batch_size
        self._x = tile[0]
        self._y = tile[1]
        self.n_tiles = self._get_number_of_tiles()

    def next(self, slice_start=None, slice_end=None, step=1):
        # update arrays based on x, y and cursor.
        self._k = self._file[f"traces/tile_{self._x}/tile_{self._y}/key"][self._cursor:self._cursor + self._batch_size]
        self._ptxt = self._file[f"traces/tile_{self._x}/tile_{self._y}/plaintext"][self._cursor:self._cursor + self._batch_size]

        if slice_start is None and slice_end is None and step == 1:
            self._samples = self._file[f"traces/tile_{self._x}/tile_{self._y}/samples"][self._cursor:self._cursor + self._batch_size]
        elif slice_start is None and slice_end is None and step != 1:
            self._samples = self._file[f"traces/tile_{self._x}/tile_{self._y}/samples"][self._cursor:self._cursor + self._batch_size][::step]
        else:
            self._samples = self._file[f"traces/tile_{self._x}/tile_{self._y}/samples"][self._cursor:self._cursor + self._batch_size][slice_start:slice_end:step]

        self._cursor += self._batch_size

        # if end of data on last tiles close file and return false.
        if len(self._k) == 0:
            # print("End of File")
            self.close_file()
            return False

        return True

    def close_file(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def get_batch_keys(self):
        return self._k

    def get_batch_ptxts(self):
        return self._ptxt

    def get_batch_samples(self):
        return self._samples

    def plot_samples(self):
        plt.style.use("_mpl-gallery")
        for sample in self._samples:
            plt.plot(sample, color="b")
        plt.show()

    def _get_number_of_tiles(self):
        x = 0
        y = 0
        y_save = 0
        exists = True
        while (exists):
            try:
                self._file[f"traces/tile_{x}/tile_{y}"]
                y += 1
            except KeyError:
                x += 1
                y_save = y
                y = 0
                try:
                    self._file[f"traces/tile_{x}/tile_{y}"]
                except KeyError:
                    exists = False
        y = y_save
        return (x, y)

    @property
    def tiles(self):
        return self.n_tiles
