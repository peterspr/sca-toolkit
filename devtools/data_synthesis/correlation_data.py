import numpy as np
import h5py
from src.notscared.utils.leakage import Sbox


class CorrelationData:
    def __init__(self, num_traces, sample_size, hamming_weight=True):
        self.num_traces = num_traces
        self.sample_size = sample_size
        self.key = np.random.randint(0, 256, (16), dtype=np.uint8)
        self.plaintext = np.random.randint(0, 256, (num_traces, 16), dtype=np.uint8)
        self.hamming_weight = hamming_weight

    def generate_data(self, file_name):
        HW_LUT = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
                           1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                           1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                           2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                           1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                           2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                           2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                           3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8], dtype='uint8')

        f = h5py.File(file_name, "w")

        f.create_dataset("traces/tile_0/tile_0/samples", data=np.zeros((self.num_traces, self.sample_size), dtype=np.uint8), dtype=np.uint8)
        f.create_dataset("traces/tile_0/tile_0/plaintext", data=np.zeros((self.num_traces, 16), dtype=np.uint8), dtype=np.uint8)
        f.create_dataset("traces/tile_0/tile_0/key", data=np.zeros((self.num_traces, 16), dtype=np.uint8), dtype=np.uint8)

        def hw(x):
            return HW_LUT[x]

        for i in range(self.num_traces):
            f["traces/tile_0/tile_0/plaintext"][i] = self.plaintext[i]
            f["traces/tile_0/tile_0/key"][i] = self.key
            temp_samples = np.random.randint(32, 192, (self.sample_size), dtype=np.uint8)

            temp_samples = hw(temp_samples).astype('uint8')

            for byte in range(16):
                perfect_sbox_leakage = HW_LUT[self.plaintext[i, byte] ^ self.key[byte]]
                temp_samples[24 + byte] = perfect_sbox_leakage

            f["traces/tile_0/tile_0/samples"][i] = temp_samples

        f.close()
