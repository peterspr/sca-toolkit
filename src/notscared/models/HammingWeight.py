import numpy as np
from .Model import Model
from .utils import HW_LUT

# init array with key candidates 0-255 for one byte
# Needs to be a column vector to broadcast
keys = np.arange(256).reshape(-1, 1)


class HammingWeight(Model):
    def create_leakage_table(self, plaintext_bytes: np.ndarray):
        intermediate = np.bitwise_xor(plaintext_bytes, keys)
        intermediate[:] = HW_LUT[intermediate]
        return intermediate

    def create_perfect_correlation(self, plaintext, key):
        intermediate = np.bitwise_xor(plaintext, key)
        intermediate[:] = HW_LUT[intermediate]
        return intermediate
