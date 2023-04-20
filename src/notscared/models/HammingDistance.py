import numpy as np
from .Model import Model
from .utils import AES_SBOX, HW_LUT

# init array with key candidates 0-255 for one byte
# Needs to be a column vector to broadcast
keys = np.arange(256).reshape(-1, 1)


class HammingDistance(Model):
    def create_leakage_table(self, plaintext_bytes: np.ndarray):
        sbox_in = np.bitwise_xor(plaintext_bytes, keys)
        sbox_out = AES_SBOX[sbox_in]
        result = HW_LUT[np.bitwise_xor(sbox_out, sbox_in)]
        return result
