import numpy as np

class CorrelationData:
    def __init__(self, num_traces, sample_size, hamming_weight=True):
        self.num_traces = num_traces
        self.sample_size = sample_size
        self.key = np.random.randint(0, 256, (16), dtype=np.uint8)
        self.plaintext = np.random.randint(0, 256, (num_traces, 16), dtype=np.uint8)
        self.samples = np.random.randint(32, 192, (self.num_traces, self.sample_size), dtype=np.uint8)
        self.hamming_weight = hamming_weight
        self.cursor = 0

    def generate_data(self):
        HW_LUT = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
                           1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                           1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                           2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                           1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                           2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                           2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                           3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8], dtype='uint8')

        

        def hw(x):
            return HW_LUT[x]

        for i in range(self.num_traces):
            self.samples[i] = hw(self.samples[i]).astype('uint8')

            for byte in range(16):
                perfect_sbox_leakage = HW_LUT[self.plaintext[i, byte] ^ self.key[byte]]
                self.samples[i, 24 + byte] = perfect_sbox_leakage

    def get_plaintext(self):
        return self.plaintext

    def get_samples(self):
        return self.samples

    def get_key(self):        
        return self.key