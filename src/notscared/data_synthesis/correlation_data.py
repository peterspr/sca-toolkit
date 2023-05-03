import numpy as np
from src.notscared.models.Model import Model

class CorrelationData:
    def __init__(self, num_traces, sample_size, model: Model):
        self.num_traces = num_traces
        self.sample_size = sample_size
        self.key = np.random.randint(0, 256, (16), dtype=np.uint8)
        self.plaintext = np.random.randint(0, 256, (num_traces, 16), dtype=np.uint8)
        self.samples = np.random.randint(0, 8, (self.num_traces, self.sample_size), dtype=np.uint8)
        self.model = model

    def generate_model_perfect_correlation_data(self):
        for i in range(self.num_traces):
            perfect_correlation = self.model.create_perfect_correlation(self.plaintext[i], self.key)
            for byte in range(16):
                self.samples[i, 24 + byte] = perfect_correlation[byte]

    def generate_signal_and_noise_data(self):
        self.samples = np.random.randint(10, 255, (self.num_traces, self.sample_size), dtype=np.uint8)
        for i in range(self.num_traces):
            for byte in range(16):
                self.samples[i, 24 + byte * 300] = self.plaintext[i, byte]

    def get_plaintext(self):
        return self.plaintext

    def get_samples(self):
        return self.samples

    def get_key(self):
        return self.key
