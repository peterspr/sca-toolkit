import numpy as np
from src.notscared.statistics.welford import Welford
import fire

class SNR:
    def __init__(self):
        self._size = 0
        self._welfords = None
        self._snr = None

    def push(self, sample):
        if self._size == 0:
            self._size += sample.size
            self._welfords = [Welford() for _ in range(self._size)]
            self._snr = np.empty(self._size)
        else:
            if self._size < sample.size:
                for _ in range(sample.size - self._size):
                    self._welfords.append(Welford())
                self._size += sample.size - self._size
                self._snr.reshape(self._size)

        for index in range(self._size):
            self._welfords[index].push(sample[index])

    @property
    def snr(self):
        for index, w in enumerate(self._welfords):
            signal = w.mean
            noise = w.std_dev
            self._snr[index] = signal / noise
        return self._snr

    def get_features_of_sample(self, sample):
        return sample[sample >= [w.mean for w in self._welfords]]

    def plot_signal_vs_noise(self, samples):
        pass


if __name__ == '__main__':
    fire.Fire(SNR)


