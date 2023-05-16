from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from .Task import Task, Options

"""
USAGE:

snr_instance = SNR( [...array of byte positions...] )

read_in_values = ReadH5(filename, batchsize=1000)

while read_in_values.next():
    snr_instance.push(read_in_values.get_batch_ptxts(), read_in_values.get_batch_samples())

snr = snr_instance.snr # this calls calculate for you.

snr_instance.plot() # this shows a pyplot of SNR, different colors per byte position, can call calculate and self.snr for you.
"""

@dataclass
class SNROptions(Options):
    byte_positions: list

class SNR(Task):

    def __init__(self, options: SNROptions):
        # VALUES TO SAMPLE LENGTH AND TRACES PROCESSED BY TOTAL AND BUCKET.
        self.trace_duration = 0
        # self.traces_processed = 0
        self.traces_processed_bins = None

        # ACCUMULATOR OF MEAN AND VARIANCE
        self._mean_accumulator = None
        self._S_accumulator = None
        # HIDDEN SNR AND BYTE POSITIONS.

        self._snr = None
        self._byte_positions = np.array(options.byte_positions, dtype=np.uint8)
        self._has_data = False
        self._indices = None

    def push(self, traces: np.ndarray, plaintexts: np.ndarray):
        """
        input: takes in a plaintexts, a sample, and an array of byte positions.
        output: None, updates accumulator values.
        description: takes in plaintext  and sample and calculates the accumulator values of 
            mean and variance (accumulated vertically) based on the byte values passed.
        """

        try:
            if not self._has_data:
                self._has_data = True
                self.trace_duration = traces.shape[1]
                self.traces_processed_bins = np.zeros((len(self._byte_positions), 256), dtype=np.uint16)

                self._mean_accumulator = np.zeros((len(self._byte_positions), 256, self.trace_duration), dtype=np.float32)
                self._S_accumulator = np.zeros((len(self._byte_positions), 256, self.trace_duration), dtype=np.float32)

            old_mean = np.empty((self.trace_duration), dtype=np.float32)
            for index in range(traces.shape[0]):
                for key_byte in self._byte_positions:
                    self.traces_processed_bins[key_byte, plaintexts[index, key_byte]] += 1
                    np.copyto(old_mean, self._mean_accumulator[key_byte, plaintexts[index, key_byte]])
                    self._mean_accumulator[key_byte, plaintexts[index, key_byte]] += ((traces[index] - old_mean) * 1.0) / self.traces_processed_bins[key_byte, plaintexts[index, key_byte]]
                    self._S_accumulator[key_byte, plaintexts[index, key_byte]] += ((traces[index] - old_mean)) * ((traces[index] - self._mean_accumulator[key_byte, plaintexts[index, key_byte]]))

        except IndexError as e:
            print(f"Failed to push... {e}")

    def calculate(self):
        """
        input: None. Prerequisite -> push values to SNR
        output: None. Sets hidden value self._snr
        description: Computes SNR for pushed traces, vertically.
        """
        try:
            signal = np.var(self._mean_accumulator, axis=1)
            noise = np.mean(self._S_accumulator, axis=1)
            snr = signal / noise
            self._snr = snr

        except IndexError as e:
            print(f"Failed to Calculate... {e}")

        del self._mean_accumulator
        del self._S_accumulator
        del self.traces_processed_bins

    def plot(self):
        """
        input: None. Uses self.snr
        output: pyplot graph
        description: plots SNR x-axis=trace_duration(time) y-axis=SNR(of that time)
        """
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g']
        plt.style.use("_mpl-gallery")
        for pos in range(len(self._byte_positions)):
            plt.plot(self.snr[pos], color=f"{colors[pos]}")
        plt.show()

    @property
    def snr(self):
        # calculate if not previously calculated
        if self._snr is None:
            self.calculate()
        # return the snr array
        return self._snr

    def get_results(self):
        return self.snr
    
    def get_heat_map_value(self):
        return np.amax(self.snr)
