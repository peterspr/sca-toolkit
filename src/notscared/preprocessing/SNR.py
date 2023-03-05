import numpy as np
import matplotlib.pyplot as plt
import random

"""
    1. Take in a plaintxt with samples
    2. Get a ptxt with byte 0x00 in the 0th place make a bucket
    3. Do step 2 for each ptxt/sample with 0x01 -> 0xFF in 0th place
    4. Update mean and variance in each bucket as an np.array 
    """


class SNR:

    def __init__(self, n_traces: int):
        # declare an array of 256 (0x00 -> 0xFF), number of traces in a sample, and then [0] == mean and [1] == variance
        self._accumulator = np.empty((256, n_traces, 2))
        # declare n_traces to be accessed by self
        self.n_traces = n_traces

        self._snr = None

    def push(self, ptxts: np.ndarray, samples: np.ndarray):
        """
        input: takes in an array of plaintexts and an array of samples
        output: None, updates accumulator values.
        description: takes in plaintext array and samples array and calculates the accumulator values of 
            mean and variance based on the first byte value and trace point in time. (vertically)

        """
        try:
            # for each ptxt in
            for ptxt in range(ptxts.shape[0]):
                p_byte = int(ptxts[ptxt][0])  # first byte of ptxt
                # for each trace
                for trace_index in range(self.n_traces):
                    # index in by first byte value, trace value (vertical column), mean value access OR variance value access
                    current_mean = self._accumulator[p_byte][trace_index][0]
                    current_S = self._accumulator[p_byte][trace_index][1]

                    # get the new mean by adding (xi - x_mean)/n
                    new_mean = current_mean + (samples[ptxt][trace_index] - current_mean) * 1.0 / self.n_traces
                    # get new S by multiplying (xi - pop_mean) and (xi - sample_mean) and adding it to current_S
                    new_S = current_S + (samples[ptxt][trace_index] - current_mean) * (
                            samples[ptxt][trace_index] - new_mean)

                    # set accumulator values for next loop.
                    self._accumulator[p_byte][trace_index][0] = new_mean
                    self._accumulator[p_byte][trace_index][1] = new_S

        except IndexError as e:
            print(f"Failed to push... {e}")

    def calculate(self):

        try:
            # vertorize divide with lambda function on vertical variances
            apply_variance = lambda x: x / self.n_traces
            apply_variance(self._accumulator[:][:][1])

            # the Vertical Variance of the Vertical Means
            signal = np.var(self._accumulator[:][:][0])

            # the Vertical Mean of the Vertical Variance
            noise = np.mean(self._accumulator[:][:][1])

            # return SNR
            self._snr = signal / noise

        except IndexError as e:
            print(f"Failed to Calculate... {e}")


    def plot_snr(self, plot_values):
        s = self.snr
        vertical_plot_values = np.stack(plot_values, axis=-1)
        plt.style.use("_mpl-gallery")
        for value in vertical_plot_values:
            plt.plot(value[value > s], color="r")
            plt.plot(value[value < s], color="b")
        plt.show()

    @property
    def snr(self):
        if self._snr is None:
            self.calculate()
        return self._snr

