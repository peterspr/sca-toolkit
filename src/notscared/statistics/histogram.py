import numpy as np


class Histogram_Method:
    """
    Implements a histogram method for calculating first and higher order moments
    across a 2d array (number of trace samples) x (number of bins).

    Attributes
        num_samples_per_trace (int): number of samples in a single sample array
        num_bins (int): The number of individual arrays (bins) representing the frequency
            of each element in a single sample index <- can be optimized to fit.
        histogram (uint64 numpy array): Array of size (samples, bins) holding arrays (histograms)
            representing the frequency of each element for each sample
        mean (float128 or longdouble): Array of means for each sample index
        variance (float128 or longdouble): Array of variances for each sample index
        std_dev (float128 or longdouble): Array of standard deviations for each sample index


    Usage:
        hist = Histogram_Method(num_samples_per_trace, 256)
        for trace in traces:
            hist.push(trace['samples'])
        print(hist.mean)
        print(hist.variance)
        print(hist.std_dev)
        print(hist.get_moments())

        OR
        hist = Histogram_Method(num_samples_per_trace, 256)
        for row in array:
            hist.push(row)

    """

    def __init__(self, num_samples_per_trace=0, num_bins=256):
        self.num_samples_per_trace = num_samples_per_trace
        self.num_bins = num_bins
        self.histogram = np.zeros((self.num_samples_per_trace, self.num_bins), dtype=np.uint64)
        self.mean_cache = np.zeros((self.num_bins), dtype=np.uint64)
        self.variance_cache = np.zeros((self.num_bins), dtype=np.uint64)
        self.standard_dev_cache = np.zeros((self.num_bins), dtype=np.uint64)
        self._up_to_date = False

    def set_num_samples_per_trace(self, num_samples):
        self.num_samples_per_trace = num_samples
        self.histogram = np.zeros((self.num_samples_per_trace, self.num_bins), dtype=np.uint64)

    def push(self, trace_samples):
        for sample_num in range(self.num_samples_per_trace):
            self.histogram[sample_num][trace_samples[sample_num]] += 1
        self._up_to_date = False

    @property
    def mean(self):
        if self._up_to_date:
            return self.mean_cache
        totalObservations = np.apply_along_axis(np.sum, 1, self.histogram[:][:])
        mulOperands = np.multiply(self.histogram[:][:], np.uint64(range(self.num_bins)), dtype=np.uint64)
        meanSum = np.sum(mulOperands, 1, dtype=np.uint64)
        return np.divide(meanSum, totalObservations, where=totalObservations != 0)

    @property
    def variance(self):
        if self._up_to_date:
            return self.variance_cache
        totalObservations = np.apply_along_axis(np.sum, 1, self.histogram[:][:])
        centeredDiff = np.subtract(np.vstack([np.uint64(range(self.num_bins))] * self.num_samples_per_trace), np.vstack([self.mean] * self.num_bins).T)
        centeredDiff_squared = np.power(centeredDiff, 2, dtype=np.longdouble)
        mulOperands = np.multiply(self.histogram[:][:], centeredDiff_squared, dtype=np.longdouble)
        varSum = np.sum(mulOperands, 1, dtype=np.longdouble)
        return np.divide(varSum, totalObservations, where=totalObservations != 0)

    @property
    def std_dev(self):
        if self._up_to_date:
            return self.std_dev_cache
        return np.sqrt(self.variance, dtype=np.longdouble)

    def get_moments(self):
        totalObservations = np.apply_along_axis(np.sum, 1, self.histogram[:][:])
        mulOperands = np.multiply(self.histogram[:][:], np.uint64(range(self.num_bins)), dtype=np.uint64)
        meanSum = np.sum(mulOperands, 1, dtype=np.uint64)
        mean_list = np.divide(meanSum, totalObservations, where=totalObservations != 0)

        totalObservations = np.apply_along_axis(np.sum, 1, self.histogram[:][:])
        centeredDiff = np.subtract(np.vstack([np.uint64(range(self.num_bins))] * self.num_samples_per_trace), np.vstack([mean_list] * self.num_bins).T)
        centeredDiff_squared = np.power(centeredDiff, 2, dtype=np.longdouble)
        mulOperands = np.multiply(self.histogram[:][:], centeredDiff_squared, dtype=np.longdouble)
        varSum = np.sum(mulOperands, 1, dtype=np.longdouble)
        variance_list = np.divide(varSum, totalObservations, where=totalObservations != 0)

        standard_deviation = np.sqrt(variance_list, dtype=np.longdouble)

        return mean_list, variance_list, standard_deviation
