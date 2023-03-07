import unittest
import numpy as np
from src.notscared.preprocessing.SNR import SNR
from src.notscared.file_handling.readh5 import ReadH5
from devtools.data_synthesis.correlation_data import CorrelationData

class testSNR(unittest.TestCase):
    def test_1(self):

        fn = "test_data.h5"

        # generate data
        cd = CorrelationData(1000, 10)
        cd.generate_data(fn)

        # declare class instances
        reader = ReadH5(fn)
        signal_noise = SNR(1000)

        # push to SNR
        while reader.next():
            signal_noise.push(reader.get_batch_ptxts(), reader.get_batch_samples())

        # calculate
        signal_noise.calculate()

        # numpy SNR
        stacked_variance = np.stack(reader.get_batch_samples(), axis=-1)
        stacked_variance = np.apply_along_axis(np.var, axis=0, arr=stacked_variance)
        stacked_mean = np.stack(reader.get_batch_samples(), axis=-1)
        stacked_mean = np.apply_along_axis(np.mean, axis=0, arr=stacked_mean)

        np_snr = np.var(stacked_mean) / np.mean(stacked_variance)

        print(signal_noise.snr, np_snr)


if __name__ == '__main__':
    test = testSNR()
    test.test_1()
