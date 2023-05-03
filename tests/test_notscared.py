import os
import unittest
import numpy as np
import h5py
from src.notscared.statistics.histogram import Histogram_Method
from src.notscared.statistics.welford import Welford
from src.notscared.tasks.CPA import CPA, CPAOptions
from src.notscared.tasks.SNR import SNR, SNROptions
from src.notscared.file_handling.readh5 import ReadH5
# from src.notscared.file_handling.readh5 import ReadH5
from src.notscared.data_synthesis.correlation_data import CorrelationData
from src.notscared.models.HammingDistance import HammingDistance
from src.notscared.models.HammingWeight import HammingWeight



class TestNotScared(unittest.TestCase):
    pass
    # def test_example(self):
    #     self.assertTrue(2 == 2)


class TestHistogram(unittest.TestCase):

    def test_Mean_20_50(self):
        hist = Histogram_Method(50, 256)
        data = np.random.randint(32, 192, (20, 50), dtype=np.uint8)
        for row in data:
            hist.push(row)
        self.assertTrue(np.allclose(hist.mean, np.apply_along_axis(np.mean, 0, data)))

    def test_Variance_20_50(self):
        hist = Histogram_Method(50, 256)
        data = np.random.randint(32, 192, (20, 50), dtype=np.uint8)
        for row in data:
            hist.push(row)
        self.assertTrue(np.allclose(hist.variance, np.apply_along_axis(np.var, 0, data)))

    def test_standard_deviation_20_50(self):
        hist = Histogram_Method(50, 256)
        data = np.random.randint(32, 192, (20, 50), dtype=np.uint8)
        for row in data:
            hist.push(row)
        self.assertTrue(np.allclose(hist.std_dev, np.apply_along_axis(np.std, 0, data)))


class TestWelford(unittest.TestCase):
    def test_all_0(self):
        # all values when n=0
        welford = Welford()
        # math.isclose since we'll be dealing with floating point arithmetic
        self.assertAlmostEqual(0, welford.mean)
        self.assertAlmostEqual(0, welford.std_dev)
        self.assertAlmostEqual(0, welford.variance)
        self.assertAlmostEqual(0, welford.n)

    def test_mean_10(self):
        # mean when n=10
        welford = Welford()
        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for x in arr:
            welford.push(x)
        self.assertAlmostEqual(np.mean(arr), welford.mean)

    def test_variance_10(self):
        # variance when n=10
        welford = Welford()
        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for x in arr:
            welford.push(x)
        self.assertAlmostEqual(np.var(arr), welford.variance)

    def test_stddev_10(self):
        # standard deviation when n=10
        welford = Welford()
        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for x in arr:
            welford.push(x)
        self.assertAlmostEqual(np.std(arr), welford.std_dev)

    # Review Tests
    def test_mean_random_10(self):
        # Mean when n=10 and random integers are chosen
        welford = Welford()
        random_list = np.random.randint(32, 192, 10, dtype=np.uint8)
        for x in random_list:
            welford.push(x)
        self.assertAlmostEqual(np.mean(random_list), welford.mean)

    def test_variance_random_10(self):
        # Variance when n=10 and random integers are chosen
        welford = Welford()
        random_list = np.random.randint(32, 192, 10, dtype=np.uint8)
        for x in random_list:
            welford.push(x)
        self.assertAlmostEqual(np.var(random_list), welford.variance)

    def test_stddev_random_10(self):
        # Mean when n=10 and random integers are chosen
        welford = Welford()
        random_list = np.random.randint(32, 192, 10, dtype=np.uint8)
        for x in random_list:
            welford.push(x)
        self.assertAlmostEqual(np.std(random_list), welford.std_dev)


class TestCPA(unittest.TestCase):
    def test_hamming_weight_perfect_correlation(self):
        print("Generating Data.")
        cd = CorrelationData(50, 10000, HammingWeight())

        options = CPAOptions(
            byte_range=(0, 16)
        )
        cpa_instance = CPA(options)

        batch_num = 0

        while batch_num != 5:
            cd.generate_model_perfect_correlation_data()
            cpa_instance.push(cd.get_samples(), cd.get_plaintext())
            batch_num += 1
            # print("Batches pushed %d", batch_num)

        key_candidates = cpa_instance.get_results()

        np.set_printoptions(precision=1)
        print("KEY:\n", cd.key)
        print("KEY CANDIDATES:\n", key_candidates[0])
        print("KEY CANDIDATES CORRELATION:\n", key_candidates[1])

        # read.close_file()
        self.assertTrue(np.array_equal(key_candidates[0], cd.get_key()))


    def test_hamming_distance_perfect_correlation(self):
        print("Generating Data.")
        cd = CorrelationData(50, 10000, HammingDistance())

        options = CPAOptions(
            byte_range=(0, 16),
            leakage_model=HammingDistance()
        )

        cpa_instance = CPA(options)

        batch_num = 0

        while batch_num != 5:
            cd.generate_model_perfect_correlation_data()
            cpa_instance.push(cd.get_samples(), cd.get_plaintext())
            batch_num += 1
            # print("Batches pushed %d", batch_num)

        key_candidates = cpa_instance.get_results()

        np.set_printoptions(precision=1)
        print("KEY:\n", cd.key)
        print("KEY CANDIDATES:\n", key_candidates[0])
        print("KEY CANDIDATES CORRELATION:\n", key_candidates[1])

        # read.close_file()
        self.assertTrue(np.array_equal(key_candidates[0], cd.get_key()))

class TestSNR(unittest.TestCase):
    def test_snr_perfect_correlation(self):
        print("Generating Data.")
        cd = CorrelationData(50, 10000, HammingWeight())

        options = SNROptions(
            byte_positions = np.arange(0,16)
        )

        snr_instance = SNR(options)

        batch_num = 0

        while batch_num != 5:
            cd.generate_signal_and_noise_data()
            snr_instance.push(cd.get_samples(), cd.get_plaintext())
            batch_num += 1
            # print("Batches pushed %d", batch_num)

        snr_results = snr_instance.get_results()

        for byte in range(16):
            for i, snr in enumerate(snr_results[byte]):
                if (i - 24) % 300 < 16:
                    continue
                self.assertTrue(snr < 1.2)


            self.assertTrue(snr_results[byte][24 + byte * 300] > 1)


class TestReadH5(unittest.TestCase):
    def setUp(self):
        self.num_tile_x = 3
        self.num_tile_y = 3
        self.filename = "./test.h5"
        self.num_traces = 100
        self.trace_duration = 1000
        self.samples = np.random.randint(0, 255, (self.num_traces, self.trace_duration), dtype=np.uint8)
        self.plaintexts = np.random.randint(0, 255, (self.num_traces, 16), dtype=np.uint8)
        self.ciphertexts = np.random.randint(0, 255, (self.num_traces, 16), dtype=np.uint8)
        self.keys = np.random.randint(0, 255, (self.num_traces, 16), dtype=np.uint8)

        with h5py.File(self.filename, 'w') as hdf5_file:
            for x in range(self.num_tile_x):
                for y in range(self.num_tile_y):
                    # create datasets -- do these need a 4th dimension? --> shape = (n_tile_x, n_tile_y, number_of_traces, sample_length)
                    hdf5_file.create_dataset(f"traces/tile_{x}/tile_{y}/samples", data=self.samples, dtype=np.uint8)
                    hdf5_file.create_dataset(f"traces/tile_{x}/tile_{y}/plaintext", data=self.plaintexts, dtype=np.uint8)
                    hdf5_file.create_dataset(f"traces/tile_{x}/tile_{y}/key", data=self.keys, dtype=np.uint8)  # [[ptxt_byte_array_1], [...], ...]

    def tearDown(self):
        os.remove(self.filename)

    def test_read_h5(self):
        for x in range(self.num_tile_x):
            for y in range(self.num_tile_y):
                batch_size = 10
                read = ReadH5(self.filename, (x, y), batch_size)
                i = 0
                while read.next():
                    self.assertTrue(np.array_equal(read.get_batch_samples(), self.samples[i*batch_size:(i+1)*batch_size]))
                    self.assertTrue(np.array_equal(read.get_batch_ptxts(), self.plaintexts[i*batch_size:(i+1)*batch_size]))
                    self.assertTrue(np.array_equal(read.get_batch_keys(), self.keys[i*batch_size:(i+1)*batch_size]))
                    i += 1

if __name__ == "__main__":
    unittest.main()
