import unittest
import numpy as np
from src.notscared.statistics.histogram import Histogram_Method
from src.notscared.statistics.welford import Welford
from devtools.data_synthesis.correlation_data import CorrelationData
from src.notscared.distinguishers.cpa import CPA
from src.notscared.file_handling.readh5 import ReadH5

# import python_file_name.py


class TestNotScared(unittest.TestCase):
    def test_example(self):
        self.assertTrue(2 == 2)

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
        list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for x in list:
            welford.push(x)
        self.assertAlmostEqual(np.mean(list), welford.mean)

    def test_variance_10(self):
        # variance when n=10
        welford = Welford()
        list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for x in list:
            welford.push(x)
        self.assertAlmostEqual(np.var(list), welford.variance)

    def test_stddev_10(self):
        # standard deviation when n=10
        welford = Welford()
        list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for x in list:
            welford.push(x)
        self.assertAlmostEqual(np.std(list), welford.std_dev)

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
    def test_perfect_correlation_1(self):
        key = np.random.randint(0, 256, (16), dtype=np.uint8)
        plaintext = np.random.randint(0, 256, (100, 16), dtype=np.uint8)
        
        print("Generating Data.")
        cd = CorrelationData(100, 2000, key, plaintext, hamming_weight=True)
        cd.generate_data("test_data.h5")

        print("Reading file.")
        read = ReadH5("test_data.h5")
        
        cpa_instance = CPA((0, 1), 2000, True)

        num_batches = -1
        batch_num = 0
        while(read.next()):
            cpa_instance.push_batch(read.get_batch_samples(), read.get_batch_ptxts())
            batch_num += 1
            # print("Batches pushed %d", batch_num)
            if  batch_num == num_batches:
                break

        key_candidates = cpa_instance.get_key_candidates()
        
        print("KEY CANDIDATES:\n", key_candidates)
        print("KEY:\n", key)

        read.close_file()
        self.assertEqual(key_candidates[0], key[0])
        
        

if __name__ == "__main__":
    unittest.main()
