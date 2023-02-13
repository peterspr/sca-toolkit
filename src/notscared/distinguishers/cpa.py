import numpy as np
from utils.leakage import create_leakage_table
from statistics.welford import Welford

class CPA:
    def __init__(self, byte_range, trace_duration, use_hamming_weight=True):
        # If false it will use hamming distance
        self.use_hamming_weight = use_hamming_weight
        # Between 0 and 15
        self.byte_range = byte_range
        self.trace_duration = trace_duration
        self.traces_processed = 0
        # Statistical Accumulators
        self.trace_accs = np.array([Welford() for i in range(trace_duration)])
        self.leakage_accs = np.array([[Welford() for i in range(256)] for j in range(16)])
        self.product_acc = np.array([[[Welford() for i in range(trace_duration)] for j in range(256)] for k in range(16)])

    def push_batch(self, traces:np.ndarray, plaintext:np.ndarray):
        print(f'traces_processed: {self.traces_processed}')
        self.traces_processed += traces.shape[0]
        
        # leakage_cube = np.zeros((16, 256, traces.shape[1]), np.ubyte)
        
        leakage_cube = np.zeros((traces.shape[0], 16, 256), np.ubyte)

        for trace_index in range(traces.shape[0]):
            leakage_cube[trace_index] = create_leakage_table(plaintext[trace_index], self.use_hamming_weight)
        for byte in range(self.byte_range[0], self.byte_range[1]):
            # leakage_cube[byte] = create_leakage_table(plaintext[byte, :], self.use_hamming_weight)
            for leakage_key_hypothesis in range(leakage_cube.shape[1]):
                self.leakage_accs[byte, leakage_key_hypothesis].push_array(leakage_cube[byte, leakage_key_hypothesis])
        for point_in_time in range(traces.shape[1]):
            self.trace_accs[point_in_time].push_array(traces[:, point_in_time])
            for byte in range(self.byte_range[0], self.byte_range[1]):
                for leakage_key_hypothesis in range(leakage_cube.shape[2]):
                    self.product_acc[byte, leakage_key_hypothesis, point_in_time].push_array(np.multiply(traces[:, point_in_time], leakage_cube[:, byte, leakage_key_hypothesis]))

    def calculate(self):
        results = np.zeros((16, 256, self.trace_duration))
        for trace_acc in range(self.trace_accs.shape[0]):
            for byte in range(self.byte_range[0], self.byte_range[1]):
                for leakage_acc in range(self.leakage_accs.shape[0]):
                    sum_of_products = self.product_acc[byte, leakage_acc, trace_acc].mean * self.traces_processed
                    sum_traces = self.trace_accs[trace_acc].mean * self.traces_processed
                    sum_leakages = self.leakage_accs[byte, leakage_acc].mean * self.traces_processed
                    product_of_sums = sum_traces*sum_leakages
                    numerator = sum_of_products - product_of_sums
                    denominator = self.trace_accs[trace_acc].std_dev * self.leakage_accs[byte, leakage_acc].std_dev
                    results[byte, leakage_acc, trace_acc] = numerator/denominator
        print('.')
        return results
    
    def get_key_candidates(self):
        key_candidates = np.zeros((16), int)
        results = self.calculate()
        for byte in range(self.byte_range[0], self.byte_range[1]):
            candidates_along_bytes = np.amax(results[byte], axis=1)
            key_candidates[byte] = np.argmax(candidates_along_bytes)
        return key_candidates
