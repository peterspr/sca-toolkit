import numpy as np
import math as math
from src.notscared.utils.leakage import create_leakage_table
# from statistics.welford import Welford

class CPA:
    def __init__(self, byte_range, trace_duration, use_hamming_weight=True):
        # If false it will use hamming distance
        self.use_hamming_weight = use_hamming_weight
        # Between 0 and 15
        self.byte_range = byte_range
        self.trace_duration = trace_duration
        self.traces_processed = 0
        # Statistical Accumulators
        # trace or leakage acc[..., 0-1] -- 0: sum of vals, 1: sum of squared vals
        self.trace_accs = np.zeros(shape=(trace_duration, 2), dtype=np.float32)
        self.leakage_accs = np.zeros(shape=(16, 256, 2), dtype=np.float32)
        self.product_acc = np.zeros(shape=(16, 256, trace_duration), dtype=np.float32)

    def push_batch(self, traces:np.ndarray, plaintext:np.ndarray):
        self.traces_processed += traces.shape[0]

        leakage_cube = np.zeros((traces.shape[0], 16, 256), np.ubyte)

        for trace_index in range(traces.shape[0]):
            leakage_cube[trace_index] = create_leakage_table(plaintext[trace_index], self.use_hamming_weight)
        for byte in range(self.byte_range[0], self.byte_range[1]):
            for leakage_key_hypothesis in range(leakage_cube.shape[2]):
                self.leakage_accs[byte, leakage_key_hypothesis, 0] += np.sum(leakage_cube[:, byte, leakage_key_hypothesis])
                self.leakage_accs[byte, leakage_key_hypothesis, 1] += np.sum(np.square(leakage_cube[:, byte, leakage_key_hypothesis], dtype=np.float32))
        for point_in_time in range(traces.shape[1]):
            self.trace_accs[point_in_time, 0] += np.sum(traces[:, point_in_time])
            self.trace_accs[point_in_time, 1] += np.sum(np.square(traces[:, point_in_time]))
            for byte in range(self.byte_range[0], self.byte_range[1]):
                for leakage_key_hypothesis in range(leakage_cube.shape[2]):
                    trace_slice = traces[:, point_in_time]
                    leakage_slice = leakage_cube[:, byte, leakage_key_hypothesis]
                    product_arr = np.multiply(trace_slice, leakage_slice, dtype=np.float32)
                    self.product_acc[byte, leakage_key_hypothesis, point_in_time] += np.sum(product_arr)

        print(f'traces_processed: {self.traces_processed}')

    def calculate(self):
        results = np.zeros((16, 256, self.trace_duration))
        for trace_acc in range(self.trace_accs.shape[0]):
            for byte in range(self.byte_range[0], self.byte_range[1]):
                for leakage_acc in range(self.leakage_accs.shape[1]):
                    numerator = self.traces_processed * self.product_acc[byte, leakage_acc, trace_acc] - self.trace_accs[trace_acc, 0] * self.leakage_accs[byte, leakage_acc, 0]
                    
                    x = self.traces_processed*self.trace_accs[trace_acc, 1] - self.trace_accs[trace_acc, 0]**2
                    y = self.traces_processed*self.leakage_accs[byte, leakage_acc, 1] - self.leakage_accs[byte, leakage_acc, 0]**2
                    denominator = math.sqrt(x*y)
                    
                    results[byte, leakage_acc, trace_acc] = numerator/denominator
        return results
    
    def get_key_candidates(self, results=None):
        key_candidates = np.zeros((16), int)
        if results is None:
            results = self.calculate() 
        for byte in range(self.byte_range[0], self.byte_range[1]):
            candidates_along_bytes = np.amax(results[byte], axis=1)
            key_candidates[byte] = np.argmax(candidates_along_bytes)
        return key_candidates
