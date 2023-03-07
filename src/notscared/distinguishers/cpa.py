import numpy as np
from src.notscared.utils.leakage import create_leakage_table

NUM_POSSIBLE_BYTE_VALS = 256


class CPA:
    def __init__(self, byte_range, use_hamming_weight=True, precision=np.float32):
        # If false it will use hamming distance
        self.USE_HAMMING_WEIGHT = use_hamming_weight

        # Between 0 and 15
        self.BYTE_RANGE = byte_range
        self.NUM_AES_KEY_BYTES = byte_range[1] - byte_range[0]
        self.TRACE_DURATION = None
        self.PRECISION = precision
        self.traces_processed = 0

        # Accumulators used to calculate correlation coefficient
        self.product_acc = None
        self.trace_acc = None
        self.trace_squared_acc = None
        self.leakage_acc = np.zeros(shape=(NUM_POSSIBLE_BYTE_VALS, self.NUM_AES_KEY_BYTES), dtype=self.PRECISION)
        self.leakage_squared_acc = np.zeros(shape=(NUM_POSSIBLE_BYTE_VALS, self.NUM_AES_KEY_BYTES), dtype=self.PRECISION)

        self.results = None

    def push_batch(self, traces: np.ndarray, plaintext: np.ndarray):
        # Clear outdated results if more data is pushed after CPA.calculate()
        self.results = None

        BATCH_SIZE = traces.shape[0]

        # Init accumulators and trace duration on first push
        if self.TRACE_DURATION is None:
            self.TRACE_DURATION = traces.shape[1]
            self.trace_acc = np.zeros(shape=(self.TRACE_DURATION), dtype=self.PRECISION)
            self.trace_squared_acc = np.zeros(shape=(self.TRACE_DURATION), dtype=self.PRECISION)
            self.product_acc = np.zeros(shape=(NUM_POSSIBLE_BYTE_VALS, self.NUM_AES_KEY_BYTES, self.TRACE_DURATION), dtype=self.PRECISION)

        # Create leakage model for plaintexts
        # Utilize numpy broadcasting to create array of shape (BATCH_SIZE, NUM_POSSIBLE_BYTE_VALS, NUM_AES_KEY_BYTES)
        leakage_cube = np.apply_along_axis(
            create_leakage_table,
            axis=1,
            arr=plaintext[:, self.BYTE_RANGE[0]:self.BYTE_RANGE[1]],
            use_hamming_weight=self.USE_HAMMING_WEIGHT
        )
        # Populate accumulators with sum leakage and sum squared leakage
        # Utilize numpy broadcasting to compute sums over the first dimension of leakage_cube
        self.leakage_acc += np.sum(leakage_cube, axis=0, dtype=self.PRECISION)
        self.leakage_squared_acc += np.sum(np.square(leakage_cube, dtype=self.PRECISION), axis=0, dtype=self.PRECISION)

        # Populate accumulators with sum traces and sum squared traces
        self.trace_acc += np.sum(traces, axis=0, dtype=self.PRECISION)
        self.trace_squared_acc += np.sum(np.square(traces, dtype=self.PRECISION), axis=0, dtype=self.PRECISION)

        # Populate accumulator with sum of traces*leakages
        self.product_acc += np.sum((traces[:, np.newaxis, np.newaxis, :] * leakage_cube[:, :, :, np.newaxis]), axis=0, dtype=self.PRECISION)

        self.traces_processed += BATCH_SIZE
        print(f'traces_processed: {self.traces_processed}', end='\r')

    def calculate(self):
        # shape: (BYTE_VALS, KEY_BYTES, TRACE_DURATION)
        numerator = self.traces_processed * self.product_acc - self.trace_acc * self.leakage_acc[:, :, np.newaxis]
        xy = self.traces_processed * self.trace_squared_acc * self.leakage_squared_acc[:, :, np.newaxis] - self.trace_acc * self.leakage_acc[:, :, np.newaxis]
        denominator = np.sqrt(xy, dtype=self.PRECISION)
        results = numerator / denominator

        self.results = results
        return results

    def get_key_candidates(self):
        if self.results is None:
            self.calculate()

        candidates_along_bytes = np.amax(self.results, axis=2)
        key_candidates = np.full((16), -1, dtype=np.int16)
        key_candidates[self.BYTE_RANGE[0]:self.BYTE_RANGE[1]] = np.argmax(candidates_along_bytes, axis=0)
        candidate_correlation = np.amax(candidates_along_bytes, axis=0)
        return (key_candidates, candidate_correlation)
