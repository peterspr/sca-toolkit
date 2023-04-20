from dataclasses import dataclass
import numpy as np
from .Task import Task, Options
from ..models.Model import Model
from ..models.HammingWeight import HammingWeight

NUM_POSSIBLE_BYTE_VALS = 256

@dataclass
class CPAOptions(Options):
    """Class for specifying the options for a CPA task"""
    byte_range: tuple = (0, 2) # Defaulting to first 2 bytes as our datasets usually dont contain 3-16
    leakage_model: Model = HammingWeight()
    precision: np.dtype = np.float32

class CPA(Task):
    """
    Correlation Power Analysis distinguisher.
    
    Calculates the correlation between the measured leakage (traces) and modelled leakage for each point_in_time (sample index)
    using the Hamming Weight or Hamming Distance power models.
    """
    def __init__(self, options: CPAOptions = CPAOptions()):
        """
        byte_range: Range of key bytes to calculate. i.e. (0, 1) will only do the math for the key byte at index 0. (0, 16) for all 16 bytes.
        use_hamming_weight: Whether to use the Hamming Weight leakage model (if true) or Hamming Distance leakage model (if false)
        precision: A numpy dtype to use for the accumulators holding intermediate values for calculating the correlation coefficients.
        """
        self.LEAKAGE_MODEL = options.leakage_model

        # Between 0 and 15
        self.BYTE_RANGE = options.byte_range
        self.NUM_AES_KEY_BYTES = options.byte_range[1] - options.byte_range[0]
        self.TRACE_DURATION = None
        self.PRECISION = options.precision
        self.traces_processed = 0

        # Accumulators used to calculate correlation coefficient
        self.product_acc = None
        self.trace_acc = None
        self.trace_squared_acc = None
        self.leakage_acc = np.zeros(shape=(NUM_POSSIBLE_BYTE_VALS, self.NUM_AES_KEY_BYTES), dtype=self.PRECISION)
        self.leakage_squared_acc = np.zeros(shape=(NUM_POSSIBLE_BYTE_VALS, self.NUM_AES_KEY_BYTES), dtype=self.PRECISION)

        self.results = None

    def push(self, traces: np.ndarray, plaintexts: np.ndarray):
        """
        Push a 2d array of traces and plaintexts for processing.
        A single plaintext needs to be broken up into a numpy array of its 16 constituent bytes.
        traces.shape = (BATCH_SIZE, TRACE_DURATION)
        plaintexts.shape = (BATCH_SIZE, PLAINTEXT_BYTES)
        """
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
            self.LEAKAGE_MODEL.create_leakage_table,
            axis=1,
            arr=plaintexts[:, self.BYTE_RANGE[0]:self.BYTE_RANGE[1]]
        )
        # Populate accumulators with sum leakage and sum squared leakage
        # Utilize numpy broadcasting to compute sums over the first dimension of leakage_cube
        self.leakage_acc += np.sum(leakage_cube, axis=0, dtype=self.PRECISION)
        self.leakage_squared_acc += np.sum(np.square(leakage_cube, dtype=self.PRECISION), axis=0, dtype=self.PRECISION)

        # Populate accumulators with sum traces and sum squared traces
        self.trace_acc += np.sum(traces, axis=0, dtype=self.PRECISION)
        self.trace_squared_acc += np.sum(np.square(traces, dtype=self.PRECISION), axis=0, dtype=self.PRECISION)

        # Populate accumulator with sum of traces*leakages
        leakage_cube_t = leakage_cube.transpose((1, 2, 0))
        self.product_acc += np.dot(leakage_cube_t, traces)

        self.traces_processed += BATCH_SIZE
        print(f'traces_processed: {self.traces_processed}', end='\r')

    def calculate(self):
        """
        Calculate the Pearson correlation coefficient for the data.
        Each value in the 3d results array is a correlation coefficient.
        
        If results[129, 3, 12000] == 0.63, that means the measured traces have a 0.63 
        correlation at the point_in_time column 12000, with a 3rd key byte value of 129.
        
        results.shape = (NUM_POSSIBLE_BYTE_VALS, NUM_AES_KEY_BYTES, TRACE_DURATION)
        """
        # shape: (BYTE_VALS, KEY_BYTES, TRACE_DURATION)
        numerator = self.traces_processed * self.product_acc - self.trace_acc * self.leakage_acc[:, :, np.newaxis]
        xy = ((self.traces_processed * self.trace_squared_acc) - np.square(self.trace_acc, dtype=self.PRECISION)) * ((self.traces_processed * self.leakage_squared_acc) - np.square(self.leakage_acc, dtype=self.PRECISION))[:, :, np.newaxis]
        denominator = np.sqrt(xy, dtype=self.PRECISION)
        results = numerator / denominator

        self.results = results
        return results

    def get_results(self):
        """Return an array of the 16 key bytes with the highest correlation by byte index"""
        if self.results is None:
            self.calculate()

        candidates_along_bytes = np.amax(self.results, axis=2)
        key_candidates = np.full((16), -1, dtype=np.int16)
        key_candidates[self.BYTE_RANGE[0]:self.BYTE_RANGE[1]] = np.argmax(candidates_along_bytes, axis=0)
        candidate_correlation = np.amax(candidates_along_bytes, axis=0)
        return (key_candidates, candidate_correlation)
