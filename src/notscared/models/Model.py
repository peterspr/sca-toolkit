import numpy as np

class Model:
    """
    Base class that models (such as Hamming Weight or Hamming Distance) will inherit from. 
    Used to define common interfaces that tasks will use.
    """
    def __init__(self):
        pass

    def create_leakage_table(self, plaintext_bytes: np.ndarray):
        """Create a leakage table from given plaintexts
 
        For each byte of plaintext, create an array of values that represent the modelled leakage of processor. These arrays are
        of length 256, 1 for each value in the range of all possible values of a single byte of an AES key.
        
        Args:
            traces (np.ndarray(BATCH_SIZE, TRACE_DURATION)): An array of trace arrays.
            plaintexts (np.ndarray(BATCH_SIZE, PLAINTEXT_BYTES)): An array of plaintext byte arrays.
        
        Returns:
            leakage_table (np.ndarray(NUM_POSSIBLE_BYTE_VALS, NUM_AES_KEY_BYTES)): A 2d array of leakage values
        """

    def create_perfect_correlation(self, plaintext, key):
        pass
