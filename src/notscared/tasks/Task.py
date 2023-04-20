from dataclasses import dataclass
import numpy as np

@dataclass
class Options:
    pass

class Task:
    """
    Base class that tasks (such as CPA or SNR) will inherit from. 
    Used to define common interfaces that notscared will use.
    """
    def __init__(self, options: Options):
        pass

    def push(self, traces: np.ndarray, plaintexts: np.ndarray):
        """Push a batch of traces and plaintexts
 
        Pushes a batch of traces and their corresponding plaintexts to the task for processing.
        
        Args:
            traces (np.ndarray(BATCH_SIZE, TRACE_DURATION)): An array of trace arrays.
            plaintexts (np.ndarray(BATCH_SIZE, PLAINTEXT_BYTES)): An array of plaintext byte arrays.
        
        Returns:
            None
        """
        pass

    def calculate(self):
        """Calculate the final results
        
        Calculates the final results using the current state of the task.
        Does not return results -- use get_results instead.
        
        Args:
            None
        
        Returns:
            None 
        """
        pass

    def get_results(self):
        """Return the final results
 
        Return the final results as calculated by calculate().
        
        Args:
            None
        
        Returns:
            results (Unknown?): The final results as determined by the task.
        """
        pass

    def collapse(self):
        pass
