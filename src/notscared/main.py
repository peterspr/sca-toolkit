import fire
from file_handling.readh5 import ReadH5
from distinguishers.cpa import CPA

def run_cpa(file, low_byte, high_byte, traces, hamming_weight, num_batches=None):
    print(f"Getting traces from file: {file}...")

    num_batches = int(num_batches) if num_batches is not None else None
    traces = int(traces)

    if hamming_weight == "0":
        hamming_weight = False
    else:
        hamming_weight = True

    read = ReadH5(file)
    cpa_instance = CPA((int(low_byte), int(high_byte)), int(traces), hamming_weight)

    batch_num = 0
    while read.next():
        print(f"Pushing batch {batch_num}", end='\r')
        cpa_instance.push_batch(read.get_batch_samples(), read.get_batch_ptxts())
        batch_num += 1
        if  batch_num == num_batches:
            break
    
    print("KEY CANDIDATES:\n", cpa_instance.get_key_candidates())    


if __name__ == "__main__":
    fire.Fire({"run_cpa": run_cpa, "read": ReadH5}) 
