from src.notscared.file_handling.readh5 import ReadH5
from src.notscared.distinguishers.cpa import CPA


def run_cpa(file='C:/Users/dryic/Documents/notscared/data/1x1x100000_r1_singlerail5_sr_ise_unprofiled.h5', low_byte='0', high_byte='2', hamming_weight='1', num_batches='-1'):
    print(f"Getting traces from file: {file}...")

    num_batches = int(num_batches) if num_batches is not None else None

    if hamming_weight == "0":
        hamming_weight = False
    else:
        hamming_weight = True

    read = ReadH5(file)
    cpa_instance = CPA((int(low_byte), int(high_byte)), hamming_weight)

    batch_num = 0
    while read.next():
        cpa_instance.push_batch(read.get_batch_samples(), read.get_batch_ptxts())
        batch_num += 1
        if batch_num == num_batches:
            break

    print("KEY CANDIDATES:\n", cpa_instance.get_key_candidates())


if __name__ == "__main__":
    run_cpa()
