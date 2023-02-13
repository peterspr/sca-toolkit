# Capstone project 2022-2023

A high-performance SCA library for datasets that include NxM gridded EM traces.

## General Usage

In current release alpha (Version 0.0.1):

### CLI

Uses Google's Fire Python Package.

1) Download (or clone repository) to local
2) Change Directory with `cd local_dir_name/src/notscared/`


#### Perform CPA

Run `python3 main.py run_cpa --file=filename --low_byte=top_byte_location --end_byte=end_byte_location --traces=n_traces --hamming_weight=1 batch_size=number_of_traces_in_a_batch`

- **file** is path to your HDF5 file
- **bytes** is the index of bytes to look at
- **traces** is the number of traces in a sample
- **hamming_weight** is 1 if you want to use hamming weights, or 0 for hamming distance

### As a Python Package

1) Download (or clone repository) to local
2) Run `pip install /path/to/file`
3) In your python file: `import notscared`

## HDF5 Hierarchy

**traces**
|___ptxt
|___k
|___samples

**ptxt** is a 2D array of integers, each element a byte of the plaintext.
**k** is a 2D array of integers, each element a byte of the key.
**samples** is a 2D array of integers, each element a reading from the sample.