import sqlite3
import h5py
import numpy as np
import fire

"""
Usage: `python3 convDBtoH5.py --file=your_file_path --start=NUMBER --n=NUMBER`
    where start is the row to start conversion, and n is how many rows to convert.
"""


def simple_conv(db_name, batch_size=10):
    # make h5_name
    h5_name = db_name.split('.')[0] + ".h5"
    print(f"Converting file...")

    # Connect to SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Open HDF5 file for writing
    hdf5_file = h5py.File(h5_name, "w")

    print("Getting number of traces...")
    number_of_traces = cursor.execute("SELECT COUNT(*) FROM traces WHERE k = (SELECT k FROM traces WHERE trace_id=100000);").fetchone()[0]


    print("Getting sample length...")
    sample_length = len(cursor.execute("SELECT samples FROM traces;").fetchone()[0])

    print("Declaring dsets...")
    hdf5_file.create_dataset("traces/samples", data=np.zeros((number_of_traces, sample_length), dtype=np.uint8), dtype=np.uint8)
    hdf5_file.create_dataset("traces/ptxt", data=np.zeros((number_of_traces, 16), dtype=np.uint8), dtype=np.uint8)
    hdf5_file.create_dataset("traces/k", data=np.zeros((number_of_traces, 16), dtype=np.uint8), dtype=np.uint8)  # [[ptxt_byte_array_1], [...], ...]

    print("Getting Cursor...")
    cursor = cursor.execute(
        "SELECT k, ptxt, samples FROM traces WHERE k = (SELECT k FROM traces WHERE trace_id=100000);")  # Returns an array of tuples like: traces = [row1(key, ptxt, samples), row2...]

    for batch_offset in range(0, number_of_traces, batch_size):
        # Get data
        print(f"Getting Data Batch {batch_offset}...")
        traces = cursor.fetchmany(batch_size)
        # [(k, ptxt, sample)...]

        # print("Converting to NumPy Array...")
        # np_traces = np.stack(traces, axis=-1)  # stacks each row into numpy array -- [[keys], [plaintexts], [samples]]
        np_traces = np.array(traces)

        # Convert each sqlite column to a dataset
        # print(f"Converting columns...")
        for index, (k, ptxt, sample) in enumerate(traces):
            kbuf = np.frombuffer(k, dtype=np.uint8)
            hdf5_file["traces/k"][index + batch_offset] = kbuf
            pbuf = np.frombuffer(ptxt, dtype=np.uint8)
            hdf5_file["traces/ptxt"][index + batch_offset] = pbuf
            sbuf = np.frombuffer(sample, dtype=np.uint8)
            hdf5_file["traces/samples"][index + batch_offset] = sbuf

        # print(f"batch {batch_offset//batch_size} done")


    # Close connections
    print("Columns converted. Closing files...")
    hdf5_file.close()
    conn.close()
    print("Done!")


def convert(file, n=10):
    n = int(n)
    simple_conv(file, n)


if __name__ == "__main__":
    fire.Fire(convert)
