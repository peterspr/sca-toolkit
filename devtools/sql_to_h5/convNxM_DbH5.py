import sqlite3
import h5py
import numpy as np

"""
USAGE:

To convert DB file to an H5 file of profiled data only:
    - go to main and call convert_non_profiled("/path/to/db/file", batch_size=?)
    - in terminal: python3 convNxM_DbH5.py
    - if converting a file where key is 'key' and not 'k':
        - on line 40 cp/pst all of the following:
            "SELECT COUNT(DISTINCT tile_x) FROM traces WHERE key = (SELECT key FROM traces WHERE trace_id = (SELECT COUNT(trace_id) FROM traces));").fetchone()[
        - on line 47:
            "SELECT COUNT(DISTINCT tile_y) FROM traces WHERE key = (SELECT key FROM traces WHERE trace_id = (SELECT COUNT(trace_id) FROM traces));").fetchone()[
        - on line 66:
            "SELECT trace_id, tile_x, tile_y, key, ptxt, samples FROM traces WHERE key = (SELECT key FROM traces WHERE key = (SELECT COUNT(trace_id) FROM traces));")
"""


def convert_non_profiled(db_name, batch_size=10):
    h5_name = db_name.split('.')[0] + "_NON_PROFILED" + ".h5"
    batch_size = int(batch_size)

    # Connect to SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    print("DB Connected...")

    # Open HDF5 file for writing
    hdf5_file = h5py.File(h5_name, "w")
    print("File Made...")

    # Get Length of a sample
    print("Getting length of sample...")
    sample_length = len(cursor.execute("SELECT samples FROM traces;").fetchone()[0])

    # get number of distinct tile x and y in non-profiled scenario
    print("Getting number of tile x...")
    n_tile_x = cursor.execute(
        "SELECT COUNT(DISTINCT tile_x) FROM traces WHERE k = (SELECT k FROM traces WHERE trace_id = (SELECT COUNT(*) FROM traces));").fetchone()[
        0]
    print("Getting number of tile y...")
    n_tile_y = cursor.execute(
        "SELECT COUNT(DISTINCT tile_y) FROM traces WHERE k = (SELECT k FROM traces WHERE trace_id = (SELECT COUNT(*) FROM traces));").fetchone()[
        0]

    # Get number of Non-profiled traces
    print("Getting number of traces...")
    for x in range(n_tile_x):
        for y in range(n_tile_y):
            print(x, y)
            tile_length = cursor.execute(
                "SELECT COUNT(*) FROM traces WHERE tile_x = ? AND tile_y = ? AND k = (SELECT k FROM traces WHERE trace_id = (SELECT COUNT(*) FROM traces));", (x, y)).fetchone()[
                0]

            # create datasets -- do these need a 4th dimension? --> shape = (n_tile_x, n_tile_y, number_of_traces, sample_length)
            print("Initializing datasets in h5...")
            hdf5_file.create_dataset(f"traces/tile_{x}/tile_{y}/samples", data=np.empty((tile_length, sample_length), dtype=np.uint8),
                                    dtype=np.uint8)
            hdf5_file.create_dataset(f"traces/tile_{x}/tile_{y}/plaintext", data=np.empty((tile_length, 16), dtype=np.uint8), dtype=np.uint8)
            hdf5_file.create_dataset(f"traces/tile_{x}/tile_{y}/ciphertext", data=np.empty((tile_length, 16), dtype=np.uint8), dtype=np.uint8)
            hdf5_file.create_dataset(f"traces/tile_{x}/tile_{y}/key", data=np.empty((tile_length, 16), dtype=np.uint8),
                                    dtype=np.uint8)  # [[ptxt_byte_array_1], [...], ...]

            
            # get data from db:
            print("Getting data from DB...")
            cursor.execute(
                "SELECT k, ptxt, ctxt, samples FROM traces WHERE tile_x = ? AND tile_y = ? AND k = (SELECT k FROM traces WHERE trace_id = (SELECT COUNT(*) FROM traces));", (x, y))

            # Convert db data to h5
            print("Converting DB to H5...")
            for batch_offset in range(0, tile_length, batch_size):
                traces = cursor.fetchmany(batch_size)
                for index, (k, ptxt, ctxt, sample) in enumerate(traces):
                    kbuf = np.frombuffer(k, dtype=np.uint8)
                    hdf5_file[f"traces/tile_{x}/tile_{y}/key"][index + batch_offset] = kbuf

                    pbuf = np.frombuffer(ptxt, dtype=np.uint8)
                    hdf5_file[f"traces/tile_{x}/tile_{y}/plaintext"][index + batch_offset] = pbuf

                    cbuf = np.frombuffer(ctxt, dtype=np.uint8)
                    hdf5_file[f"traces/tile_{x}/tile_{y}/ciphertext"][index + batch_offset] = cbuf

                    sbuf = np.frombuffer(sample, dtype=np.uint8)
                    hdf5_file[f"traces/tile_{x}/tile_{y}/samples"][index + batch_offset] = sbuf

    hdf5_file.close()
    conn.close()


if __name__ == '__main__':
    fn = "/Volumes/passport0/15x8x20000_r1_singlerail5_sr_ise.db"
    convert_non_profiled(fn, 5)  # breaks on first batch defaults to True.
