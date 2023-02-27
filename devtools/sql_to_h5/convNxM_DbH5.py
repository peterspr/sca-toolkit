import sqlite3
import h5py
import numpy as np

"""
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

    # Get number of Non-profiled traces
    print("Getting number of traces...")
    number_of_traces = cursor.execute(
        "SELECT COUNT(*) FROM traces WHERE k = (SELECT k FROM traces WHERE trace_id = (SELECT COUNT(trace_id) FROM traces));").fetchone()[
        0]

    # Get Length of a sample
    print("Getting length of sample...")
    sample_length = len(cursor.execute("SELECT samples FROM traces;").fetchone()[0])

    # get number of distinct tile x and y in non-profiled scenario
    print("Getting number of tile x...")
    n_tile_x = cursor.execute(
        "SELECT COUNT(DISTINCT tile_x) FROM traces WHERE k = (SELECT k FROM traces WHERE trace_id = (SELECT COUNT(trace_id) FROM traces));").fetchone()[
        0]
    print("Getting number of tile y...")
    n_tile_y = cursor.execute(
        "SELECT COUNT(DISTINCT tile_y) FROM traces WHERE k = (SELECT k FROM traces WHERE trace_id = (SELECT COUNT(trace_id) FROM traces));").fetchone()[
        0]

    # create datasets -- do these need a 4th dimension? --> shape = (n_tile_x, n_tile_y, number_of_traces, sample_length)
    print("Initializing datasets in h5...")
    hdf5_file.create_dataset("traces/samples", data=np.empty((n_tile_x, n_tile_y, sample_length), dtype=np.uint8),
                             dtype=np.uint8)
    hdf5_file.create_dataset("traces/ptxt", data=np.empty((n_tile_x, n_tile_y, 16), dtype=np.uint8), dtype=np.uint8)
    hdf5_file.create_dataset("traces/k", data=np.empty((n_tile_x, n_tile_y, 16), dtype=np.uint8),
                             dtype=np.uint8)  # [[ptxt_byte_array_1], [...], ...]

    # get data from db:
    print("Getting data from DB...")
    cursor.execute(
        "SELECT trace_id, tile_x, tile_y, k, ptxt, samples FROM traces WHERE k = (SELECT k FROM traces WHERE k = (SELECT COUNT(trace_id) FROM traces));")

    # Convert db data to h5
    print("Converting DB to H5...")
    for batch_offset in range(0, number_of_traces, batch_size):
        traces = cursor.fetchmany(batch_size)

        for index, (trace_id, tile_x, tile_y, k, ptxt, sample) in enumerate(traces):
            print("Processing trace_id: ", trace_id)
            kbuf = np.frombuffer(k, dtype=np.uint8)
            hdf5_file["traces/k"][tile_x][tile_y][index + batch_offset] = kbuf

            pbuf = np.frombuffer(ptxt, dtype=np.uint8)
            hdf5_file["traces/ptxt"][tile_x][tile_y][index + batch_offset] = pbuf

            sbuf = np.frombuffer(sample, dtype=np.uint8)
            hdf5_file["traces/samples"][tile_x][tile_y][index + batch_offset] = sbuf

    hdf5_file.close()
    conn.close()


if __name__ == '__main__':
    fn = "/Volumes/passport0/15x8x20000_r1_singlerail5_sr_ise.db"
    convert_non_profiled(fn, 5)  # breaks on first batch defaults to True.
