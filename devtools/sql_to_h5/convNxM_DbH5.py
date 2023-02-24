import sqlite3
import h5py
import numpy as np
import fire

def convert_non_profiled(db_name, batch_size=10, dev_break_on_first_batch=True):
    h5_name = h5_name = db_name.split('.')[0] + "NON_PROFILED" + ".h5"

    # Connect to SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Open HDF5 file for writing
    hdf5_file = h5py.File(h5_name, "w")

    # Get number of Non-profiled traces
    number_of_traces = cursor.execute("SELECT COUNT(*) FROM traces WHERE k = (SELECT k FROM traces WHERE trace_id = (SELECT COUNT(trace_id) FROM traces));").fetchone()[0]

    # Get Length of a sample
    sample_length = len(cursor.execute("SELECT samples FROM traces;").fetchone()[0])

    # get number of distinct tile x and y in non-profiled scenario
    n_tile_x = cursor.execute("SELECT COUNT(DISTINCT tile_x) FROM tracesWHERE k = (SELECT k FROM traces WHERE trace_id = (SELECT COUNT(trace_id) FROM traces));").fetchone()[0]
    n_tile_y = cursor.execute("SELECT COUNT(DISTINCT tile_y) FROM tracesWHERE k = (SELECT k FROM traces WHERE trace_id = (SELECT COUNT(trace_id) FROM traces));").fetchone()[0]

    # create datasets -- do these need a 4th dimension? --> shape = (n_tile_x, n_tile_y, number_of_traces, sample_length)
    hdf5_file.create_dataset("traces/samples", data=np.empty((n_tile_x, n_tile_y, sample_length), dtype=np.uint8), dtype=np.uint8)
    hdf5_file.create_dataset("traces/ptxt", data=np.empty((n_tile_x, n_tile_y, 16), dtype=np.uint8), dtype=np.uint8)
    hdf5_file.create_dataset("traces/k", data=np.empty((n_tile_x, n_tile_y, 16), dtype=np.uint8), dtype=np.uint8)  # [[ptxt_byte_array_1], [...], ...]

    # get data from db:
    cursor.execute("SELECT trace_id, tile_x, tile_y, k, ptxt, samples FROM traces WHERE k = (SELECT k FROM traces WHERE k = (SELECT COUNT(trace_id) FROM traces));")

    # Convert db data to h5
    for batch_offset in range(0, number_of_traces, batch_size):
        traces = cursor.fetchmany(batch_size)

        for index, (trace_id, tile_x, tile_y, k, ptxt, sample) in enumerate(traces):
            print(f"Processing trace_id {trace_id}")

            kbuf = kbuf = np.frombuffer(k, dtype=np.uint8)
            hdf5_file["traces/k"][tile_x][tile_y][index + batch_offset] = kbuf

            pbuf = np.frombuffer(ptxt, dtype=np.uint8)
            hdf5_file["traces/ptxt"][tile_x][tile_y][index + batch_offset] = pbuf

            sbuf = np.frombuffer(sample, dtype=np.uint8)
            hdf5_file["traces/samples"][tile_x][tile_y][index + batch_offset] = sbuf
        
        if dev_break_on_first_batch:
            break
    
    hdf5_file.close()
    conn.close()

if __name__ == '__main__':
    fire.Fire(convert_non_profiled)

    