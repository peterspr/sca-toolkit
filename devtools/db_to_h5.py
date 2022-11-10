import sqlite3 as sq
import numpy as np
import pandas as pd
import h5py as h5
import sys
import math


def db_to_h5(db_file, start, chunk):
    """
    :param db_file: the path to the db that you want to convert.
    :param chunk: the number of traces you would like to append to that file
    :param start: where to start in db appending traces. e.g. start == 100, start at trace_id == 100.
    :return: None. creates h5 file in same dir as db file.
    """
    # OPEN DB
    with sq.connect(db_file) as db:
        print(f"Connected to: {db_file}...")
        h5_file = (db_file.split('.')[0] + '.h5')

        print("Counting tiles...")
        cur = db.cursor()
        cur.execute(f"SELECT count(DISTINCT tile_x) FROM traces WHERE trace_id BETWEEN {start} AND {chunk + start - 1}")
        count_tile_x = cur.fetchone()[0]
        cur.execute(f"SELECT count(DISTINCT tile_y) FROM traces WHERE trace_id BETWEEN {start} AND {chunk + start - 1}")
        count_tile_y = cur.fetchone()[0]
        count_traces = 20000
        print(f"Counted {count_tile_x} distinct x tiles and {count_tile_y} distinct y tiles...")

        with h5.File(h5_file, 'x') as f:
            print(f"Created h5 file: {h5_file}...")
            traces = f.create_dataset("traces", (count_tile_x, count_tile_y, count_traces), dtype=np.int8)
            metadata = f.create_group("metadata")
            grp_trace_id = metadata.create_dataset("trace_id", (2400000,), dtype=np.int8)
            grp_tile_x = metadata.create_dataset("tile_x", (160000,), dtype=np.uint8)
            grp_tile_y = metadata.create_dataset("tile_y", (160000,), dtype=np.uint8)
            grp_k = metadata.create_dataset("k", (2400000,), dtype=np.uint8)
            grp_ptxt = metadata.create_dataset("ptxt", (2400000,), dtype=np.uint8)
            grp_ctxt = metadata.create_dataset("ctxt", (2400000,), dtype=np.uint8)

            print(f"Format created in h5 file...")

            for i in range(start, chunk + start):
                trace = pd.read_sql(f"SELECT * FROM traces WHERE trace_id = {i}", con=db)
                trace_id = int(trace['trace_id'][0])
                tile_x = int(trace['tile_x'][0])
                tile_y = int(trace['tile_y'][0])

                if i == chunk + start - 1:
                    print(f"Processed {trace_id} trace_id's -> TILE -> x:{tile_x}, y:{tile_y} -- {math.floor(100* (trace_id/( (start + chunk) - 1) ))}%")
                else:
                    print(f"Processed {trace_id} trace_id's -> TILE -> x:{tile_x}, y:{tile_y} -- {math.floor(100 * (trace_id / ((start + chunk) - 1)))}%", end='\r')

                sample = pd.DataFrame(np.frombuffer(trace['samples'][0], dtype=np.uint8), dtype=np.uint8)
                k = pd.DataFrame(np.frombuffer(trace['k'][0], dtype=np.uint8), dtype=np.uint8)
                ptxt = pd.DataFrame(np.frombuffer(trace['ptxt'][0], dtype=np.uint8), dtype=np.uint8)
                ctxt = pd.DataFrame(np.frombuffer(trace['ctxt'][0], dtype=np.uint8), dtype=np.uint8)

                np.append(traces, sample)
                np.append(grp_trace_id, trace_id)
                np.append(grp_tile_x, tile_x)
                np.append(grp_tile_y, tile_y)
                np.append(grp_k, k)
                np.append(grp_ptxt, ptxt)
                np.append(grp_ctxt, ctxt)

            f.close()


if __name__ == "__main__":
    num_args = len(sys.argv)
    if num_args < 2 or num_args > 4:
        print(f"Not correct amount arguments given.\nPlease use format: \'pythonX db_to_h5.py PATH_TO_DB START_ID CHUNK_SIZE\'")
    else:
        db_to_h5(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

