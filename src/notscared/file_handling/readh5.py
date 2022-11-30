import h5py as h5

def readh5(generator):
    def create_gen(*args):
        g = generator(*args)
        return g
    return create_gen    



@readh5
def trace_gen(filename, tile_x, tile_y):
    """
    @param filename: string that is an already existing file.h5
    @param tile_x: x coordinate of tile to read
    @param tile_y: y coordinate of tile to read
    @return: generator object
    """
    with h5.File(filename, "r") as f:
        traces = f["traces"]
        for trace in traces[tile_x][tile_y]:
            yield trace
