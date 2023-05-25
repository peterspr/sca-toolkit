import h5py
import hdf5plugin

original_file = h5py.File('/Volumes/passport0/latest__15x8x20000_r1_singlerail5_sr_ise_NON_PROFILED.h5', 'r')
compressed_file = h5py.File('/Volumes/passport0/LZ4_15x8x20000_r1_singlerail5_sr_ise_NON_PROFILED.h5', 'w')

# Iterate through the groups and datasets in the original file
traces_group = original_file['traces']
compressed_traces_group = compressed_file.create_group('traces')
for tile_x in traces_group:
    compressed_traces_group.create_group(tile_x)
    for tile_y in traces_group[tile_x]:
        compressed_traces_group[tile_x].create_group(tile_y)
        print(f"Tile: {tile_x}, {tile_y}")
        for data in traces_group[tile_x][tile_y]:
            compressed_traces_group[tile_x][tile_y].create_dataset(data, data=traces_group[tile_x][tile_y][data], **hdf5plugin.LZ4())

# Close the files
original_file.close()
compressed_file.close()
