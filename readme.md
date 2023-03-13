# Capstone project 2022-2023

A high-performance SCA library for datasets that include NxM gridded EM traces.

## General Usage

1) Download (or clone repository) to local
2) Run `pip install /path/to/file`
3) In your python file: `import notscared`

## HDF5 Hierarchy

**traces**
/ptxt[tile_x][tile_y][ptxts]

/k[tile_x][tile_y][keys]

/samples[tile_x][tiley][samples]


**ptxt** is a 3D array of integers, each element a byte of the plaintext.

**k** is a 3D array of integers, each element a byte of the key.

**samples** is a 3D array of integers, each element a reading from the sample.


### ReadH5 class

- Instantiation: ```ReadH5(filname, (x tile coordinate, y tile cooordinate), batch_size)``` -> Takes filename of the file to read, and the coordinate of the tile to read, as well as the batch size, which defaults to 10.

- ```ReadH5.next()``` -> loads the next batch of *batch_size* instatiation variable into class, overwriting the previous batch. **Need to call .next() once to load first batch or else data is None.** 

- ```ReadH5.close_file()``` -> Closes .h5 file, .next() will close it self on end of file.

- ```ReadH5.get_batch_keys()``` -> Returns a 2D numpy array of keys.

- ```ReadH5.get_batch_ptxts()``` -> Returns a 2D numpy array of ptxts.

- ```ReadH5.get_batch_samples()``` -> Returns a 2D numpy array of samples.

- ```ReadH5.plot_samples()``` -> Shows samples plotted using pyplot.

### CPA class

- Instantiation: ```CPA(byte_range, use_hamming_weight=True, precision=np.float32)``` -> Takes a tuple range of bytes. If *use_hamming_weight* is True then uses hamming weight, otherwise the hamming distance. Precision, data type of the values.

- ```CPA.push_batch(traces, plaintext)``` -> Adds the trace values based on plaintexts and updates the accumulators and the leakage cube.

- ```CPA.calculate()``` -> Performs the CPA with the pushed data. Returns the results array of CPA

- ```CPA.get_key_candidates()``` -> Returns a tuple of key candidates and their corresponding correlations.

### SNR class

- Instantiation: ```SNR(byte_positions)``` -> Takes an array of byte positions to work on.

- ```SNR.push(ptxts, traces)``` -> Takes an array of ptxts and traces and updates accumulator values with traces based on plaintext and byte positions to calculate.
- ```SNR.calculate()``` -> Performs the SNR
- ```SNR.plot()``` -> Plots SNR by byte position.
- ```SNR.snr``` -> Calls calculate if needed, and returns array of SNR's [byte_position][SNRs]
