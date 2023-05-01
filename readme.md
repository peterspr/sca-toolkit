# Capstone project 2022-2023

A high-performance SCA library for datasets that include NxM gridded EM traces.

## General Usage

1) Download (or clone repository) to local
2) Run `pip install /path/to/file`
3) In your python file: `import notscared`

See the demo jupyter notebook for an example.

## File Format: HDF5 Hierarchy

**traces**/

/plaintext/tile_x/tile_y/[ptxts]

/key/tile_x/tile_y/[keys]

/samples/tile_x/tile_y/[samples]

/ciphertext/tile_x/tile_y/[ctxts]

**plaintext** is a 2D array of integers, each element a byte of the plaintext.

**key** is a 2D array of integers, each element a byte of the key.

**samples** is a 2D array of integers, each element a reading from the sample.

**ciphertext** is a 2D array of integers, each element a byte of the ciphertext.

### NotScared class

- The NotScared class takes a filename, task, and some task-specific options. It handles the running of tasks for each tile on the NxM gridded dataset. 

- Instantiation: ```NotScared(filename, task, task_options, tiles)``` -> Takes filename of the HDF5 file to read, the task to run (the class itself, not an instance), an instance of the options object specific to the task, as well as the XY dimensions of the dataset grid.

- ```NotScared.run()``` -> Runs the task on each tile, storing the results in results[x][y]. Uses pooling.

- ```NotScared.run_process_no_pool()``` -> Same as NotScared.run, but using Process instead of Pool.

- ```NotScared.tasks[x][y]``` -> Exposes the task instance for the tile (x, y) of the input dataset.

- ```NotScared.results[x][y]``` -> Alias for NotScared.tasks[x][y].results.


### Task class

- The base Task class is used to define common interfaces for use by the NotScared class. It represents a task type to be run and offers several methods to interact with the data, as relevant to the task. Users do not instanciate tasks on their own, rather NotScared takes a reference to the Task's class and creates a task for each tile (x, y) of the input dataset.


### CPA class

- The CPA class is a task that can perform correlation power analysis on the traces in the dataset.

- ```CPAOptions(byte_range, leakage_model, precision)``` -> Takes a tuple containing a range of AES key bytes to attempt to recover [a, b), a leakage model class instance, and a numpy dtype to use for internal precision.

- ```CPA.push(traces, plaintext)``` -> Uses a batch of traces and corresponding plaintexts to update an internal running state.

- ```CPA.calculate()``` -> Calculates the final results from its current internal state and stores it in CPA.results

- ```CPA.get_results()``` -> Returns a tuple of key candidates and their corresponding correlation values.

### SNR class

- The SNR class is a task that can calculate the signal-to-noise ratio of the traces in the dataset.

- ```SNROptions(byte_positions)``` -> Takes a list of byte positions [a, b, c, ...] to perform the signal-to-noise calculation for.

- ```SNR.push(ptxts, traces)``` -> Takes an array of ptxts and traces and updates accumulator values with traces based on plaintext and byte positions to calculate.
- ```SNR.calculate()``` -> Performs the SNR
- ```SNR.plot()``` -> Plots SNR by byte position.
- ```SNR.snr``` -> Calls calculate if needed, and returns array of SNR's [byte_position][SNRs]

### Model Class

- The base Model class is used to define common interfaces for use by various Tasks. It represents a leakage model for the processor and offers the create_leakage_table method for tasks to create a leakage table from a batch of traces.


### ReadH5 class

- Instantiation: ```ReadH5(filname, (x tile coordinate, y tile cooordinate), batch_size)``` -> Takes filename of the file to read, and the coordinate of the tile to read, as well as the batch size, which defaults to 10.

- ```ReadH5.next()``` -> loads the next batch of *batch_size* instatiation variable into class, overwriting the previous batch. **Need to call .next() once to load first batch or else data is None.** 

- ```ReadH5.close_file()``` -> Closes .h5 file, .next() will close it self on end of file.

- ```ReadH5.get_batch_keys()``` -> Returns a 2D numpy array of keys.

- ```ReadH5.get_batch_ptxts()``` -> Returns a 2D numpy array of ptxts.

- ```ReadH5.get_batch_samples()``` -> Returns a 2D numpy array of samples.

- ```ReadH5.plot_samples()``` -> Shows samples plotted using pyplot.

