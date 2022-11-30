# File Handling
This will explain usage of the file handling module in src/notscared.

## readh5

readh5.py contains a generator fn.  Needs to be used like this:

```python

import readh5

gen = readh5.read_gen("filename.h5", x=0, y=0) # initialize generator for tile (0, 0)

while ((value = next(gen)) != StopIteration):
    print(value)
    # do something with value ...
```
