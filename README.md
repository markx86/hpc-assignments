# HPC Assignment #1

## Goal

Optimize [ADI](https://en.wikipedia.org/wiki/Alternating-direction_implicit_method).
The original code comes from [Polybench](https://sourceforge.net/p/polybench/wiki/Home/).

## Building

The build system uses a `make`. The allowed targets are:
- `run`: build and run the executable;
- `build`: only build the executable;
- `clean`: remove all built executables.

The variables that can be passed to `make` to configure the build are:
- `VER`: specifies the version to build (default: `ref`, example values: `v1`, `v2`, ...);
- `DATASET`: specifies the dataset size (default: `mini`, acceptable values: `mini`, `small`, `standard`, `large`, `extralarge`);
- `NTHREADS`: controls the value of the `NTHREADS` define (default: number of threads of the current system);
- `DUMP_ARRAYS`: if defined to something other than an empty string, enables printing of the output array.

## Testing

To test a version use the `test.sh` script as follows:

```sh
$ ./test.sh version
```

`version` can be any value of the makefile's `VER` variable.
