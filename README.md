# l_bnl_compress
## Lossy but not lossy compression

(C) Copyright 2 May 2025 Herbert J. Bernstein
Portions suggested by claude.ai from Anthropic
You may redistribute l_bnl_compress.py under GPL2 or LGPL2 

Implementing the lossy compressions described in [Herbert J. Bernstein, Alexei S. Soares, Kimberly Horvat,
Jean Jakoncic, 2025.  "Massive Compression for High Data Rate Macromolecular Crystallography (HDRMX): 
"Impact on Diffraction Data and Subsequent Structural Analysis," J. Synchrotron Radiation,
 Mar 1;32(2)]

## usage: l_bnl_compress.py 
    [-h] [-1 [FIRST_IMAGE]] [-b [BIN_RANGE]] [-c [COMPRESSION]] [-d [DATA_BLOCK_SIZE]]
    [-H HCOMP_SCALE] [-i INFILE] [-J J2K_TARGET_COMPRESSION_RATIO]
    [-K J2K_ALT_TARGET_COMPRESSION_RATIO] [-l COMPRESSION_LEVEL] [-m OUT_MASTER]
    [-N LAST_IMAGE] [-o OUT_FILE] [-p THREADS] [-q OUT_SQUASH] [-s [SUM_RANGE]]
    [-t [THREAD]] [-u [UINT]] [-v] [-V]

## Bin and sum images from a range and optionally apply JPEG-2000 or HCompress

## options:
```
  -h, --help            show this help message and exit

  -1 [FIRST_IMAGE], --first_image [FIRST_IMAGE]
                        first selected image counting from 1, defaults to 1

  -b [BIN_RANGE], --bin [BIN_RANGE]
                        an integer image binning range (1 ...) to apply to each selected 
                        image, defaults to 1

  -c [COMPRESSION], --compression [COMPRESSION]
                        optional compression, bslz4, bszstd, bshuf, or zstd, defaults to zstd

  -d [DATA_BLOCK_SIZE], --data_block_size [DATA_BLOCK_SIZE]
                        data block size in images for out_file, defaults to 100

  -H HCOMP_SCALE, --Hcompress HCOMP_SCALE
                        Hcompress scale compression, immediately followed by decompression

  -i INFILE, --infile INFILE
                        the input hdf5 file to read images from

  -J J2K_TARGET_COMPRESSION_RATIO, --J2K J2K_TARGET_COMPRESSION_RATIO
                        JPEG-2000 target compression ratio, immediately followed by decompression

  -K J2K_ALT_TARGET_COMPRESSION_RATIO, --J2K2 J2K_ALT_TARGET_COMPRESSION_RATIO
                        JPEG-2000 alternate target compression ratio, immediately followed 
                        by decompression

  -l COMPRESSION_LEVEL, --compression_level COMPRESSION_LEVEL
                        optional compression level for bszstd or zstd

  -m OUT_MASTER, --out_master OUT_MASTER
                        the output hdf5 master to which to write metadata

  -N LAST_IMAGE, --last_image LAST_IMAGE
                        last selected image counting from 1, defaults to number of 
                        images collected

  -o OUT_FILE, --out_file OUT_FILE
                        the output hdf5 data file out_file_?????? with an .h5 extension 
                        are files to which to write images

  -p THREADS, -- parallel THREADS
                        the number of parallel threads with extra thread 0 used by itself 
                        to generate the new master file first

  -q OUT_SQUASH, --out_squash OUT_SQUASH
                        the output hdf5 data file out_squash_?????? with an .h5 
                        extension are optional files to which to write raw j2k or hcomp images

  -s [SUM_RANGE], --sum [SUM_RANGE]
                        an integer image summing range (1 ...) to apply to the selected 
                        images, defaults to 1

  -t [THREAD], --thread [THREAD]
                        the thread number for the action of the current invocation of 
                        l_bnl_compress, 0 to just make a new master file, otherwise 
                        between 1 and the number of datablocks/threads

  -u [UINT], --uint [UINT]
                        clip the output above 0 and limit to 2 byte or 4 byte integers

  -v, --verbose         provide addtional information

  -V, --version         report version and version_date

```
The program reads a NeXus hdf5 file dataset following the Dectris 
FileWriter Eiger detector conventions and produces a new file dataset 
following the same conventions.

For the current release (1.1.3) the format does allow direct processing by XDS,
but does not yet conform to the requirements for fast_dp.  For fast_dp, conversion
of the output to miniCBFs is necessary at present.

## Installation Prerequisites

This is a python program that needs Python 3 running in a Linux-like environment. 
It was created on a Debian 12 system and tested on a RHEL8 environment.  It is 
not likely to work with a python version prior to 3.9.  A suitable python with 
many modules upon which l_bnl_compress.py depends can be provided by use 
of cctbx.python from the DIALS package.

The modules upon which this program depends are:

      sys
      os
      argparse
      numpy
      skimage
      h5py
      tifffile
      astropy
      glymur
      hdf5plugin
      tempfile
      numcodecs
      BytesIO
      PIL
      warnings
      subprocess
      threading
      time
      queue
      string

This version supports coarse-grained process-level parallelism in writing
the -H, -J, or -K compressions.  In the near future
we intend to upgrade to parallelism using concurrent.futures, which
will also requre an upgrade to parallel hdf5 and parallel h5py.
In order to achieve reasonable timing without major use of parallelism
as many files as possible should be in memory.  Use of /dev/shm is
highly recommended.
