# l_bnl_compress
## Lossy but not lossy compression

Implementing the lossy compressions described in [Herbert J. Bernstein, Alexei S. Soares, Kimberly Horvat,
Jean Jakoncic, 2024.  "Massive Compression for High Data Rate Macromolecular Crystallography (HDRMX): 
"Impact on Diffraction Data and Subsequent Structural Analysis," Journal of Synchrotron Radiation,
in preparation.]

### usage: l_bnl_compress.py [-h] [-1 FIRST_IMAGE] [-b BIN_RANGE] [-c COMPRESSION] [-d DATA_BLOCK_SIZE] [-H HCOMP_SCALE] [-i INFILE] [-J J2K_TARGET_COMPRESSION_RATIO] [-l COMPRESSION-LEVEL] [-m OUT_MASTER] [-N LAST_IMAGE] [-o OUT_FILE] [-s SUM_RANGE] [-v] [-V]

## Bin and sum images from a range and optionally apply JPEG-2000 or HCompress

## options:
```
  -h, --help            show this help message and exit

  -1 FIRST_IMAGE, --first_image FIRST_IMAGE
                        first selected image counting from 1

  -b BIN_RANGE, --bin BIN_RANGE
                        an integer image binning range (1 ...) 
                        to apply to each selected image

  -c COMPRESSION, --compression COMPRESSION
                        optional compression, bslz4, bszstd, or bshuf

  -d DATA_BLOCK_SIZE, --data_block_size DATA_BLOCK_SIZE
                        data block size in images for out_file

  -H HCOMP_SCALE, --Hcompress HCOMP_SCALE
                        Hcompress scale compression, immediately 
                        followed by decompression

  -i INFILE, --infile INFILE
                        the input hdf5 file to read images from

  -J J2K_TARGET_COMPRESSION_RATIO, --J2K J2K_TARGET_COMPRESSION_RATIO
                        JPEG-2000 target compression ratio, immediately 
                        followed by decompression

  -l COMPRESSION-LEVEL, --compression-level COMPRESSION-LEVEL
                        optional compression level for bszstd

  -m OUT_MASTER, --out_master OUT_MASTER
                        the output hdf5 master to which to write metadata

  -N LAST_IMAGE, --last_image LAST_IMAGE
                        last selected image counting from 1

  -o OUT_FILE, --out_file OUT_FILE
                        the output hdf5 data file out_file_?????? 
                        with a .h5 extension are files to which to 
                        write images

  -s SUM_RANGE, --sum SUM_RANGE
                        an integer image summing range (1 ...) to apply 
                        to the selected images

  -v, --verbose         provide addtional information

  -V, --version         report version and version_date
```
The program reads a NeXus hdf5 file dataset following the Dectris 
FileWriter Eiger detector conventions and produces a new file dataset 
following the same conventions.

For the current release (1.0.1) the format does allow direct processing by XDS,
but does not yet conform to the requirements for fast_dp.  For fast_dp, conversion
of the output to miniCBFs with eiger2cbf is necessary at present.

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


