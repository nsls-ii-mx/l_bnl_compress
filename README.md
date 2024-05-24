# l_bnl_compress
Lossy but not lossy compression

usage: l_bnl_compress.py [-h] [-1 FIRST_IMAGE] [-b BIN_RANGE] [-c COMPRESSION] [-d DATA_BLOCK_SIZE]
                         [-i INFILE] [-l COMPRESSION_LEVEL] [-m OUT_MASTER] [-N LAST_IMAGE] [-o OUT_FILE]
                         [-s SUM_RANGE] [-v]

Bin and sum images from a range and apply compressions

options:
  -h, --help            show this help message and exit
  -1 FIRST_IMAGE, --first_image FIRST_IMAGE
                        first selected image counting from 1
  -b BIN_RANGE, --bin BIN_RANGE
                        an integer image binning range (1 ...) to apply to each selected image
  -c COMPRESSION, --compression COMPRESSION
                        compression filter name or id number
  -d DATA_BLOCK_SIZE, --data_block_size DATA_BLOCK_SIZE
                        data block size in images for out_file
  -i INFILE, --infile INFILE
                        the input hdf5 file to read images from
  -l COMPRESSION_LEVEL, --compression_level COMPRESSION_LEVEL
                        target level for the compression filter used
  -m OUT_MASTER, --out_master OUT_MASTER
                        the output hdf5 master to which to write metadata
  -N LAST_IMAGE, --last_image LAST_IMAGE
                        last selected image counting from 1
  -o OUT_FILE, --out_file OUT_FILE
                        the output hdf5 data file out_file_?????? with a .h5 extension are files to which
                        to write images
  -s SUM_RANGE, --sum SUM_RANGE
                        an integer image summing range (1 ...) to apply to the selected images
  -v, --verbose         provide addtional information
