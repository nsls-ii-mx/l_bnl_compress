'''l_bnl_compress.py, lossy, but not lossy, compresss,
       a script to apply lossy compression to HDF5 MX image files.
 
usage: l_bnl_compress.py [-h] [-1 FIRST_IMAGE] [-b BIN_RANGE] [-c COMPRESSION] [-d DATA_BLOCK_SIZE] \
                         [-H HCOMP_SCALE] [-i INFILE] [-J J2K_TARGET_COMPRESSION_RATIO] \
                         [-l COMPRESSION-LEVEL] [-m OUT_MASTER] [-N LAST_IMAGE] [-o OUT_FILE] \
                         [-s SUM_RANGE] [-v]

Bin and sum images from a range

options:
  -h, --help            show this help message and exit
  -1 FIRST_IMAGE, --first_image FIRST_IMAGE
                        first selected image counting from 1
  -b BIN_RANGE, --bin BIN_RANGE
                        an integer image binning range (1 ...) to apply to each selected image
  -c COMPRESSION, --compression COMPRESSION
                        optional compression, bslz4, bszstd, or bshuf
  -d DATA_BLOCK_SIZE, --data_block_size DATA_BLOCK_SIZE
                        data block size in images for out_file
  -H HCOMP_SCALE, --Hcompress HCOMP_SCALE
                        Hcompress scale compression, immediately followed by decompression
  -i INFILE, --infile INFILE
                        the input hdf5 file to read images from
  -J J2K_TARGET_COMPRESSION_RATIO, --J2K J2K_TARGET_COMPRESSION_RATIO
                        JPEG-2000 target compression ratio, immediately followed by decompression
  -l COMPRESSION-LEVEL, --compression-level COMPRESSION-LEVEL
                        optional compression level for bszstd
  -m OUT_MASTER, --out_master OUT_MASTER
                        the output hdf5 master to which to write metadata
  -N LAST_IMAGE, --last_image LAST_IMAGE
                        last selected image counting from 1
  -o OUT_FILE, --out_file OUT_FILE
                        the output hdf5 data file out_file_?????? with a .h5 extension are files to which to write images
  -s SUM_RANGE, --sum SUM_RANGE
                        an integer image summing range (1 ...) to apply to the selected images
  -v, --verbose         provide addtional information

  -V, --version         report the version and build_date

'''

import sys
import os
import argparse
import numpy as np
import skimage as ski
import h5py
import tifffile
from astropy.io import fits
from io import BytesIO
import glymur
import hdf5plugin
import tempfile
import numcodecs
from astropy.io.fits.hdu.compressed._codecs import HCompress1

version = "1.0.1"
version_date = "23Jul24"

def ntstr(xstr):
    return str(xstr)+('0x00')

def ntstrdt(str):
    return h5py.string_dtype(encoding='utf-8',length=len(str)+1)

def conv_pixel_mask(old_mask,bin_range):
    ''' conv_pixel_mask -- returns a new pixel_mask
    array, adjusting for binning
    '''

    if bin_range < 2 :
        return np.asarray(old_mask,dtype='u4')
    old_shape=old_mask.shape
    if len(old_shape) != 2:
        print('l_bnl_compress.py: invalid mask shape for 2D binning')
        return None
    sy=old_shape[0]
    nsy=int(sy+bin_range-1)//bin_range
    ymargin=0
    if nsy*bin_range > sy:
        ymargin=bin_range-(sy%bin_range)
    sx=old_shape[1]
    nsx=int(sx+bin_range-1)//bin_range
    xmargin=0
    if nsx*bin_range > sx:
        xmargin=bin_range-(sx%bin_range)
    if ((xmargin > 0) or (ymargin > 0)):
        old_mask_rev=np.pad(np.asarray(old_mask,dtype='u4'),((0,ymargin),(0,xmargin)),'constant',constant_values=((0,0),(0,0)))
    else:
        old_mask_rev=np.asaary(old_mask,dtype='u4')
    new_mask=np.zeros((nsy,nsx),dtype='u4')
    for iy in range(0,sy,bin_range):
        for ix in range(0,sx,bin_range):
            for iyy in range(0,bin_range):
                for ixx in range(0,bin_range):
                    if ix+ixx < sx and iy+iyy < sy:
                        if old_mask_rev[iy+iyy,ix+ixx] != 0:
                            new_mask[iy//bin_range,ix//bin_range] = new_mask[iy//bin_range,ix//bin_range]| old_mask_rev[iy+iyy,ix+ixx]
    return new_mask


def conv_image_to_block_offset(img,npb):
    ''' conv_image_to_block_offset(img,npb)

    convert an image number, img, counting from 1,
    given the number of images per block, npb,
    into a 2-tuple consisting of the block
    number, counting from zero, and an image
    nuber within that block counting from zero.
    '''

    nblk=(img-1)//npb
    return (nblk,(int(img-1)%int(npb)))

def conv_image_shqpe(old_shape,bin_range):
    if len(old_shape) != 2:
        print('l_bnl_compress.py: invalid image shape for 2D binning')
        return None
    sy=old_shape[0]
    nsy=int(sy+bin_range-1)//bin_range
    ymargin=0
    if nsy*bin_range > sy:
        ymargin=bin_range-(sy%bin_range)
    sx=old_shape[1]
    nsx=int(sx+bin_range-1)//bin_range
    xmargin=0
    if nsx*bin_range > sx:
        xmargin=bin_range-(sx%bin_range)
    return ((sy+ymargin)//bin_range,(sx+xmargin)//bin_range)

def bin(old_image,bin_range,satval):
    ''' bin(old_image,bin_range,satval)

    convert an image in old_image to a returned u2 numpy array by binning
    the pixels in old_image in by summing bin_range by bin_range rectanglar 
    blocks, clipping values between 0 and satval.  If bin_range does not divide
    the original dimensions exactly, the old_image is padded with zeros.
    '''

    if bin_range < 2:
        return (np.asarray(old_image,dtype='i2')).clip(0,satval)
    s=old_image.shape
    if len(s) != 2:
        print('l_bnl_compress.py: invalid image shape for 2D binning')
        return None
    sy=s[0]
    nsy=int(sy+bin_range-1)//bin_range
    ymargin=0
    if nsy*bin_range > sy:
        ymargin=bin_range-(sy%bin_range)
    sx=s[1]
    nsx=int(sx+bin_range-1)//bin_range
    xmargin=0
    if nsx*bin_range > sx:
        xmargin=bin_range-(sx%bin_range)
    if ((xmargin > 0) or (ymargin > 0)):
        new_image=np.clip(np.pad(np.asarray(old_image,dtype='i2'),((0,ymargin),(0,xmargin)),'constant',constant_values=((0,0),(0,0))),0,satval)
    else:
        new_image=np.clip(np.asarray(old_image,dtype='i2'),0,satval)
    new_image=(np.asarray(new_image,dtype='i2')).clip(0,satval)
    new_image=np.round(ski.measure.block_reduce(new_image,(bin_range,bin_range),np.sum))
    new_image=np.asarray(np.clip(new_image,0,satval),dtype='i2')
    return new_image


parser = argparse.ArgumentParser(description='Bin and sum images from a range')
parser.add_argument('-1','--first_image', dest='first_image', type=int,
   help= 'first selected image counting from 1')
parser.add_argument('-b','--bin', dest='bin_range', type=int,
   help= 'an integer image binning range (1 ...) to apply to each selected image') 
parser.add_argument('-c','--compression', dest='compression',
   help= 'optional compression, bslz4, bszstd, or bshuf')
parser.add_argument('-d','--data_block_size', dest='data_block_size', type=int,
   help= 'data block size in images for out_file')
parser.add_argument('-H','--Hcompress', dest='hcomp_scale', type=int,
   help= 'Hcompress scale compression, immediately followed by decompression')
parser.add_argument('-i','--infile',dest='infile',
   help= 'the input hdf5 file to read images from')
parser.add_argument('-J','--J2K', dest='j2k_target_compression_ratio', type=int,
   help= 'JPEG-2000 target compression ratio, immediately followed by decompression')
parser.add_argument('-l','--compression-level', dest='compression-level', type=int,
   help= 'optional compression level for bszstd')
parser.add_argument('-m','--out_master',dest='out_master',
   help= 'the output hdf5 master to which to write metadata')
parser.add_argument('-N','--last_image', dest='last_image', type=int,
   help= 'last selected image counting from 1')
parser.add_argument('-o','--out_file',dest='out_file',default='out_data',
   help= 'the output hdf5 data file out_file_?????? with a .h5 extension are files to which to write images')
parser.add_argument('-s','--sum', dest='sum_range', type=int,
   help= 'an integer image summing range (1 ...) to apply to the selected images')
parser.add_argument('-v','--verbose',dest='verbose',action='store_true', 
   help= 'provide addtional information')
parser.add_argument('-V','--version',dest='report_version',action='store_true',
   help= 'report version and version_date')
args = vars(parser.parse_args())

#h5py._hl.filters._COMP_FILTERS['blosc']    =32001
#h5py._hl.filters._COMP_FILTERS['lz4']      =32004
#h5py._hl.filters._COMP_FILTERS['bshuf']    =32008
#h5py._hl.filters._COMP_FILTERS['zfp']      =32013
#h5py._hl.filters._COMP_FILTERS['zstd']     =32015
#h5py._hl.filters._COMP_FILTERS['sz']       =32017
#h5py._hl.filters._COMP_FILTERS['fcidecomp']=32018
#h5py._hl.filters._COMP_FILTERS['jpeg']     =32019
#h5py._hl.filters._COMP_FILTERS['sz3']      =32024
#h5py._hl.filters._COMP_FILTERS['blosc2']   =32026
#h5py._hl.filters._COMP_FILTERS['j2k']      =32029
#h5py._hl.filters._COMP_FILTERS['hcomp']    =32030

if args['report_version'] == True:
    print('l_bnl_compress-'+version+'-'+version_date)

if args['verbose'] == True:
    print(args)
    #print(h5py._hl.filters._COMP_FILTERS)

try:
    fin = h5py.File(args['infile'], 'r')
except:
    print('l_bnl_compress.py: infile not specified')
    sys.exit(-1)

try:
    detector=fin['entry']['instrument']['detector']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector: ', detector)
except:
    print('l_bnl_compress.py: detector not found')
    detector='unknown'

try:
    description=detector['description']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/description: ', description)
        print('                 detector/description[()]: ', description[()])
except:
    print('l_bnl_compress.py: detector/description not found')
    description='unknown'

try:
    detector_number=detector['detector_number']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector_number: ', detector_number)
        print('                 detector_number[()]: ', detector_number[()])
except:
    print('l_bnl_compress.py: detector/detector_number not found')
    detector_number='unknown'

try:
    bit_depth_image=detector['bit_depth_image']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/bit_depth_image: ', bit_depth_image)
        print('                 detector/bit_depth_image[()]: ', bit_depth_image[()])
except:
    print('l_bnl_compress.py: detector/bit_depth_image not found')
    bit_depth_image='unknown'

try:
    bit_depth_readout=detector['bit_depth_readout']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/bit_depth_readout: ', bit_depth_readout)
        print('                 detector/bit_depth_readout[()]: ', bit_depth_readout[()])
except:
    print('l_bnl_compress.py: detector/bit_depth_readout not found')
    bit_depth_readout = None

try:
    thickness=detector['sensor_thickness']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/sensor_thickness: ', thickness)
        print('                 detector/sensor_thickness[()]: ', thickness[()])
except:
    print('l_bnl_compress.py: detector/sensor_thickness not found')
    thickness='unknown'

pixel_mask=None
try:
    pixel_mask=detector['pixel_mask']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/pixel_mask: ', pixel_mask)
        print('                 detector/pixel_mask[()]: ', pixel_mask[()])
except:
    print('l_bnl_compress.py: detector/pixel_mask not found')
    pixel_mask=None

try:
    beamx=detector['beam_center_x']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/beam_center_x: ', beamx)
        print('                 detector/beam_center_x[()]: ', beamx[()])
except:
    print('l_bnl_compress.py: detector/beam_center_x not found')
    fin.close()
    sys.exit(-1)

try:
    beamy=detector['beam_center_y']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/beam_center_y: ', beamy)
        print('                 detector/beam_center_y[()]: ', beamy[()])
except:
    print('l_bnl_compress.py: detector/beam_center_y not found')
    fin.close()
    sys.exit(-1)

try:
    count_time=detector['count_time']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/count_time: ', count_time)
        print('                 detector/count_time[()]: ', count_time[()])
except:
    print('l_bnl_compress.py: detector/count_time not found')
    fin.close()
    sys.exit(-1)

try:
    countrate_correction_applied=detector['countrate_correction_applied']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/countrate_correction_applied: ', countrate_correction_applied)
        print('                 detector/countrate_correction_applied[()]: ', countrate_correction_applied[()])
except:
    print('l_bnl_compress.py: detector/countrate_correction_applied not found')
    countrate_correction_applied=None

try:
    distance=detector['distance']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/distance: ', distance)
        print('                 detector/distance[()]: ', distance[()])
except:
    print('l_bnl_compress.py: detector/detector_distance not found')
    try:
        distance=detector['detector_distance']
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/detector_distance: ', distance)
            print('                 detector/detector_distance[()]: ', distance[()])
    except:
        print('l_bnl_compress.py: detector/detector_distance not found')
        fin.close()
        sys.exit(-1)

try:
    frame_time=detector['frame_time']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/frame_time: ', frame_time)
        print('                   detector/frame_time[()]: ', frame_time[()])
        print('                   detector/frame_time.attrs.keys(): ',frame_time.attrs.keys())
except:
    print('l_bnl_compress.py: detector/frame_time not found')
    fin.close()
    sys.exit(-1)

try:
    pixelsizex=detector['x_pixel_size']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/x_pixel_size: ', pixelsizex)
        print('                 detector/x_pixel_size[()]: ', pixelsizex[()])
except:
    print('l_bnl_compress.py: detector/x_pixel_size not found')
    fin.close()
    sys.exit(-1)

try:
    pixelsizey=detector['y_pixel_size']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/y_pixel_size: ', pixelsizey)
        print('                 detector/y_pixel_size[()]: ', pixelsizey[()])
except:
    print('l_bnl_compress.py: detector/y_pixel_size not found')
    fin.close()
    sys.exit(-1)

try:
    detectorSpecific=fin['entry']['instrument']['detector']['detectorSpecific']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific: ', detectorSpecific)
except:
    print('l_bnl_compress.py: detectorSpecific not found')
    fin.close()
    sys.exit(-1)

try:
    xpixels=detectorSpecific['x_pixels_in_detector']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/x_pixels_in_detector: ', xpixels)
        print('                 detectorSpecific/x_pixels_in_detector[()]: ', xpixels[()])
except:
    print('l_bnl_compress.py: detectorSpecific/x_pixels_in_detector not found')
    fin.close()
    sys.exit(-1)

try:
    ypixels=detectorSpecific['y_pixels_in_detector']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/y_pixels_in_detector: ', ypixels)
        print('l_bnl_compress.py: detectorSpecific/y_pixels_in_detector: ', ypixels[()])
except:
    print('l_bnl_compress.py: detectorSpecific/y_pixels_in_detector not found')
    fin.close()
    sys.exit(-1)

try:
    xnimages=detectorSpecific['nimages']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/nimages: ', xnimages)
        print('                 detectorSpecific/nimages[()]: ', xnimages[()])
except:
    print('l_bnl_compress.py: detectorSpecific/nimages not found')
    fin.close()
    sys.exit(-1)

try:
    xntrigger=detectorSpecific['ntrigger']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/ntrigger: ', xntrigger)
        print('                 detectorSpecific/ntrigger[()]: ', xntrigger[()])
except:
    print('l_bnl_compress.py: detectorSpecific/ntrigger not found')
    fin.close()
    sys.exit(-1)

try:
    software_version=detectorSpecific['software_version']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/software_version: ', software_version)
        print('                 detectorSpecific/software_version[()]: ', software_version[()])
except:
    print('l_bnl_compress.py: detectorSpecific/software_version not found')
    fin.close()
    sys.exit(-1)

try:
    countrate_cutoff=detectorSpecific['saturation_value']
    satval=countrate_cutoff[()]
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/saturation_value: ', countrate_cutoff)
        print('                 detectorSpecific/saturation_value[()]: ', countrate_cutoff[()])
except:
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/saturation_value not found')
    try:
        countrate_cutoff=detectorSpecific['countrate_correction_count_cutoff']
        satval=countrate_cutoff[()]+1
        if args['verbose'] == True:
            print('l_bnl_compress.py: detectorSpecific/countrate_correction_count_cutoff: ',
                countrate_cutoff)
            print('                 detectorSpecific/countrate_correction_count_cutoff[()]: ',
                countrate_cutoff[()])
    except:
        if args['verbose'] == True:
            print('l_bnl_compress.py: detectorSpecific/countrate_correction_count_cutoff not found')
        try:
            countrate_cutoff=detectorSpecific['detectorModule_000']['countrate_correction_count_cutoff']
            satval=countrate_cutoff[()]+1
            if args['verbose'] == True:
                print('l_bnl_compress.py: detectorSpecific/detectorModule_000/countrate_correction_count_cutoff: ',
                    countrate_cutoff)
                print('                 detectorSpecific/detectorModule_000countrate_correction_count_cutoff[()]: ',
                    countrate_cutoff[()])
                print('                  *** use of this dataset is deprecated ***')
        except:
            print('l_bnl_sinsum.py: ...count_cutoff not found, using 32765')
            satval=32765

dS_pixel_mask = None
try:
    dS_pixel_mask=detectorSpecific['pixel_mask']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/pixel_mask: ', dS_pixel_mask)
        print('                   detectorSpecific/pixel_mask[()]: ', dS_pixel_mask[()])
except:
    print('l_bnl_compress.py: detectorSpecific/pixel_mask not found')
    dS_pixel_mask=None

best_wavelength = None
try:
    sample_wavelength=fin['entry']['sample']['beam']['incident_wavelength']
    if best_wavelength == None:
        best_wavelength = sample_wavelength
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/beam/incident_wavelength: ', sample_wavelength)
        print('                   entry/sample/beam/incident_wavelength[()]: ', sample_wavelength[()])
except:
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/beam/incident_wavelength not found')
    sample_wavelength = None

try:
    instrument_wavelength=fin['entry']['instrument']['beam']['wavelength']
    if best_wavelength == None:
        best_wavelength = instrument_wavelength
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/instrument/beam/wavelength: ', instrument_wavelength)
        print('                   entry/instrument/beam/wavelength[()]: ', instrument_wavelength[()])
except:
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/instrument/beam/wavelength not found')
    instrument_wavelength = None

try:
    monochromater_wavelength=fin['entry']['instrument']['monochromater']['wavelength']
    if best_wavelength == None:
        best_wavelength = monochromater_wavelength
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/instrument/monochromater/wavelength: ', monochromater_wavelength)
        print('                   entry/instrument/monochromater/wavelength[()]: ', monochromater_wavelength[()])
except:
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/instrument/monochromater/wavelength not found')
    monochromater_wavelength=None

try:
    beam_incident_wavelength=fin['entry']['instrument']['beam']['incident_wavelength']
    if best_wavelength == None:
        best_wavelength = beam_incident_wavelength
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/instrument/beam/incident_wavelength: ', beam_incident_wavelength)
        print('                 entry/instrument/beam/incident_wavelength[()]: ', beam_incident_wavelength[()])
except:
    print('l_bnl_compress.py:entry/instrument/beam/incident_wavelength not found')
    beam_incident_wavelength = None

if best_wavelength==None:
    print('l_bnl_compress.py: ... wavelength not found')
    fin.close()
    sys.exit(-1)

try:
    osc_width=fin['entry']['sample']['goniometer']['omega_range_average']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/goniometer/omega_range_average: ', osc_width)
        print('                 entry/sample/goniometer/omega_range_average[()]: ', osc_width[()])
except:
    print('l_bnl_compress.py: entry/sample/goniometer/omega_range_average not found')
    fin.close()
    sys.exit(-1)

nimages=xnimages[()]
ntrigger=xntrigger[()]
if args['verbose'] == True:
    print('nimages: ',nimages)
    print('ntrigger: ',ntrigger)
if int(nimages) == 1:
    nimages = int(ntrigger)
    print('l_bnl_compress.py: warning: settng nimages to ',int(nimages),' from ntrigger')

try:
    angles=fin['entry']['sample']['goniometer']['omega']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/omega: ', angles)
        print('l_bnl_compress.py: entry/sample/omega[()]: ', angles[()])
except:
   print('l_bnl_compress.py: entry/sample/gonimeter/omega not found')
   angles=None

try:
    phi=fin['entry']['sample']['goniometer']['phi']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/phi: ', phi)
        print('l_bnl_compress.py: entry/sample/phi[()]: ', phi[()])
except:
   print('l_bnl_compress.py: entry/sample/gonimeter/phi not found')
   phi=None


try:
    chi=fin['entry']['sample']['goniometer']['chi']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/chi: ', chi)
        print('l_bnl_compress.py: entry/sample/chi[()]: ', chi[()])
except:
   print('l_bnl_compress.py: entry/sample/gonimeter/chi not found')
   chi=None


try:
    kappa=fin['entry']['sample']['goniometer']['kappa']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/kappa: ', kappa)
        print('l_bnl_compress.py: entry/sample/kappa[()]: ', kappa[()])
except:
   print('l_bnl_compress.py: entry/sample/gonimeter/kappa not found')
   kappa=None


try:
    datagroup=fin['entry']['data']
except:
    print('l_bnl_compress.py: entry/data not found')
    fin.close()
    sys.exit(-1)

block_start=0
try:
    data_start=datagroup['data_000000']
except:
    block_start=1
    try:
         data_start=datagroup['data_000001']
    except:
        print('l_bnl_compress.py: first data block not found')
        fin.close()
        sys.exit(-1)

block_shape=data_start.shape
if  len(block_shape) != 3:
    print('compress.py: dimension of /entry/data data block is not 3')
    fin.close()
    sys.exit

number_per_block=int(block_shape[0])
if args['verbose']==True:
    print('l_bnl_compress.py: first data block: ', data_start)
    print('                 dir(first data block): ', dir(data_start))
    print('                 number_per_block: ',number_per_block)

print (args['first_image'],args['last_image']+1,args['sum_range'])

if (args['sum_range'] == None) or (args['sum_range']  < 2):
    args['sum_range'] = 1
if (args['bin_range'] == None) or (args['bin_range'] < 2):
    args['bin_range'] = 1

new_nimage = 0
new_images = {}

for image in range(args['first_image'],(args['last_image'])+1,args['sum_range']):
    lim_image=image+int(args['sum_range'])
    if lim_image > args['last_image']+1:
        lim_image = args['last_image']+1
    prev_out=None
    for cur_image in range(image,lim_image):
        print('image, (block,offset): ',cur_image,\
          conv_image_to_block_offset(cur_image,number_per_block))
        cur_source=conv_image_to_block_offset(cur_image,number_per_block)
        cur_source_img_block='data_'+str(cur_source[0]+block_start).zfill(6)
        cur_source_img_imgno=cur_source[1]
        cur_source=datagroup[cur_source_img_block][cur_source_img_imgno,:,:]
        print('cur_source_img_block: ', cur_source_img_block)
        print('cur_source_img_imgno: ', cur_source_img_imgno)
        if args['bin_range'] > 1:
            cur_source=bin(cur_source,int(args['bin_range']),satval)
        if cur_image > image:
            prev_out = np.clip(prev_out+cur_source,0,satval)
        else:
            prev_out = (np.asarray(cur_source,dtype='i2')).clip(0,satval)
    new_nimage = new_nimage+1
    new_images[new_nimage]=prev_out

if args['verbose'] == True:
    print('new_nimage: ', new_nimage)
    print('cur_source: ',cur_source)
    print('cur_source.shape: ', cur_source.shape)
if (args['data_block_size'] == None) or (args['data_block_size'] < 2):
    args['data_block_size'] = 1
out_number_per_block = args['data_block_size']
out_number_of_blocks = int(new_nimage+out_number_per_block-1)//out_number_per_block
out_max_image=new_nimage
if args['verbose'] == True:
    print('out_number_per_block: ', out_number_per_block)
    print('out_number_of_blocks: ', out_number_of_blocks)
fout={}

# create the master file
master=0
if args['out_master']==None:
    args['out_master']=args['out_file']+"_master"
fout[master] = h5py.File(args['out_master']+".h5",'w')
fout[master].attrs.create('default',ntstr('entry'),dtype=ntstrdt('entry'))
fout[master].create_group('entry') 
fout[master]['entry'].attrs.create('NX_class',ntstr('NXentry'),dtype=ntstrdt('NXentry'))
fout[master]['entry'].attrs.create('default',ntstr('data'),dtype=ntstrdt('data'))
fout[master]['entry'].create_group('data') 
fout[master]['entry']['data'].attrs.create('NX_class',ntstr('NXdata'),dtype=ntstrdt('NXdata')) 
fout[master]['entry'].create_group('instrument') 
fout[master]['entry']['instrument'].attrs.create('NX_class',ntstr('NXinstrument'),dtype=ntstrdt('NXinstrument'))
fout[master]['entry'].create_group('sample') 
fout[master]['entry']['sample'].attrs.create('NX_class',ntstr('NXsample'),dtype=ntstrdt('NXsample')) 
fout[master]['entry']['sample'].create_group('goniometer') 
fout[master]['entry']['sample']['goniometer'].attrs.create('NX_class',ntstr('NXgoniometer'),dtype=ntstrdt('NXgoniometer')) 
fout[master]['entry']['instrument'].attrs.create('NX_class',ntstr('NXinstrument'),dtype=ntstrdt('NXinstrument'))
fout[master]['entry']['instrument'].create_group('detector')
fout[master]['entry']['instrument']['detector'].attrs.create(\
    'NX_class',ntstr('NXdetector'),dtype=ntstrdt('NXdetector'))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'description',shape=description.shape,dtype=description.dtype)
fout[master]['entry']['instrument']['detector']['description'][()]=\
    description[()]
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'detector_number',shape=detector_number.shape,dtype=detector_number.dtype)
fout[master]['entry']['instrument']['detector']['detector_number'][()]=\
    detector_number[()]
if bit_depth_image != None:
    fout[master]['entry']['instrument']['detector'].create_dataset(\
        'bit_depth_image',shape=bit_depth_image.shape,dtype='u4')
    fout[master]['entry']['instrument']['detector']['bit_depth_image'][()]=\
        16
    fout[master]['entry']['instrument']['detector']['bit_depth_image'].attrs.create(\
        'units',ntstr('NX_UINT32'),dtype=ntstrdt('NX_UINT32'))
if bit_depth_readout != None:
    fout[master]['entry']['instrument']['detector'].create_dataset(\
        'bit_depth_readout',shape=bit_depth_readout.shape,dtype=bit_depth_readout.dtype)
    fout[master]['entry']['instrument']['detector']['bit_depth_readout'][()]=\
        bit_depth_readout[()]
    fout[master]['entry']['instrument']['detector']['bit_depth_readout'].attrs.create(\
        'units',ntstr('NX_UINT32'),dtype=ntstrdt('NX_UINT32'))
if countrate_correction_applied != None:
    fout[master]['entry']['instrument']['detector'].create_dataset(\
        'countrate_correction_applied',shape=countrate_correction_applied.shape,dtype=countrate_correction_applied.dtype)
    fout[master]['entry']['instrument']['detector']['bit_depth_image'][()]=\
        countrate_correction_applied[()]
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'sensor_thickness',shape=thickness.shape,dtype=thickness.dtype)
fout[master]['entry']['instrument']['detector']['sensor_thickness'][()]=\
    thickness[()]
fout[master]['entry']['instrument']['detector']['sensor_thickness'].attrs.create(\
    'units',ntstr(thickness.attrs['units']),dtype=ntstrdt(thickness.attrs['units']))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'beam_center_x',shape=beamx.shape,dtype=beamx.dtype)
fout[master]['entry']['instrument']['detector']['beam_center_x'][()]=\
    beamx[()]/int(args['bin_range'])
fout[master]['entry']['instrument']['detector']['beam_center_x'].attrs.create(\
    'units',ntstr(beamx.attrs['units']),dtype=ntstrdt(beamx.attrs['units']))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'beam_center_y',shape=beamy.shape,dtype=beamy.dtype)
fout[master]['entry']['instrument']['detector']['beam_center_y'][()]\
    =beamy[()]/int(args['bin_range'])
fout[master]['entry']['instrument']['detector']['beam_center_y'].attrs.create(\
   'units',ntstr(beamy.attrs['units']),dtype=ntstrdt(beamy.attrs['units']))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'count_time',shape=count_time.shape,dtype=count_time.dtype)
fout[master]['entry']['instrument']['detector']['count_time'][()]=\
    count_time[()]*int(args['sum_range'])
fout[master]['entry']['instrument']['detector']['count_time'].attrs.create(\
    'units',ntstr(count_time.attrs['units']),dtype=ntstrdt(count_time.attrs['units']))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'detector_distance',shape=distance.shape,dtype=distance.dtype)
fout[master]['entry']['instrument']['detector']['detector_distance'][()]=\
    distance[()]
fout[master]['entry']['instrument']['detector']['detector_distance'].attrs.create(\
    'units',ntstr(distance.attrs['units']),dtype=ntstrdt(distance.attrs['units']))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'frame_time',shape=frame_time.shape,dtype=frame_time.dtype)
fout[master]['entry']['instrument']['detector']['frame_time'][()]=\
    frame_time[()]*int(args['sum_range'])
fout[master]['entry']['instrument']['detector']['frame_time'].attrs.create(\
    'units',ntstr(frame_time.attrs['units']),dtype=ntstrdt(frame_time.attrs['units']))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'x_pixel_size',shape=pixelsizex.shape,dtype=pixelsizex.dtype)
fout[master]['entry']['instrument']['detector']['x_pixel_size'][()]=\
    pixelsizex[()]*int(args['sum_range'])
fout[master]['entry']['instrument']['detector']['x_pixel_size'].attrs.create(\
    'units',ntstr(pixelsizex.attrs['units']),dtype=ntstrdt(pixelsizex.attrs['units']))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'y_pixel_size',shape=pixelsizey.shape,dtype=pixelsizey.dtype)
fout[master]['entry']['instrument']['detector']['y_pixel_size'][()]=\
    pixelsizey[()]*int(args['sum_range'])
fout[master]['entry']['instrument']['detector']['y_pixel_size'].attrs.create(\
    'units',ntstr(pixelsizey.attrs['units']),dtype=ntstrdt(pixelsizey.attrs['units']))
if pixel_mask!=None:
    new_pixel_mask=conv_pixel_mask(pixel_mask,int(args['bin_range']))
    fout[master]['entry']['instrument']['detector'].create_dataset(\
        'pixel_mask',shape=new_pixel_mask.shape,dtype='u4',\
        data=new_pixel_mask,\
        **hdf5plugin.Bitshuffle(nelems=0,cname='lz4'))
    del new_pixel_mask
fout[master]['entry']['instrument']['detector'].create_group(\
    'detectorSpecific')
new_shape=conv_image_shqpe((int(ypixels[()]),int(xpixels[()])),int(args['bin_range']))
fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
    'nimages',shape=xnimages.shape,dtype=xnimages.dtype)
fout[master]['entry']['instrument']['detector']['detectorSpecific']['nimages'][()]\
    =new_nimage
fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
    'ntrigger',shape=xntrigger.shape,dtype=xntrigger.dtype)
fout[master]['entry']['instrument']['detector']['detectorSpecific']['ntrigger'][()]=\
    new_nimage
fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
    'x_pixels_in_detector',shape=xpixels.shape,dtype=xpixels.dtype)
fout[master]['entry']['instrument']['detector']['detectorSpecific'][\
    'x_pixels_in_detector'][()]=new_shape[1]
fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
    'y_pixels_in_detector',shape=ypixels.shape,dtype=ypixels.dtype)
fout[master]['entry']['instrument']['detector']['detectorSpecific']['y_pixels_in_detector'][()]=\
    new_shape[0]
fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
    'software_version',shape=software_version.shape,dtype=software_version.dtype)
fout[master]['entry']['instrument']['detector']['detectorSpecific']['software_version'][()]=\
    software_version[()]
fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
    'saturation_value',shape=countrate_cutoff.shape,dtype=countrate_cutoff.dtype)
fout[master]['entry']['instrument']['detector']['detectorSpecific']['saturation_value'][()]=\
    satval
if dS_pixel_mask!=None:
    new_pixel_mask=conv_pixel_mask(dS_pixel_mask,int(args['bin_range']))
    fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
        'pixel_mask',shape=new_pixel_mask.shape,dtype='u4',\
        data=new_pixel_mask,chunks=None,\
        **hdf5plugin.Bitshuffle(nelems=0,cname='lz4'))
    del new_pixel_mask
elif pixel_mask!=None:
    new_pixel_mask=conv_pixel_mask(pixel_mask,int(args['bin_range']))
    fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
        'pixel_mask',shape=pixel_mask.shape,dtype='u4',\
        data=new_pixel_mask,chunks=None,\
        **hdf5plugin.Bitshuffle(nelems=0,cname='lz4'))
    del new_pixel_mask

if sample_wavelength!=None:
    if not ('beam' in fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('beam') 
        fout[master]['entry']['sample']['beam'].attrs.create('NX_class',ntstr('NXbeam'),dtype=ntstrdt('NXbeam'))
    if not ('incident_wavelength' in  fout[master]['entry']['sample']['beam'].keys()): 
        fout[master]['entry']['sample']['beam'].create_dataset(\
        'incident_wavelength',shape=sample_wavelength.shape,dtype=sample_wavelength.dtype)
    fout[master]['entry']['sample']['beam']['incident_wavelength'][()]=sample_wavelength[()]
if instrument_wavelength!=None:
    if not ('beam' in fout[master]['entry']['instrument'].keys()):
        fout[master]['entry']['instrument'].create_group('beam')
        fout[master]['entry']['instrument']['beam'].attrs.create('NX_class',ntstr('NXbeam'),dtype=ntstrdt('NXbeam'))
    if not ('wavelength' in fout[master]['entry']['instrument']['beam'].keys()):
        fout[master]['entry']['instrument']['beam'].create_dataset(\
        'wavelength',shape=instrument_wavelength.shape,dtype=instrument_wavelength.dtype)
    fout[master]['entry']['instrument']['beam']['wavelength'][()]=instrument_wavelength[()]
if monochromater_wavelength!=None:
    if not ('monochromater' in fout[master]['entry']['instrument'].keys()):
        fout[master]['entry']['instrument'].create_group('monochromater')
        fout[master]['entry']['instrument']['monochromater'].attrs.create('NX_class',ntstr('NXmonochromater'),dtype=ntstrdt('NXmonochromater'))
    if not ('wavelength' in fout[master]['entry']['instrument']['monochromater'].keys()):
        fout[master]['entry']['instrument']['monochromater'].create_dataset(\
        'wavelength',shape=monochromater_wavelength.shape,dtype=monochromater_wavelength.dtype)
    fout[master]['entry']['instrument']['monochromater']['wavelength'][()]=monochromater_wavelength[()]
if beam_incident_wavelength!=None:
    if not ('beam' in fout[master]['entry']['instrument'].keys()):
        fout[master]['entry']['instrument'].create_group('beam')
        fout[master]['entry']['instrument']['beam'].attrs.create('NX_class',ntstr('NXbeam'),dtype=ntstrdt('NXbeam'))
    if not ('incident_wavelength' in fout[master]['entry']['instrument']['beam'].keys()):
        fout[master]['entry']['instrument']['beam'].create_dataset(\
        'incident_wavelength',shape=beam_incident_wavelength.shape,dtype=beam_incident_wavelength.dtype)
    fout[master]['entry']['instrument']['beam']['incident_wavelength'][()]=beam_incident_wavelength[()]

fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'omega_range_average',shape=osc_width.shape,dtype=osc_width.dtype)
fout[master]['entry']['sample']['goniometer']['omega_range_average'][()]=\
    osc_width[()]*int(args['sum_range'])
fout[master]['entry']['sample']['goniometer']['omega_range_average'].attrs.create(\
    'units',ntstr(osc_width.attrs['units']),dtype=ntstrdt(osc_width.attrs['units']))
fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'omega',shape=angles.shape,dtype=angles.dtype) 
fout[master]['entry']['sample']['goniometer']['omega'][0:(int(args['last_image'])-\
    int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
    angles[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
fout[master]['entry']['sample']['goniometer']['omega_end'] = \
    fout[master]['entry']['sample']['goniometer']['omega']+osc_width[()]*int(args['sum_range'])
if phi != None:
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'phi',shape=phi.shape,dtype=phi.dtype)
    fout[master]['entry']['sample']['goniometer']['phi'][()]=phi[()]
if chi != None:
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'chi',shape=chi.shape,dtype=chi.dtype)
    fout[master]['entry']['sample']['goniometer']['phi'][()]=chi[()]
if kappa != None:
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'kappa',shape=kappa.shape,dtype=kappa.dtype)
    fout[master]['entry']['sample']['goniometer']['phi'][()]=kappa[()]

for nout_block in range(1,out_number_of_blocks+1):
    nout_image=1+(nout_block-1)*out_number_per_block
    image_nr_low=nout_image
    lim_nout_image = nout_image+out_number_per_block
    if lim_nout_image > out_max_image+1:
        lim_nout_image = out_max_image+1
    image_nr_high = lim_nout_image-1
    nout_data_shape = new_images[nout_image].shape
    fout[nout_block] = h5py.File(args['out_file']+"_"+str(nout_block).zfill(6)+".h5",'w')
    fout[nout_block].create_group('entry')
    fout[nout_block]['entry'].attrs.create('NX_class',ntstr('NXentry'),dtype=ntstrdt('NXentry'))
    fout[nout_block]['entry'].create_group('data')
    fout[nout_block]['entry']['data'].attrs.create('NX_class',ntstr('NXdata'),dtype=ntstrdt('NXdata'))
    if args['compression']==None:
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype='u2',chunks=(1,nout_data_shape[0],nout_data_shape[1]))
    elif args['compression']=='bshuf' or args['compression']=='BSHUF':
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype='u2',chunks=(1,nout_data_shape[0],nout_data_shape[1]),
            **hdf5plugin.Bitshuffle(nelems=0,cname='none'))
    elif args['compression']=='bslz4' or args['compression']=='BSLZ4':
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype='u2',chunks=(1,nout_data_shape[0],nout_data_shape[1]),
            **hdf5plugin.Bitshuffle(nelems=0,cname='lz4'))
    elif args['compression']=='bzstd' or args['compression']=='BSZSTD':
        if args['compression-level'] == None:
            clevel=3
        elif int(args['compression-level']) > 22:
            clevel=22
        elif int(args['compression-level']) < -2:
            clevel=-2
        else:
            clevel=int(args['compression-level'])
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype='u2',chunks=(1,nout_data_shape[0],nout_data_shape[1]),
            **hdf5plugin.Bitshuffle(nelems=0,cname='zstd',clevel=clevel))
    fout[nout_block]['entry']['data']['data'].attrs.create('image_nr_low',dtype=np.uint64,
        data=np.uint64(image_nr_low))
    fout[nout_block]['entry']['data']['data'].attrs.create('image_nr_high',dtype=np.uint64,
        data=np.uint64(image_nr_high))
    print("fout[nout_block]['entry']['data']['data']: ",fout[nout_block]['entry']['data']['data'])
    for out_image in range(nout_image,lim_nout_image):
        if args['hcomp_scale']==None and args['j2k_target_compression_ratio']==None: 
            fout[nout_block]['entry']['data']['data'][out_image-nout_image,0:nout_data_shape[0],0:nout_data_shape[1]] \
              =np.clip(new_images[out_image][0:nout_data_shape[0],0:nout_data_shape[1]],0,satval)
        elif args['hcomp_scale']!=None:
            myscale=args['hcomp_scale']
            if myscale < 1 :
                myscale=16
            img16=np.asarray(new_images[out_image][0:nout_data_shape[0],0:nout_data_shape[1]],dtype='i2')
            img32=(np.clip(img16,0,satval)).astype('i4')
            hcomp = HCompress1(scale=int(myscale),smooth=False,bytepix=4,nx=nout_data_shape[0],ny=nout_data_shape[1])
            hcomp_data=hcomp.encode(img32)
            hdecomp_data=hcomp.decode(np.frombuffer(hcomp_data,dtype=np.uint8))
            decompressed_data = hdecomp_data.astype('i4').reshape((nout_data_shape[0],nout_data_shape[1]))
            decompressed_data = np.clip(decompressed_data,0,satval)
            fout[nout_block]['entry']['data']['data'][out_image-nout_image, \
                0:nout_data_shape[0],0:nout_data_shape[1]] \
                = np.asarray(decompressed_data,dtype='u2')
            del decompressed_data
            del hdecomp_data
            del hcomp_data
            del img32
        else:
            mycrat=int(args['j2k_target_compression_ratio'])
            if mycrat < 1:
                mycrat=125
            img16=new_images[out_image][0:nout_data_shape[0],0:nout_data_shape[1]].astype('u2')
            outtemp=args['out_file']+"_"+str(out_image).zfill(6)+".j2k"
            print("outtemp: ",outtemp)
            j2k=glymur.Jp2k(outtemp, data=img16, cratios=[mycrat])
            jdecomped = j2k[:]
            arr_final = np.array(jdecomped, dtype='u2')
            fout[nout_block]['entry']['data']['data'][out_image-nout_image, \
                0:nout_data_shape[0],0:nout_data_shape[1]] \
                = np.clip(arr_final,0,satval)
            del arr_final
            del jdecomped
            del j2k
            os.remove(outtemp)
    fout[nout_block].close()
    fout[master]['entry']['data']["data_"+str(nout_block).zfill(6)] \
        = h5py.ExternalLink(args['out_file']+"_"+str(nout_block).zfill(6)+".h5", "/entry/data/data")
fout[master].close()
