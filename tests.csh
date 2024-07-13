#!/bin/csh
set verbose
nice cctbx.python l_bnl_compress.py  -i JLJ735_14_l_Raster_7785_master.h5  -1 1 -N 29 -b 2 -s 2 -v -d 5 -H 4 -m JLJ735_14_l_Raster_7785_b2s2h4_master -o JLJ735_14_l_Raster_7785_b2s2h4 -c bslz4
tar -cf - ./JLJ735_14_l_Raster_7785_b2s2h4*.h5 | zstd -6 > JLJ735_14_l_Raster_7785_b2s2h4.tar.zstd
nice cctbx.python l_bnl_compress.py  -i JLJ735_14_l_Raster_7785_master.h5  -1 1 -N 29 -b 2 -s 2 -v -d 5 -H 16 -m JLJ735_14_l_Raster_7785_b2s2h16_master -o JLJ735_14_l_Raster_7785_b2s2h16 -c bslz4
tar -cf - ./JLJ735_14_l_Raster_7785_b2s2h16*.h5 | zstd -6 > JLJ735_14_l_Raster_7785_b2s2h16.tar.zstd
nice cctbx.python l_bnl_compress.py  -i JLJ735_14_l_Raster_7785_master.h5  -1 1 -N 29 -b 2 -s 2 -v -d 5 -J 75 -m JLJ735_14_l_Raster_7785_b2s2j75_master -o JLJ735_14_l_Raster_7785_b2s2j75 -c bslz4
tar -cf - ./JLJ735_14_l_Raster_7785_b2s2j75*.h5 | zstd -6 > JLJ735_14_l_Raster_7785_b2s2j75.tar.zstd
nice cctbx.python l_bnl_compress.py  -i JLJ735_14_l_Raster_7785_master.h5  -1 1 -N 29 -b 2 -s 2 -v -d 5 -J 500 -m JLJ735_14_l_Raster_7785_b2s2j500_master -o JLJ735_14_l_Raster_7785_b2s2j500 -c bslz4
tar -cf - ./JLJ735_14_l_Raster_7785_b2s2j500*.h5 | zstd -6 > JLJ735_14_l_Raster_7785_b2s2j500.tar.zstd
rm  ./tlys-1023_14689_b2s2h4*.h5
nice cctbx.python l_bnl_compress.py  -i tlys-1023_14689_master.h5  -1 1 -N 20 -b 2 -s 2 -v -d 5 -H 4  -o tlys-1023_14689_b2s2h4 -c bslz4
tar -cf - ./tlys-1023_14689_b2s2h4*.h5 | zstd -6 > tlys-1023_14689_b2s2h4.tar.zstd
rm  ./tlys-1023_14689_b2s2h16*.h5
nice cctbx.python l_bnl_compress.py  -i tlys-1023_14689_master.h5  -1 1 -N 20 -b 2 -s 2 -v -d 5 -H 16 -o tlys-1023_14689_b2s2h16 -c bslz4
tar -cf - ./tlys-1023_14689_b2s2h16*.h5 | zstd -6 > tlys-1023_14689_b2s2h16.tar.zstd
rm  ./tlys-1023_14689_b1s1h16*.h5
nice cctbx.python l_bnl_compress.py  -i tlys-1023_14689_master.h5  -1 1 -N 20 -b 1 -s 1 -v -d 5 -H 16 -o tlys-1023_14689_b1s1h16 -c bslz4
tar -cf - ./tlys-1023_14689_b1s1h16*.h5 | zstd -6 > tlys-1023_14689_b1s1h16.tar.zstd
rm  ./tlys-1023_14689_b1s1h32*.h5
nice cctbx.python l_bnl_compress.py  -i tlys-1023_14689_master.h5  -1 1 -N 20 -b 1 -s 1 -v -d 5 -H 32 -o tlys-1023_14689_b1s1h32 -c bslz4
tar -cf - ./tlys-1023_14689_b1s1h32*.h5 | zstd -6 > tlys-1023_14689_b1s1h32.tar.zstd
rm  ./tlys-1023_14689_b1s1h64*.h5
nice cctbx.python l_bnl_compress.py  -i tlys-1023_14689_master.h5  -1 1 -N 20 -b 1 -s 1 -v -d 5 -H 64 -o tlys-1023_14689_b1s1h64 -c bslz4
tar -cf - ./tlys-1023_14689_b1s1h64*.h5 | zstd -6 > tlys-1023_14689_b1s1h64.tar.zstd
rm  ./tlys-1023_14689_b2s2j75*.h5
nice cctbx.python l_bnl_compress.py  -i tlys-1023_14689_master.h5  -1 1 -N 20 -b 2 -s 2 -v -d 5 -J 75  -o tlys-1023_14689_b2s2j75 -c bslz4
tar -cf - ./tlys-1023_14689_b2s2j75*.h5 | zstd -6 > tlys-1023_14689_b2s2j75.tar.zstd
rm  ./tlys-1023_14689_b2s2j500*.h5
nice cctbx.python l_bnl_compress.py  -i tlys-1023_14689_master.h5  -1 1 -N 20 -b 2 -s 2 -v -d 5 -J 500  -o tlys-1023_14689_b2s2j500 -c bslz4
tar -cf - ./tlys-1023_14689_b2s2j500*.h5 | zstd -6 > tlys-1023_14689_b2s2j500.tar.zstd
