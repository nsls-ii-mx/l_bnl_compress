'''l_bnl_compress.py, lossy, but not lossy, compresss,
       a script to apply lossy compression to HDF5 MX image files.

  (C) Copyright 16 March 2025 Herbert J. Bernstein
  Portions suggested by claude.ai from Anthropic
  You may redistribute l_bnl_compress.py under GPL2 or LGPL2
  Rev 26 Jul 2025 Herbert J. Bernstein, add log and exp 
 
usage: l_bnl_compress.py [-h] [-1 FIRST_IMAGE] [-b BIN_RANGE] [-c COMPRESSION] [-d DATA_BLOCK_SIZE] \
                         [-f SIZE] [-H HCOMP_SCALE] [-i INFILE] [-J J2K_TARGET_COMPRESSION_RATIO] \
                         [-l COMPRESSION_LEVEL] [-m OUT_MASTER] [-N LAST_IMAGE] [-o OUT_FILE] \
                         [-q OUT_SQUASH] [-s SUM_RANGE] [-S SCALE] [-v]

Bin and sum images from a range

options:
  -h, --help            show this help message and exit
  -1 FIRST_IMAGE, --first_image FIRST_IMAGE
                        first selected image counting from 1
  -b BIN_RANGE, --bin BIN_RANGE
                        an integer image binning range (1 ...) to apply to each selected image
  -c COMPRESSION, --compression COMPRESSION
                        optional compression, bslz4, bszstd, bshuf, or zstd
  -d DATA_BLOCK_SIZE, --data_block_size DATA_BLOCK_SIZE
                        data block size in images for out_file
  -f UFLOAT, --float UFLOAT
                        clip the output above 0 and limit to 2 byte or 4 byte floats
  -H HCOMP_SCALE, --Hcompress HCOMP_SCALE
                        Hcompress scale compression, immediately followed by decompression
  -i INFILE, --infile INFILE
                        the input hdf5 file to read images from
  -J J2K_TARGET_COMPRESSION_RATIO, --J2K J2K_TARGET_COMPRESSION_RATIO
                        JPEG-2000 target compression ratio, immediately followed by decompression
  -K J2K_ALT_TARGET_COMPRESSION_RATIO, --J2K J2K_ALT_TARGET_COMPRESSION_RATIO
                        JPEG-2000 alternatetarget compression ratio, immediately followed by decompression
  -l COMPRESSION_LEVEL, --compression_level COMPRESSION_LEVEL
                        optional compression level for bszstd or zstd
  -L LOGARITHM_BASE, --logarithm LOGARITHM_BASE
                        convert non-negative values v to log(v+1), negative values to-log(-v+1)
                        as short float, if --uint is not specified or whatever is specified by --unit or by --float 
  -m OUT_MASTER, --out_master OUT_MASTER
                        the output hdf5 master to which to write metadata, defaults to OUT_FILE_MASTER
                        if not given, out given as out_file
  -N LAST_IMAGE, --last_image LAST_IMAGE
                        last selected image counting from 1
  -o OUT_FILE, --out_file OUT_FILE
                        the output hdf5 data file out_file_?????? with an .h5 extension are files to which to write images
  -p THREADS, -- parallel THREADS
                        the number of parallel threads with to spawn to generate -H, -J, or -K data files in parallel 
  -q OUT_SQUASH, --out_squash OUT_SQUASH
                        an optional hdf5 data file out_squash_?????? with an .h5 extension are optional files to which
                        raw j2k or hcomp files paralleling OUT_FILE are written, defaults to OUT_FILE_SQUASH
                        if given as out_file 
  -s SUM_RANGE, --sum SUM_RANGE
                        an integer image summing range (1 ...) to apply to the selected images
  -S SCALE, --scale SCALE
                        a non negative scaling factor to apply both the the images and satval
  -t THREAD_NO, --thread THREAD_NO
                        process the given thread number, 0, for the master file only, which runs first by itself and 
                        produces no data files
  -u UINT ,--uint UINT
                        clip the output above 0 and limit to 2 byte or 4 byte integers 
  -v, --verbose         provide addtional information

  -V, --version         report the version and build_date

  -X EXPONENT, --exponential EXPONENT
                        convert non-negative calues v to exp(v)-1 and negative values to -exp(-v)+1
                        as short unsigned int, if --float is not specified  or whatever is specified by --unit or by --float

'''

import sys
import os
import argparse
import numpy as np
import skimage as ski
import h5py
import tifffile
from astropy.io import fits
from astropy.io.fits.hdu.compressed import COMPRESSION_TYPES
import glymur
import hdf5plugin
import tempfile
import numcodecs
from astropy.io.fits.hdu.compressed._codecs import HCompress1
from io import BytesIO

import h5py
from PIL import Image
import warnings
from PIL import TiffTags, TiffImagePlugin
import subprocess
import threading
import time
import queue
import string

from collections import OrderedDict
import gc

import h5py
from collections import OrderedDict
import threading
import time
import gc

class HDF5DatasetBuffer:
    """
    Efficient buffering system for large HDF5 datasets with automatic caching,
    chunking, and memory management.
    """
    
    def __init__(self, filepath, max_cache_size=50, max_memory_mb=2048):
        """
        Initialize the buffer manager.
        
        Args:
            filepath: Path to HDF5 file
            max_cache_size: Maximum number of datasets to keep in memory
            max_memory_mb: Maximum memory usage in MB
        """
        self.filepath = filepath
        self.max_cache_size = max_cache_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # LRU cache using OrderedDict
        self.cache = OrderedDict()
        self.cache_sizes = {}  # Track memory usage per dataset
        self.current_memory = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        
    def _get_optimal_chunks(self, shape, dtype, target_chunk_mb=1):
        """Calculate optimal chunk size for HDF5 dataset."""
        element_size = np.dtype(dtype).itemsize
        target_elements = (target_chunk_mb * 1024 * 1024) // element_size
        
        if len(shape) == 1:
            chunk_size = min(shape[0], target_elements)
            return (chunk_size,)
        elif len(shape) == 2:
            # For 2D arrays, try to keep chunks roughly square
            sqrt_elements = int(np.sqrt(target_elements))
            chunk_rows = min(shape[0], sqrt_elements)
            chunk_cols = min(shape[1], target_elements // chunk_rows)
            return (chunk_rows, chunk_cols)
        else:
            # For higher dimensions, distribute evenly
            chunk_per_dim = int(target_elements ** (1/len(shape)))
            return tuple(min(s, chunk_per_dim) for s in shape)
    
    def create_dataset(self, name, data, compression='lzf', shuffle=True):
        """
        Create a new dataset with optimal chunking and compression.
        
        Args:
            name: Dataset name
            data: NumPy array data
            compression: Compression algorithm ('gzip', 'lzf', 'szip')
            shuffle: Whether to shuffle bytes for better compression
        """
        with h5py.File(self.filepath, 'a') as f:
            if name in f:
                del f[name]  # Replace existing dataset
            
            chunks = self._get_optimal_chunks(data.shape, data.dtype)
            
            dset = f.create_dataset(
                name, 
                data=data,
                dtype=data.dtype,
                shape=data.shape,
                chunks=chunks,
                compression=compression,
                shuffle=shuffle,
                fletcher32=True  # Checksum for data integrity
            )
            
            # Add metadata
            dset.attrs['created'] = time.time()
            dset.attrs['chunk_size'] = chunks
            
        print(f"Created dataset '{name}' with chunks {chunks}, compression: {compression}")
    
    def _evict_if_needed(self, required_bytes):
        """Evict datasets from cache if memory limit would be exceeded."""
        while (self.current_memory + required_bytes > self.max_memory_bytes or 
               len(self.cache) >= self.max_cache_size):
            if not self.cache:
                break
                
            # Remove least recently used item
            oldest_name, oldest_data = self.cache.popitem(last=False)
            freed_bytes = self.cache_sizes.pop(oldest_name)
            self.current_memory -= freed_bytes
            
            print(f"Evicted '{oldest_name}' ({freed_bytes / (1024*1024):.1f} MB)")
    
    def get_dataset(self, name, use_cache=True):
        """
        Retrieve dataset with intelligent caching.
        
        Args:
            name: Dataset name
            use_cache: Whether to use/update cache
            
        Returns:
            NumPy array containing dataset data
        """
        with self.lock:
            # Check cache first
            if use_cache and name in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(name)
                self.hits += 1
                return self.cache[name]
            
            # Load from disk
            self.misses += 1
            with h5py.File(self.filepath, 'r') as f:
                if name not in f:
                    raise KeyError(f"Dataset '{name}' not found")
                
                data = f[name][...]  # Load entire dataset
                
            if use_cache:
                # Calculate memory footprint
                data_bytes = data.nbytes
                
                # Evict old data if necessary
                self._evict_if_needed(data_bytes)
                
                # Add to cache
                self.cache[name] = data.copy()  # Ensure we own the memory
                self.cache_sizes[name] = data_bytes
                self.current_memory += data_bytes
                
                print(f"Cached '{name}' ({data_bytes / (1024*1024):.1f} MB)")
            
            return data
    
    def get_dataset_slice(self, name, slice_obj):
        """
        Get a slice of a dataset without loading the entire dataset.
        Useful for very large datasets where you only need part of the data.
        """
        with h5py.File(self.filepath, 'r') as f:
            if name not in f:
                raise KeyError(f"Dataset '{name}' not found")
            return f[name][slice_obj]
    
    def update_dataset(self, name, data, start_idx=None):
        """
        Update dataset on disk and invalidate cache.
        
        Args:
            name: Dataset name
            data: New data
            start_idx: Starting index for partial updates (tuple)
        """
        with self.lock:
            with h5py.File(self.filepath, 'a') as f:
                if name not in f:
                    raise KeyError(f"Dataset '{name}' not found")
                
                if start_idx is None:
                    f[name][...] = data
                else:
                    f[name][start_idx] = data
            
            # Invalidate cache
            if name in self.cache:
                freed_bytes = self.cache_sizes.pop(name)
                self.current_memory -= freed_bytes
                del self.cache[name]
                print(f"Invalidated cache for '{name}'")
    
    def prefetch_datasets(self, dataset_names):
        """
        Preload multiple datasets into cache.
        Useful when you know which datasets you'll need soon.
        """
        for name in dataset_names:
            try:
                self.get_dataset(name, use_cache=True)
            except KeyError:
                print(f"Warning: Dataset '{name}' not found, skipping prefetch")
    
    def list_datasets(self):
        """List all datasets in the HDF5 file with their properties."""
        datasets = []
        with h5py.File(self.filepath, 'r') as f:
            def visit_func(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets.append({
                        'name': name,
                        'shape': obj.shape,
                        'dtype': obj.dtype,
                        'size_mb': obj.size * obj.dtype.itemsize / (1024*1024),
                        'chunks': obj.chunks,
                        'compression': obj.compression
                    })
            
            f.visititems(visit_func)
        
        return datasets
    
    def get_cache_stats(self):
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_rate_percent': hit_rate,
            'cache_size': len(self.cache),
            'memory_usage_mb': self.current_memory / (1024*1024),
            'cached_datasets': list(self.cache.keys())
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        with self.lock:
            self.cache.clear()
            self.cache_sizes.clear()
            self.current_memory = 0
            gc.collect()  # Force garbage collection
            print("Cache cleared")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_cache()

'''
# Example usage and utility functions
def create_sample_datasets(buffer_manager, num_datasets=10):
    """Create sample datasets for testing."""
    print(f"Creating {num_datasets} sample datasets...")
    
    for i in range(num_datasets):
        # Create 5MB datasets (roughly 1.25M float32 values)
        data = np.random.random((1250000,)).astype(np.float32)
        dataset_name = f"dataset_{i:04d}"
        buffer_manager.create_dataset(dataset_name, data)
    
    print("Sample datasets created!")

def benchmark_access_patterns(buffer_manager, dataset_names):
    """Benchmark different access patterns."""
    print("\n=== Benchmarking Access Patterns ===")
    
    # Random access pattern
    import random
    random_names = random.sample(dataset_names, min(20, len(dataset_names)))
    
    start_time = time.time()
    for name in random_names:
        data = buffer_manager.get_dataset(name)
        # Simulate some processing
        result = np.mean(data)
    
    random_time = time.time() - start_time
    stats = buffer_manager.get_cache_stats()
    
    print(f"Random access: {random_time:.2f}s, Hit rate: {stats['hit_rate_percent']:.1f}%")
    
    # Sequential access pattern (should have lower hit rate initially)
    buffer_manager.clear_cache()
    start_time = time.time()
    
    for name in dataset_names[:20]:
        data = buffer_manager.get_dataset(name)
        result = np.mean(data)
    
    sequential_time = time.time() - start_time
    stats = buffer_manager.get_cache_stats()
    
    print(f"Sequential access: {sequential_time:.2f}s, Hit rate: {stats['hit_rate_percent']:.1f}%")
'''

'''
# Usage example
if __name__ == "__main__":
    # Initialize buffer manager
    with HDF5DatasetBuffer("large_datasets.h5", max_cache_size=50, max_memory_mb=1024) as buffer:
        
        # Create sample datasets (uncomment to create test data)
        # create_sample_datasets(buffer, num_datasets=100)
        
        # List all datasets
        datasets = buffer.list_datasets()
        print(f"Found {len(datasets)} datasets")
        
        # Example: Access datasets with caching
        dataset_names = [d['name'] for d in datasets[:10]]
        
        # Prefetch datasets you know you'll need
        buffer.prefetch_datasets(dataset_names[:5])
        
        # Access datasets (will use cache when available)
        for name in dataset_names:
            data = buffer.get_dataset(name)
            print(f"Loaded {name}: shape {data.shape}, mean = {np.mean(data):.4f}")
        
        # Get performance statistics
        stats = buffer.get_cache_stats()
        print(f"\nCache Stats: {stats}")
        
        # Example: Partial dataset access for very large datasets
        # slice_data = buffer.get_dataset_slice("dataset_0001", slice(0, 1000))
        

        
        # Benchmark different access patterns
        if len(datasets) > 0:
            benchmark_access_patterns(buffer, [d['name'] for d in datasets])
'''


def exp_base(base, x, out=None):
    """
    Numerically stable computation of base^x for arbitrary bases.
    
    This function computes base^x in a numerically stable way by using
    the identity: base^x = exp(x * ln(base))
    
    For better numerical stability, it handles special cases and uses
    log-space arithmetic when appropriate.
    
    Parameters:
    -----------
    base : array_like
        The base(s). Must be positive for real results.
    x : array_like
        The exponent(s).
    out : ndarray, optional
        Output array to store results.
        
    Returns:
    --------
    ndarray
        base^x computed in a numerically stable manner.
        
    Examples:
    ---------
    >>> exp_base(2, 10)
    1024.0
    >>> exp_base(np.e, 2)  # Should be close to np.exp(2)
    7.38905609893065
    >>> exp_base([2, 3, 10], [8, 4, 2])
    array([ 256.,   81.,  100.])
    """
    base = np.asarray(base, dtype=float)
    x = np.asarray(x, dtype=float)
    
    # Handle special cases for numerical stability
    result_shape = np.broadcast_shapes(base.shape, x.shape)
    if out is None:
        result = np.empty(result_shape, dtype=float)
    else:
        result = out
        if result.shape != result_shape:
            raise ValueError(f"Output array shape {result.shape} doesn't match broadcast shape {result_shape}")
    
    # Broadcast inputs for element-wise operations
    base_bc, x_bc = np.broadcast_arrays(base, x)
    result_flat = result.ravel()
    base_flat = base_bc.ravel()
    x_flat = x_bc.ravel()
    
    for i in range(len(result_flat)):
        b, exp_val = base_flat[i], x_flat[i]
        
        # Handle special cases
        if b <= 0:
            if b == 0:
                result_flat[i] = 0.0 if exp_val > 0 else (1.0 if exp_val == 0 else np.inf)
            else:
                # Negative base - only defined for integer exponents in real domain
                if np.isclose(exp_val, np.round(exp_val)):
                    result_flat[i] = np.power(b, exp_val)
                else:
                    result_flat[i] = np.nan
        elif b == 1:
            result_flat[i] = 1.0
        elif exp_val == 0:
            result_flat[i] = 1.0
        elif exp_val == 1:
            result_flat[i] = b
        elif np.isinf(exp_val):
            if b > 1:
                result_flat[i] = np.inf if exp_val > 0 else 0.0
            elif b < 1:
                result_flat[i] = 0.0 if exp_val > 0 else np.inf
            else:  # b == 1, already handled above
                result_flat[i] = 1.0
        else:
            # General case: use log-space computation for stability
            # base^x = exp(x * ln(base))
            log_base = np.log(b)
            log_result = exp_val * log_base
            
            # Check for potential overflow/underflow in exp
            if log_result > 700:  # exp(700) is near overflow threshold
                result_flat[i] = np.inf
            elif log_result < -700:  # exp(-700) is effectively zero
                result_flat[i] = 0.0
            else:
                result_flat[i] = np.exp(log_result)
    
    return result.reshape(result_shape)


def exp_base_vectorized(base, x, out=None):
    """
    Vectorized version of exp_base for better performance on large arrays.
    
    This version uses numpy's vectorized operations where possible while
    still maintaining numerical stability.
    """
    base = np.asarray(base, dtype=float)
    x = np.asarray(x, dtype=float)
    
    # Handle the general case using log-space arithmetic
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        log_base = np.log(base)
        log_result = x * log_base
        
        # Create result array
        result = np.exp(log_result, out=out)
        
        # Handle special cases that need correction
        # base = 1: result should be 1
        result = np.where(base == 1, 1.0, result)
        
        # x = 0: result should be 1 (except when base = 0)
        result = np.where((x == 0) & (base != 0), 1.0, result)
        
        # base = 0: handle separately
        zero_base_mask = (base == 0)
        result = np.where(zero_base_mask & (x > 0), 0.0, result)
        result = np.where(zero_base_mask & (x == 0), 1.0, result)
        result = np.where(zero_base_mask & (x < 0), np.inf, result)
        
        # Handle negative bases (set to NaN for non-integer exponents)
        neg_base_mask = base < 0
        non_int_exp_mask = ~np.isclose(x, np.round(x))
        result = np.where(neg_base_mask & non_int_exp_mask, np.nan, result)
        
        # For negative bases with integer exponents, use np.power
        int_exp_mask = neg_base_mask & ~non_int_exp_mask
        if np.any(int_exp_mask):
            result = np.where(int_exp_mask, np.power(base, x), result)
    
    return result

'''
# Example usage and testing
if __name__ == "__main__":
    # Test cases
    print("Testing exp_base function:")
    
    # Basic tests
    print(f"exp_base(2, 10) = {exp_base(2, 10)}")  # Should be 1024
    print(f"exp_base(np.e, 2) = {exp_base(np.e, 2)}")  # Should be close to exp(2)
    print(f"np.exp(2) = {np.exp(2)}")  # For comparison
    
    # Array tests
    bases = np.array([2, 3, 10])
    exponents = np.array([8, 4, 2])
    print(f"exp_base({bases}, {exponents}) = {exp_base(bases, exponents)}")
    
    # Edge cases
    print(f"exp_base(1, 100) = {exp_base(1, 100)}")  # Should be 1
    print(f"exp_base(2, 0) = {exp_base(2, 0)}")      # Should be 1
    print(f"exp_base(0, 2) = {exp_base(0, 2)}")      # Should be 0
    print(f"exp_base(0, 0) = {exp_base(0, 0)}")      # Should be 1
    
    # Test vectorized version
    print("\nTesting vectorized version:")
    print(f"exp_base_vectorized(2, 10) = {exp_base_vectorized(2, 10)}")
    
    # Performance comparison for large arrays
    import time
    
    large_base = np.random.uniform(1.1, 10, 10000)
    large_exp = np.random.uniform(-5, 5, 10000)
    
    # Time the vectorized version
    start = time.time()
    result_vec = exp_base_vectorized(large_base, large_exp)
    vec_time = time.time() - start
    
    # Time naive np.power for comparison
    start = time.time()
    result_naive = np.power(large_base, large_exp)
    naive_time = time.time() - start
    
    print(f"\nVectorized version time: {vec_time:.4f}s")
    print(f"Naive np.power time: {naive_time:.4f}s")
    print(f"Max difference: {np.max(np.abs(result_vec - result_naive))}")

'''

def log_transform(arr,base='e',data_signed=False,data_size=2,data_type=np.uint16):
    """
    Transform array values >= 0 by adding 1 and applying natural log,
    If unsigned convert negative values to zero
    If signed, convert negative values, v, convert to -log(-v+1)
    
    Parameters:
    arr : numpy.ndarray
        2D numpy array of any numeric type
        
    Returns:
    numpy.ndarray
        Array of same shape with dtype data_type, containing ln(x+1) for x >= 0
        Values < 0 are set to 0 if data_signed=False
        Values v < 0 are set to -ln(-v+1)
    """
    # get logarithm base
    log_base=np.exp(1.)
    log_base_div=1.
    if base != 'e':
        log_base = float(base)
        if log_base < 2.:
            log_base = 2.
        log_base_div = np.log(log_base)
        

    # Convert to data_type for output
    result = np.full(arr.shape, 0, dtype=np.float32)
    
    # Create mask for values >= 0
    mask = arr >= 0

    # Create mask for values < 0
    mask_neg = arr < 0
    
    # Apply transformation: ln(x + 1) for x >= 0
    result[mask] = np.log(arr[mask] + 1).astype(np.float32)
    if data_signed:
        result[mask_neg] = -np.log(-arr[mask_neg]+1).astype(np.float32)
    else:
        result[mask_neg] = 0
    
    return (result/log_base_div).astype(data_type)

def inverse_log_transform(arr,base='e',data_signed=False,data_size=2,data_type=np.uint16):
    """
    Inverse transform by applying exp and subtracting 1.
    
    Parameters:
    arr : numpy.ndarray
        Array containing log-transformed values
        
    Returns:
    numpy.ndarray
        Array of same shape containing exp(x) - 1
    """

    # Convert to data_type for output
    result = np.full(arr.shape, 0, dtype=np.float32)
  

    # Create mask for values >= 0 
    mask = arr >= 0

    # Create mask for values < 0
    mask_neg = arr < 0

    result[mask] = exp_base(base, arr[mask]).astype(np.float32)-1
    if data_signed:
        result[mask_neg] = exp_base(base,-arr[mask_neg]).astype(np.float32)-1
    else:
        result[mask_neg] = 0
 
    return result.astype(data_type)

'''
# Example usage:
if __name__ == "__main__":
    # Test with sample data
    test_array = np.array([[0, 1, 2, 3],
                          [4, 5, -1, 10]], dtype=np.float32)
    
    print("Original array:")
    print(test_array)
    print(f"Original dtype: {test_array.dtype}")
    
    # Apply log transform
    transformed = log_transform(test_array)
    print(f"\nLog transformed (dtype: {transformed.dtype}):")
    print(transformed)
    
    # Apply inverse transform
    restored = inverse_log_transform(transformed)
    print(f"\nInverse transformed:")
    print(restored)
    
    # Check if we get back original values (ignoring NaN entries)
    mask = ~np.isnan(transformed)
    original_valid = test_array[mask]
    restored_valid = restored[mask]
    print(f"\nMax difference for valid values: {np.max(np.abs(original_valid - restored_valid))}")
'''

def log_base(x, base=np.e):
    """
    Compute logarithm of x with arbitrary base.
    
    Parameters:
    -----------
    x : array_like
        Input values. Must be positive.
    base : float or array_like, optional
        Base of the logarithm. Default is e (natural logarithm).
        Must be positive and not equal to 1.
    
    Returns:
    --------
    ndarray
        Logarithm of x with specified base.
        
    Notes:
    ------
    Uses the change of base formula: log_base(x) = ln(x) / ln(base)
    
    Examples:
    ---------
    >>> log_base(8, 2)  # log base 2 of 8
    3.0
    >>> log_base(100, 10)  # log base 10 of 100
    2.0
    >>> log_base([1, 2, 4, 8], 2)  # log base 2 of array
    array([0., 1., 2., 3.])
    """
    x = np.asarray(x)
    base = np.asarray(base)
    
    # Input validation
    if np.any(x <= 0):
        raise ValueError("Input values must be positive")
    if np.any(base <= 0) or np.any(base == 1):
        raise ValueError("Base must be positive and not equal to 1")
    
    # Use change of base formula: log_base(x) = ln(x) / ln(base)
    return np.log(x) / np.log(base)


def log2_extended(x):
    """Logarithm base 2 (more explicit than numpy.log2)"""
    return log_base(x, 2)


def log10_extended(x):
    """Logarithm base 10 (more explicit than numpy.log10)"""
    return log_base(x, 10)

'''

# Example usage and demonstrations
if __name__ == "__main__":
    # Basic examples
    print("Basic logarithm examples:")
    print(f"log_2(8) = {log_base(8, 2)}")
    print(f"log_10(100) = {log_base(100, 10)}")
    print(f"log_3(27) = {log_base(27, 3)}")
    print(f"log_e(e^2) = {log_base(np.e**2)}")  # Natural log (default)
    
    # Array inputs
    print("\nArray examples:")
    x_vals = np.array([1, 2, 4, 8, 16])
    print(f"log_2({x_vals}) = {log_base(x_vals, 2)}")
    
    # Different bases for same input
    print("\nSame input, different bases:")
    x = 64
    bases = [2, 4, 8]
    for base in bases:
        print(f"log_{base}({x}) = {log_base(x, base)}")
    
    # Comparison with numpy's built-in functions
    print("\nComparison with numpy functions:")
    test_val = 1000
    print(f"log_base(1000, 10) = {log_base(test_val, 10)}")
    print(f"numpy.log10(1000) = {np.log10(test_val)}")
    print(f"Difference: {abs(log_base(test_val, 10) - np.log10(test_val))}")
'''


def sanitize_string(input_str):
    """
    Sanitizes a string for shell command usage by removing dangerous characters.
    Does NOT add quote marks, but removes potentially harmful characters.
    """
    if not input_str:
        return ""
    
    # Convert to string if needed
    input_str = str(input_str)
    
    # Remove non-printable characters
    printable_chars = set(string.printable)
    sanitized = ''.join(c for c in input_str if c in printable_chars)
    
    # Remove shell metacharacters
    dangerous_chars = ['&', ';', '|', '>', '<', '(', ')', '$', '`', '"', "'", '\\', '!', '*', '?', '{', '}', '[', ']', '~', '#']
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    if len(sanitized) == 0:
        sanitized = 'EMPTY'
    
    return sanitized

class BackgroundJobManager:
    def __init__(self):
        self.jobs = {}
        self.job_counter = 0
        self.results_queue = queue.Queue()
        self._start_collector()
    
    def _start_collector(self):
        """Start thread to collect finished job results"""
        collector = threading.Thread(target=self._collect_results)
        collector.daemon = True
        collector.start()
    
    def _collect_results(self):
        """Process completed jobs from queue"""
        while True:
            job_id, result = self.results_queue.get()
            print(f"Job {job_id} completed with code {result.returncode}")
            # Store or process results as needed
            self.jobs[job_id]['result'] = result
            self.jobs[job_id]['status'] = 'completed'
            self.results_queue.task_done()
    
    def _run_job(self, job_id, command, job_name):
        """Execute the job and put result in queue"""
        try:
            self.jobs[job_id]['status'] = 'running'
            job_output_name=job_name+'output.txt'
            with open(job_output_name, "w") as file:
                result = subprocess.run(command, shell=True, stdout=file, text=True)
            self.results_queue.put((job_id, result))
        except Exception as e:
            print(f"Error in job {job_id}: {e}")
            self.jobs[job_id]['status'] = 'failed'
            self.jobs[job_id]['error'] = str(e)
    
    def submit(self, command, job_name):
        """Submit a new background job"""
        job_id = self.job_counter
        self.job_counter += 1
        
        self.jobs[job_id] = {
            'command': command,
            'status': 'submitted',
            'result': None,
            'thread': None
        }
        
        thread = threading.Thread(target=self._run_job, args=(job_id, command,job_name))
        thread.daemon = True
        thread.start()
        
        self.jobs[job_id]['thread'] = thread
        return job_id
    
    def get_status(self, job_id):
        """Get status of a job"""
        if job_id not in self.jobs:
            return 'unknown job'
        return self.jobs[job_id]['status']
    
    def get_result(self, job_id):
        """Get result of a completed job"""
        if job_id not in self.jobs:
            return None
        return self.jobs[job_id].get('result')
'''
# Example usage
if __name__ == "__main__":
    manager = BackgroundJobManager()
    
    # Submit some jobs
    job1 = manager.submit(["sleep", "2"],"name1")
    job2 = manager.submit(["echo", "Hello World"],"name2")
    job3 = manager.submit(["ls", "-la"],"name3")
    
    print(f"Submitted jobs: {job1}, {job2}, {job3}")
    
    # Check status immediately
    print(f"Job 1 status: {manager.get_status(job1)}")
    print(f"Job 2 status: {manager.get_status(job2)}")
    
    # Wait a bit and check again
    time.sleep(3)
    
    # Get results
    result1 = manager.get_result(job1)
    result2 = manager.get_result(job2)
    result3 = manager.get_result(job3)
    
    print(f"Job 1 output: '{result1.stdout.strip()}', errors: '{result1.stderr.strip()}'")
    print(f"Job 2 output: '{result2.stdout.strip()}', errors: '{result2.stderr.strip()}'")
    print(f"Job 3 first 100 chars: '{result3.stdout[:100]}...'")
'''

def scale_with_saturation(arr, scale_factor, satval=65534, only_uint16=True):
    """
    Scale a uint16 or int32  array by a non-negative float, with saturation handling.
    change to int32 if satval scales above 65534
    
    Parameters:
    -----------
    arr : numpy.ndarray
        Input 2D array with uint16 or int32 data type
    scale_factor : float
        Non-negative scaling factor
    satval : int, optional
        Saturation threshold value (default 65534)
    only_uint16: bool, optional
        If true limit scaling to salval <= 65534
    
    Returns:
    --------
    numpy.ndarray
        Scaled array, converted to uint32 if scaled satval exceeds 65534
            unless only_uint16 is true
    """
    
    my_dtype = mydata_type

    # Check that scale factor is non-negative
    if scale_factor < 0:
        raise ValueError("Scale factor must be non-negative")

    # scale the satval
    new_satval = int(satval*scale_factor+0.5)

    new_arr = np.clip(arr.astype(np.int32),0,satval)
    
    # Check if any value exceeds the saturation threshold
    if new_satval > 65534:
        if only_uint16 == True:
            new_scale_factor = scale_factor*65534./new_satval
            scaled_arr = new_arr * new_scale_factor
            result = np.clip(scaled_arr.astype(np.uint16),0,65534)
            return (result, 65534)
        else:
            scaled_arr = arr * scale_factor
            # Convert to int32 to prevent overflow
            result = np.clip(scaled_arr.astype(np.int32),0,new_satval)
            if args['verbose'] == True:
                print("l_bnl_compress.py Warning: Values exceed saturation threshold. Converting to uint32.")
            return (result.astype(my_dtype), new_satval)
    else:
        scaled_arr = new_arr * scale_factor
        # Keep as my_dtype
        result = np.clip(scaled_arr.astype(np.int32),0,new_satval)
        return (result.astype(my_dtype), new_satval)



# Ultra-simplified version that relies on PIL's default behavior for most tags
def save_uint16_tiff_simple(array, output_filename, verbose=False):
    """
    Absolute minimal version that only sets the critical tags.
    
    Parameters:
    -----------
    array : 2D NumPy array (uint16)
        The image data to save as a TIFF file.
    output_filename : str
        The filename for the output TIFF file.
    verbose : bool, optional
        Whether to print information about the saved file
        
    Returns:
    --------
    str
        Path to the saved TIFF file
    """
    # Check input
    if array.dtype != np.uint16:
        array = array.astype(np.uint16)
        
    if len(array.shape) != 2:
        raise ValueError("Input array must be 2D")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
    
    # Create image from array
    img = Image.fromarray(array, mode='I;16')
    
    # Only set the absolute minimum required tags
    tiff_tags = {
        277: 1,  # SamplesPerPixel (1 = grayscale)
        284: 1,  # PlanarConfiguration (1 = CONTIG)
        258: 16, # BitsPerSample (16 for unint16)
        262: 1   # MinisBlack 
    }
    
    # Save with minimal intervention
    img.save(
        output_filename,
        format="TIFF",
        tiffinfo=tiff_tags
    )
    
    if verbose:
        print("Saved uint16 TIFF to "+output_filename)

    
    return output_filename


def load_tiff_to_numpy(tiff_filename):
    """
    Load a TIFF file into a NumPy array, preserving the original data type.
    
    Parameters:
    -----------
    tiff_filename : str
        The filename of the TIFF file to load.
    
    Returns:
    --------
    numpy.ndarray: The loaded image as a NumPy array
    """
    img = Image.open(tiff_filename)
    
    # Map PIL mode to NumPy dtype
    mode_to_dtype = {
        'L': np.uint8,
        'I;16': np.uint16,
        'I;16S': np.int16,
        'I': np.int32,
        'F': np.float32
    }
    
    # Get the image mode and convert to array with appropriate dtype
    mode = img.mode
    if mode in mode_to_dtype:
        dtype = mode_to_dtype[mode]
    else:
        dtype = None  # Let numpy decide
    
    # Convert to numpy array
    numpy_array = np.array(img, dtype=dtype)
    if args['verbose'] == True:
        print('l_bnl_compress.py: ', \
            "Loaded TIFF image from ", tiff_filename, \
            " as array with shape ", numpy_array.shape," and dtype ", numpy_array.dtype)
    
    return numpy_array

def compress_tif_to_jp2(input_file, output_file, ratios=[4000,2000,1000,500,250,125]):
    """
    Compress a TIF file to JP2 format using OpenJPEG
    
    Args:
        input_file: Path to input TIF file
        output_file: Path to output JP2 file
        quality: Compression quality (lower = more compression)
    
    Returns:
        CompletedProcess instance
    """
    cmd = "opj_compress" \
        +" -i "+ input_file \
        +" -o "+ output_file \
        +" -r "+",".join(str(element) for element in ratios)
    
    result = subprocess.run( \
        cmd, \
        capture_output=True, \
        text=True, \
        shell=True, \
        check=True  \
    )
    
    return result

def decompress_jp2_to_tif(input_file, output_file):
    """
    Decompress a JP2 file to TIF format using OpenJPEG
    
    Args:
        input_file: Path to input JP2 file
        output_file: Path to output TIF file
    
    Returns:
        CompletedProcess instance
    """
    cmd = "opj_decompress"+" " \
        +" -i "+ input_file \
        +" -o "+ output_file
    
    result = subprocess.run( \
        cmd, \
        capture_output=True, \
        text=True, \
        shell=True, \
        check=True \
    )
    
    return result



def run_bash_script_on_tiff(input_tiff, output_tiff, script_path):
    """
    Run a bash script on a TIFF file to produce a new TIFF file.
    
    Parameters:
    -----------
    input_tiff : str
        Path to the input TIFF file.
    output_tiff : str
        Path where the output TIFF file should be saved.
    script_path : str
        Path to the bash script that should be run.
    
    Returns:
    --------
    str: Path to the output TIFF file if successful
    """
    try:
        # Make the script executable if it isn't already
        os.chmod(script_path, 0o755)
        
        # Run the bash script
        cmd = [script_path, input_tiff, output_tiff]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        if args['verbose'] == True:        
            print("l_bnl_compress.py: ", \
                "Bash script executed successfully: \n",
                result.stdout)
        
        # Check if the output file was created
        if os.path.exists(output_tiff):
            return output_tiff
        else:
            raise FileNotFoundError("Output TIFF file "+output_tiff+" was not created by the script")
    
    except subprocess.CalledProcessError as e:
        print("l_bnl_compress.py: Error executing bash script: ",e)
        print("l_bnl_compress.py: Script stderr: ",e.stderr)
        raise
    except Exception as e:
        print("l_bnl_compress.py: Error: ",e)
        raise


def crat_list(mycrat):
    '''
    Convert a single target compression ratio into a list of
    layer compression ratios at 2:1 ratios declining to mycrat
    There will be no more than 10 ratios in the list.
    '''
    maxcrat = int(mycrat)
    if maxcrat < 1:
        maxcrat = 1
    resrat = [ maxcrat ]
    while maxcrat < 5001 and len(resrat) < 10:
        maxcrat = maxcrat*2
        resrat.insert(0,maxcrat)
    return resrat

def compress_HCarray(input_array, satval=32767, scale=16):
    """
    Compress a numpy int16 array using HCompress with lossy compression.
    
    Parameters:
    -----------
    input_array : numpy.ndarray
        Input array with numpy.int16 (np.int16) dtype
    satval : int, optional
        Saturation value, default is 32767 (max value for signed 16-bit)
    scale : int, optional
        Compression scale factor (higher for more compression), default is 16
        Common values are 12 or 16 for lossy compression
    
    Returns:
    --------
    compressed_data : bytes
        Compressed data as bytes
    shape : tuple
        Original shape of the array for later decompression
    """
    # Ensure input is np.int16
    if input_array.dtype != np.int16:
        input_array = input_array.astype(np.int16)
    
    # Clip values between 0 and satval
    clipped_array = np.clip(input_array, 0, satval)
    
    # Get original shape for decompression
    original_shape = clipped_array.shape
    
    # Create a compressed HDU using HCompress
    comp_hdu = fits.CompImageHDU(data=clipped_array, 
                                 compression_type='HCOMPRESS_1',
                                 hcomp_scale=scale)
    
    # Write to BytesIO object
    buffer = fits.HDUList([fits.PrimaryHDU(), comp_hdu])
    bio = BytesIO()
    buffer.writeto(bio)
    bio.seek(0)
    
    # Read the compressed data back and store the entire compressed HDU
    with fits.open(bio) as hdul:
        # Store the entire FITS file as bytes for later decompression
        bio.seek(0)  # Reset to the beginning
        fits_bytes = bio.read()
    
    # Return the entire FITS file, original shape
    return fits_bytes, original_shape

def decompress_HCarray(fits_bytes, original_shape, scale=16):
    """
    Decompress HCompressed data back to a numpy int16 array.
    
    Parameters:
    -----------
    fits_bytes : bytes
        Complete FITS file as bytes from compress_array
    original_shape : tuple
        Original shape of the array
    scale : int, optional
        Compression scale factor used for compression, default is 16
    
    Returns:
    --------
    decompressed_array : numpy.ndarray
        Decompressed array with the same shape as the original, as np.int16
    """
    # Create a BytesIO object from the FITS bytes
    bio = BytesIO(fits_bytes)
    
    # Open the FITS file from memory
    with fits.open(bio) as hdul:
        # Extract the decompressed data
        decompressed_array = hdul[1].data.copy()
    
    # Ensure output is np.int16
    if decompressed_array.dtype != np.int16:
        decompressed_array = decompressed_array.astype(np.int16)
    
    return decompressed_array


version = "1.1.4"
version_date = "30May25"
xnt=int(1)

def ntstr(xstr):
    return str(xstr)

def ntstrdt(str):
    return h5py.string_dtype(encoding='utf-8',length=len(str)+xnt)

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
        old_mask_rev=np.pad(np.asarray(old_mask,dtype='u4'),((0,ymargin),(0,xmargin)),\
        'constant',constant_values=((0,0),(0,0)))
    else:
        old_mask_rev=np.asaary(old_mask,dtype='u4')
    new_mask=np.zeros((nsy,nsx),dtype='u4')
    for iy in range(0,sy,bin_range):
        for ix in range(0,sx,bin_range):
            for iyy in range(0,bin_range):
                for ixx in range(0,bin_range):
                    if ix+ixx < sx and iy+iyy < sy:
                        if old_mask_rev[iy+iyy,ix+ixx] != 0:
                            new_mask[iy//bin_range,ix//bin_range] = new_mask[iy//bin_range,ix//bin_range]| \
                            old_mask_rev[iy+iyy,ix+ixx]
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

def conv_image_shape(old_shape,bin_range):
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

def xfer_axis_attrs(dst,src):
    if src != None:
        mykeys=src.attrs.keys()
        if 'units' in mykeys:
            dst.attrs.create('units',ntstr(src.attrs['units']),\
            dtype=ntstrdt(src.attrs['units']))
        if 'depends_on' in mykeys:
            dst.attrs.create('depends_on',ntstr(src.attrs['depends_on']),\
            dtype=ntstrdt(src.attrs['depends_on']))
        if 'transformation_type' in mykeys:
            dst.attrs.create('transformation_type',ntstr(src.attrs['transformation_type']),\
            dtype=ntstrdt(src.attrs['transformation_type']))
        if 'vector' in mykeys:
            dst.attrs.create('vector',src.attrs['vector'],\
            dtype=src.attrs['vector'].dtype)
        if 'offset' in mykeys:
            dst.attrs.create('offset',src.attrs['offset'],\
            dtype=src.attrs['offset'].dtype)



def bin_array(input_array, nbin, satval):
    """
    Bin a 2D numpy array into blocks of nbin x nbin pixels.
    
    Parameters:
    -----------
    input_array : numpy.ndarray
        2D input array to be binned
    nbin : int
        Size of the binning block (nbin x nbin)
    satval : int
        Maximum value to clip the summed block values
    
    Returns:
    --------
    numpy.ndarray
        Binned array with reduced dimensions
    """
    # Determine padded dimensions
    height, width = input_array.shape
    padded_height = int(np.ceil(height / nbin) * nbin)
    padded_width = int(np.ceil(width / nbin) * nbin)
    
    # Create padded array initialized with zeros
    padded_array = np.zeros((padded_height, padded_width), dtype=input_array.dtype)
    
    # Copy original array into padded array
    padded_array[:height, :width] = input_array
    
    # Reshape and sum with clipping
    reshaped = padded_array.reshape(
        padded_height // nbin, nbin, 
        padded_width // nbin, nbin
    )
    
    # Sum each nbin x nbin block and clip
    binned_array = np.clip(
        reshaped.sum(axis=(1, 3)), 
        0, 
        satval
    )
    
    return binned_array



def bin(old_image,bin_range,satval):
    ''' bin(old_image,bin_range,satval)

    convert an image in old_image to a returned u2 or i4  numpy array by binning
    the pixels in old_image in by summing bin_range by bin_range rectanglar 
    blocks, clipping values between 0 and satval.  If bin_range does not divide
    the original dimensions exactly, the old_image is padded with zeros.

    Because of problems with numpy and h5py, uint 4 is handled as a subset of
    i4. amd u2 is handled without np.clip
    '''
    s=old_image.shape
    if len(s) != 2:
        print('l_bnl_compress.py: invalid image shape for 2D binning')
        return None
    my_dtype = np.int32
    if mydata_float:
        my_dtype = np.float32
    new_image = old_image.astype(my_dtype)
    new_image = np.maximum(new_image,0)
    if bin_range < 2:
        new_image = new_image(0,satval)
        if args['uint'] == 2:
            return new_image.astype('np.uint2')
        return new_image
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
        new_image=np.clip(np.pad(np.asarray(new_image,dtype='u2'),((0,ymargin),(0,xmargin)),'constant',constant_values=((0,0),(0,0))),0,satval)
    else:
        new_image=new_image.clip(0,satval)
    new_image=np.round(ski.measure.block_reduce(new_image,(bin_range,bin_range),np.sum))
    new_image=new_image.clip(0,satval)
    if args['verbose'] == True:
        print('l_bnl_compress binned image of shape ', s, ' to ', mydata_type, ' binned by ', bin_range)
    return new_iamge.astype(mydata_type)

parser = argparse.ArgumentParser(description='Bin and sum images from a range')
parser.add_argument('-1','--first_image', dest='first_image', type=int, nargs='?', const=1, default=1,
   help= 'first selected image counting from 1, defaults to 1')
parser.add_argument('-b','--bin', dest='bin_range', type=int, nargs='?', const=1, default=1,
   help= 'an integer image binning range (1 ...) to apply to each selected image, defaults to 1') 
parser.add_argument('-c','--compression', dest='compression', nargs='?', const='zstd', default='zstd',
   help= 'optional compression, bslz4, bszstd,  bshuf, or zstd, defaults to zstd')
parser.add_argument('-d','--data_block_size', dest='data_block_size', type=int, nargs='?', const=100, default=100,
   help= 'data block size in images for out_file, defaults to 100')
parser.add_argument('-f','--float',dest='ufloat', type=float, nargs="?", const=2,
   help='clip the output above 0 and limit to 2 byte or 4 byte floats (positive values), or unclipped (negative values) ')
parser.add_argument('-H','--Hcompress', dest='hcomp_scale', type=int,
   help= 'Hcompress scale compression, immediately followed by decompression')
parser.add_argument('-i','--infile',dest='infile',
   help= 'the input hdf5 file to read images from')
parser.add_argument('-J','--J2K', dest='j2k_target_compression_ratio', type=int,
   help= 'JPEG-2000 target compression ratio, immediately followed by decompression')
parser.add_argument('-K','--J2K2', dest='j2k_alt_target_compression_ratio', type=int,
   help= 'JPEG-2000 alternate target compression ratio, immediately followed by decompression')
parser.add_argument('-l','--compression_level', dest='compression_level', type=int,
   help= 'optional compression level for bszstd or zstd')
parser.add_argument('-L','--logarithm', dest='logarithm_base', nargs='?', const='e',
   help= 'convert non-negative values v to log(v+1), negative values to -log(-v+1)') 
parser.add_argument('-m','--out_master',dest='out_master',
   help= 'the output hdf5 master to which to write metadata')
parser.add_argument('-N','--last_image', dest='last_image', type=int,
   help= 'last selected image counting from 1, defaults to number of images collected')
parser.add_argument('-o','--out_file',dest='out_file',default='out_data',
   help= 'the output hdf5 data file out_file_?????? with an .h5 extension are files to which to write images')
parser.add_argument('-p','-- parallel',dest='threads', default=0,
   help= 'the number of parallel threads with extra thread 0 used by itself to generate the new master file first') 
parser.add_argument('-q','--out_squash',dest='out_squash',
   help= 'the output hdf5 data file out_squash_?????? with an .h5 extension are optional files to which to write raw j2k or hcomp images')
parser.add_argument('-s','--sum', dest='sum_range', type=int, nargs='?', const=1, default=1,
   help= 'an integer image summing range (1 ...) to apply to the selected images, defaults to 1')
parser.add_argument('-S', '--scale', dest='scale_factor', type=float,default=1.,
   help= 'a non negative scaling factor to apply both images and satval, defaults to 1,')
parser.add_argument('-t','--thread',dest='thread', type=int,nargs='?',
   help= 'the thread number for the action of the current invocation of l_bnl_compress, 0 to just make a new master file, otherwise between 1 and the number of datablocks/threads')
parser.add_argument('-u','--uint', dest='uint', type=int, nargs='?', const=2, default=2,
   help= 'clip the output above 0 and limit to 2 byte or 4 byte integer (positive values), or unclipped (negative values)')
parser.add_argument('-v','--verbose',dest='verbose',action='store_true',
   help= 'provide addtional information')
parser.add_argument('-V','--version',dest='report_version',action='store_true',
   help= 'report version and version_date')
parser.add_argument('-X', '--exponential',dest='exponent', nargs='?', const= 'e',
   help= 'convert non-negative values v to exp(v)-1 and negative values to -exp(-v)+1')
args = vars(parser.parse_args())

#Sanitize file names

oout_file = args['out_file']
oout_master = args['out_master']
oout_squash = args['out_squash']

if oout_file != None:
    args['out_file'] = sanitize_string(oout_file)
    if oout_file != args['out_file'] and args['verbose'] == True:
        print("l_bnl_compress.py: sanitized out_file to ",args['out_file'])

if oout_master != None:
    args['out_master'] = sanitize_string(oout_master)
    if oout_master != args['out_master'] and args['verbose'] == True:
        print("l_bnl_compress.py: sanitized out_master to ",args['out_master'])

if oout_squash != None:
    args['out_squash'] = sanitize_string(out_squash)
    if oout_squash != args['out_squasg'] and args['verbose'] == True:
        print("l_bnl_compress.py: sanitized out_squash to ",args['out_squash'])


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

if args['threads'] != None:
    args['threads'] = int(args['threads'])
    if args['threads'] < 2:
        args['threads'] = 0
else:
    args['threads'] = 0
threads = args['threads']

if args['verbose'] == True:
    print('l_bnl_compress.py: threads: ',threads)

try:
    fin = h5py.File(args['infile'], 'r')
except:
    print('l_bnl_compress.py: infile not specified')
    sys.exit(-1)

try:
    top_definition=fin['entry']['definition']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/definition: ', top_definition)
        print('                   entry/definition[()]: ', top_definition[()])
except:
    print('l_bnl_compress.py: entry/definition not found')
    top_definition = None

try:
    detector=fin['entry']['instrument']['detector']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector: ', detector)
except:
    print('l_bnl_compress.py: detector not found')
    detector = None

try:
    description=detector['description']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/description: ', description)
        print('                   detector/description[()]: ', description[()])
except:
    print('l_bnl_compress.py: detector/description not found')
    description = None

try:
    detector_number=detector['detector_number']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector_number: ', detector_number)
        print('                   detector_number[()]: ', detector_number[()])
except:
    print('l_bnl_compress.py: detector/detector_number not found')
    detector_number=None

try:
    depends_on=detector['depends_on']
    if args['verbose'] == True:
        print('l_bnl_compress.py: depends_on: ', depends_on)
        print('                   depends_on[()]: ', depends_on[()])
except:
    print('l_bnl_compress.py: detector/depends_on not found')
    depends_on=None

try:
    bit_depth_image=detector['bit_depth_image']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/bit_depth_image: ', bit_depth_image)
        print('                   detector/bit_depth_image[()]: ', bit_depth_image[()])
except:
    print('l_bnl_compress.py: detector/bit_depth_image not found')
    bit_depth_image=None

try:
    bit_depth_readout=detector['bit_depth_readout']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/bit_depth_readout: ', bit_depth_readout)
        print('                   detector/bit_depth_readout[()]: ', bit_depth_readout[()])
except:
    print('l_bnl_compress.py: detector/bit_depth_readout not found')
    bit_depth_readout = None

try:
    thickness=detector['sensor_thickness']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/sensor_thickness: ', thickness)
        print('                   detector/sensor_thickness[()]: ', thickness[()])
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

satval = 32767
satval_not_found = True
try:
    saturation_value = detector['saturation_value']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/saturation_value: ', saturation_value)
        print('                   detector/saturation_value[()]: ', saturation_value[()])
    satval = saturation_value[()]
    satval_not_found = False 
except:
    print('l_bnl_compress.py: detector/saturation_value not found')
    saturation_value = None

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

nimages=int(xnimages[()])
ntrigger=int(xntrigger[()])
if args['verbose'] == True:     
    print('nimages: ',nimages)
    print('ntrigger: ',ntrigger)
if nimages == 1:   
    nimages = ntrigger
    print('l_bnl_compress.py: warning: settng nimages to ',nimages,' from ntrigger')
if args['last_image'] == None:
    args['last_image'] =  nimages 

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
    countrate_correction_count_cutoff = detectorSpecific['countrate_correction_count_cutoff']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/countrate_correction_count_cutoff: ', countrate_correction_count_cutoff)
        print('                   detectorSpecific/countrate_correction_count_cutoff[()]: ', countrate_correction_count_cutoff[()])
    if satval_not_found:
        satval = countrate_correction_count_cutoff[()]
        satval_not_found = False 

except:
    print('l_bnl_compress.py: detectorSpecific/countrate_correction_count_cutoff not found')
    countrate_correction_count_cutoff = None

try:
    dS_saturation_value = detectorSpecific['saturation_value']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/saturation_value: ', dS_saturation_value)
        print('                   detectorSpecific/saturation_value[()]: ', dS_saturation_value[()])
    if satval_not_found:
        satval = dS_saturation_value[()]
        satval_not_found = False 
except:
     print('l_bnl_compress.py: detectorSpecific/saturation_value not found')
     dS_saturation_value = None  

try:
    mod0_countrate_cutoff = detectorSpecific['detectorModule_000']['countrate_correction_count_cutoff']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/detectorModule_000/countrate_correction_count_cutoff: ',mod0_countrate_cutoff)
        print('                   detectorSpecific/detectorModule_000countrate_correction_count_cutoff[()]: ',  mod0_countrate_cutoff[()])
        print('                  *** use of this dataset is deprecated ***')
    if satval_not_found:
        satval = mod0_countrate_cutoff[()]
        satval_not_found = False
except:
    print('l_bnl_compress.py: detectorSpecific/detectorModule_000/countrate_correction_count_cutoff not found')
    if satval_not_found:
        print('l_bnl_compress.py: ...count_cutoff not found, using 32765')
        satval=32765
    mod0_countrate_cutoff = None

dS_pixel_mask = None
try:
    dS_pixel_mask=detectorSpecific['pixel_mask']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/pixel_mask: ', dS_pixel_mask)
        print('                   detectorSpecific/pixel_mask[()]: ', dS_pixel_mask[()])
except:
    print('l_bnl_compress.py: detectorSpecific/pixel_mask not found')
    dS_pixel_mask=None

det_gon = None
det_gon_two_theta = None
det_gon_two_theta_end = None
det_gon_two_theta_range_average = None
det_gon_two_theta_range_total = None
try:
    det_gon = detector['goniometer']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/goniometer: ', det_gon)
        print('                   detector/goniometer[()]: ', det_gon[()])
    try:
        det_gon_two_theta = detector['goniometer']['two_theta']
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/goniometer/two_theta: ', det_gon_two_theta)
            print('                   detector/goniometer/two_theta[()]: ', det_gon_two_theta[()])
    except:
        print('l_bnl_compress.py: detector/goniometer/two_theta not found')
        det_gon_translation = None
    try:
        det_gon_two_theta_end = detector['goniometer']['two_theta_end']
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/goniometer/two_theta_end: ', det_gon_two_theta_end)
            print('                   detector/goniometer/two_theta_end[()]: ', det_gon_two_theta_end[()])
    except:
        print('l_bnl_compress.py: detector/goniometer/two_theta_end not found')
        det_gon_two_theta_end = None
    try:
        det_gon_two_theta_range_average = detector['goniometer']['two_theta_range_average']
        if det_gon_two_theta_range_average.shape == ():
            det_gon_two_theta_range_average = None
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/goniometer/two_theta_range_average: ',\
            det_gon_two_theta_range_average)
            print('                   detector/goniometer/two_theta_range_average[()]: ',\
            det_gon_two_theta_range_average[()])
    except:
        print('l_bnl_compress.py: detector/goniometer/two_theta_range_average not found')
        det_gon_range_average = None
    try:
        det_gon_two_theta_range_total = detector['goniometer']['two_theta_range_total']
        if det_gon_two_theta_range_average.shape == ():
            det_gon_two_theta_range_average.shape = Nonw
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/goniometer/two_theta_range_total: ',\
            det_gon_two_theta_range_total)
            print('                   detector/goniometer/two_theta_range_total[()]: ',\
            det_gon_two_theta_range_total[()])
    except:
        print('l_bnl_compress.py: detector/goniometer/two_theta_range_total not found')
        det_gon_two_theta_range_total = None
except:
    print('l_bnl_compress.py: detector/goniometer not found')
    det_gon = None



det_nxt = None
det_nxt_transation = None
det_nxt_two_theta = None
det_nxt_two_theta_end = None
det_nxt_two_theta_range_average = None
det_nxt_two_theta_range_total = None
try:
    det_nxt = detector['transformations']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/transformations: ', det_nxt)
        print('                   detector/transformations[()]: ', det_nxt[()])
    try:
        det_nxt_translation = detector['transformations']['translation']
        if det_nxt_translation.shape == ():
            det_nxt_translation = None
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/transformations/translation: ', det_nxt_translation)
            print('                   detector/transformations/translation[()]: ', det_nxt_translation[()])
    except:
        print('l_bnl_compress.py: detector/transformations/translation not found')
        det_nxt_translation = None
    try:
        det_nxt_two_theta = detector['transformations']['two_theta']
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/transformations/two_theta: ', det_nxt_two_theta)
            print('                   detector/transformations/two_theta[()]: ', det_nxt_two_theta[()])
    except:
        print('l_bnl_compress.py: detector/transformations/two_theta not found')
        det_nxt_translation = None
    try:
        det_nxt_two_theta_end = detector['transformations']['two_theta_end']
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/transformations/two_theta_end: ', det_nxt_two_theta_end)
            print('                   detector/transformations/two_theta_end[()]: ', det_nxt_two_theta_end[()])
    except:
        print('l_bnl_compress.py: detector/transformations/two_theta_end not found')
        det_nxt_two_theta_end = None
    try:
        det_nxt_two_theta_range_average = detector['transformations']['two_theta_range_average']
        if det_nxt_two_theta_range_average.shape == ():
            det_nxt_two_theta_range_average = None
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/transformations/two_theta_range_average: ',\
            det_nxt_two_theta_range_average)
            print('                   detector/transformations/two_theta_range_average[()]: ',\
            det_nxt_two_theta_range_average[()])
    except:
        print('l_bnl_compress.py: detector/transformations/two_theta_range_average not found')
        det_nxt_range_average = None
    try:
        det_nxt_two_theta_range_total = detector['transformations']['two_theta_range_total']
        if det_nxt_two_theta_range_total.shape == ():
            det_nxt_two_theta_range_total = None
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/transformations/two_theta_range_total: ',\
            det_nxt_two_theta_range_total)
            print('                   detector/transformations/two_theta_range_total[()]: ',\
            det_nxt_two_theta_range_total[()])
    except:
        print('l_bnl_compress.py: detector/transformations/two_theta_range_total not found')
        det_nxt_two_theta_range_total = None
except:
    print('l_bnl_compress.py: detector/transformations not found')
    det_nxt = None

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


# find {chi, kappa, omega, phi, translation} in entry/sample/transformations/*
try:
    nxt_chi=fin['entry']['sample']['transformations']['chi']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/chi: ', nxt_chi)
        print('                   entry/sample/transformations/chi[()]: ', nxt_chi[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/chi not found')
    nxt_chi = None

try:
    nxt_kappa=fin['entry']['sample']['transformations']['kappa']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/kappa: ', nxt_kappa)
        print('                   entry/sample/transformations/kappa[()]: ', nxt_kappa[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/kappa not found')
    nxt_kappa = None

try:
    nxt_omega=fin['entry']['sample']['transformations']['omega']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/omega: ', nxt_omega)
        print('                   entry/sample/transformations/omega[()]: ', nxt_omega[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/omega not found')
    nxt_omega = None

try:
    nxt_phi=fin['entry']['sample']['transformations']['phi']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/phi: ', nxt_phi)
        print('                   entry/sample/transformations/phi[()]: ', nxt_phi[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/phi not found')
    nxt_phi = None

try:
    nxt_translation=fin['entry']['sample']['transformations']['translation']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/translation: ', nxt_translation)
        print('                   entry/sample/transformations/translation[()]: ', nxt_translation[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/translation not found')
    nxt_translation = None

# find {chi_end, kappa_end, omega_end, phi_end} in entry/sample/transformations/*
try:
    nxt_chi_end=fin['entry']['sample']['transformations']['chi_end']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/chi_end: ', nxt_chi_end)
        print('                   entry/sample/transformations/chi_end[()]: ', nxt_chi_end[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/chi_end not found')
    nxt_chi_end = None

try:
    nxt_kappa_end=fin['entry']['sample']['transformations']['kappa_end']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/kappa_end: ', nxt_kappa_end)
        print('                   entry/sample/transformations/kappa_end[()]: ', nxt_kappa_end[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/kappa_end not found')
    nxt_kappa_end = None

try:
    nxt_omega_end=fin['entry']['sample']['transformations']['omega_end']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/omega_end: ', nxt_omega_end)
        print('                   entry/sample/transformations/omega_end[()]: ', nxt_omega_end[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/omega_end not found')
    nxt_omega_end = None

try:
    nxt_phi_end=fin['entry']['sample']['transformations']['phi_end']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/phi_end: ', nxt_phi_end)
        print('                   entry/sample/transformations/phi_end[()]: ', nxt_phi_end[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/phi_end not found')
    nxt_phi_end = None


# find {chi_range_average, kappa_range_average, omega_range_average, phi_range_average} 
#     in entry/sample/transformations/*_end
try:
    nxt_chi_range_average=fin['entry']['sample']['transformations']['chi_range_average']
    if nxt_chi_range_average.shape == ():
        nxt_chi_range_average = None
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/chi_range_average: ', \
            nxt_chi_range_average)
        print('                   entry/sample/transformations/chi_range_average[()]: ', \
            nxt_chi_range_average[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/chi_range_average not found')
    nxt_chi_range_average = None

try:
    nxt_kappa_range_average=fin['entry']['sample']['transformations']['kappa_range_average']
    if nxt_kappa_range_average.shape == ():
        nxt_kappa_range_average = Mome
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/kappa_range_average: ', \
            nxt_kappa_range_average)
        print('                   entry/sample/transformations/kappa_range_average[()]: ', \
            nxt_kappa_range_average[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/kappa_range_average not found')
    nxt_kappa_range_average = None

try:
    nxt_omega_range_average=fin['entry']['sample']['transformations']['omega_range_average']
    if nxt_omega_range_average.shape == ():
        nxt_omega_range_average = None
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/omega_range_average: ', \
            nxt_omega_range_average)
        print('                   entry/sample/transformations/omega_range_average[()]: ', \
            nxt_omega_range_average[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/omega_range_average not found')
    nxt_omega_range_average = None

try:
    nxt_phi_range_average=fin['entry']['sample']['transformations']['phi_range_average']
    if nxt_phi_range_average.shape == ():
        nxt_phi_range_average.shape = Nonw
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/phi_range_average: ', \
            nxt_phi_range_average)
        print('                   entry/sample/transformations/phi_range_average[()]: ', \
            nxt_phi_range_average[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/phi_range_average not found')
    nxt_phi_range_average = None


# find {chi_range_total, kappa_range_total, omega_range_total, phi_range_total} 
#    in entry/sample/transformations/*_range_total
try:
    nxt_chi_range_total=fin['entry']['sample']['transformations']['chi_range_total']
    if nxt_chi_range_total.shape == ():
        nxt_chi_range_total = None
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/chi_range_total: ', \
            nxt_chi_range_total)
        print('                   entry/sample/transformations/chi_range_total[()]: ', \
            nxt_chi_range_total[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/chi_range_total not found')
    nxt_chi_range_total = None

try:
    nxt_kappa_range_total=fin['entry']['sample']['transformations']['kappa_range_total']
    if nxt_kappa_range_total.shape == ():
        nxt_kappa_range_total = None
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/kappa_range_total: ', \
            nxt_kappa_range_total)
        print('                   entry/sample/transformations/kappa_range_total[()]: ', \
            nxt_kappa_range_total[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/kappa_range_total not found')
    nxt_kappa_range_total = None

try:
    nxt_omega_range_total=fin['entry']['sample']['transformations']['omega_range_total']
    if nxt_omega_range_total.shape == ():
        nxt_omega_range_total = None
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/omega_range_total: ', \
            nxt_omega_range_total)
        print('                   entry/sample/transformations/omega_range_total[()]: ', \
            nxt_omega_range_total[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/omega_range_total not found')
    nxt_omega_range_total = None

try:
    nxt_phi_range_total=fin['entry']['sample']['transformations']['phi_range_total']
    if nxt_phi_range_total.shape == ():
        nxt_phi_range_total = None
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/phi_range_total: ', \
            nxt_phi_range_total)
        print('                   entry/sample/transformations/phi_range_total[()]: ', \
            nxt_phi_range_total[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/phi_range_total not found')
    nxt_phi_range_total = None

samp_gon = None
chi = None
chi_end = None
chi_range_average = None
chi_range_total = None
kappa = None
kappa_end = None
kappa_range_average = None
kappa_range_total = None
angles = None
angles_end = None
osc_width = None
osc_total = None
phi = None
phi_end = None
phi_range_average = None
phi_range_total = None
try:
    samp_gon = fin['entry']['sample']['goniometer']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/goniometer: ', samp_gon)
except:
    print('l_bnl_compress.py: entry/sample/goniometer not found')
    samp_gon = None
if samp_gon != None:
    try:
        chi=samp_gon['chi']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/chi: ', chi)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/chi not found')
        chi=None
    try:
        chi_end=samp_gon['chi_end']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/chi_end: ', chi_end)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/chi_end not found')
        chi_end=None
    try:
        chi_range_average = samp_gon['chi_range_average']
        if chi_range_average.shape == ():
            chi_range_average = None
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/chi_range_average: ', \
                chi_range_average)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer/chi_range_average not found')
        chi_range_average=None
    try:
        chi_range_total = samp_gon['chi_range_total']
        if chi_range_total.shape == ():
            chi_range_total = None
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/chi_range_total: ', \
                chi_range_total)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer/chi_range_total not found')
        chi_range_total=None
    try:
        kappa=samp_gon['kappa']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/kappa: ', kappa)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/kappa not found')
        kappa=None
    try:
        kappa_end=samp_gon['kappa_end']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/kappa_end: ', kappa_end)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/kappa_end not found')
        kappa_end=None
    try:
        kappa_range_average = samp_gon['kappa_range_average']
        if kappa_range_average.shape == ():
            kappa_range_average = None
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/kappa_range_average: ', \
                kappa_range_average)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer/kappa_range_average not found')
        kappa_range_average=None
    try:
        kappa_range_total = samp_gon['kappa_range_total']
        if kappa_range_total.shape == ():
            kappa_range_total = None
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/kappa_range_total: ', \
                kappa_range_total)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer/kappa_range_total not found')
        kappa_range_total=None
    try:
        angles=samp_gon['omega']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/omega: ', angles)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/omega not found')
        angles=None
    try:
        angles_end=samp_gon['omega_end']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/omega_end: ', angles_end)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/omega_end not found')
        angles_end=None
    try: 
        osc_width = samp_gon['omega_range_average']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/omega_range_average: ', osc_width)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer not found')
        osc_width = None
    try:
        osc_total = samp_gon['omega_range_total']
        if osc_total.shape == ():
            osc_total = None
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/omega_range_total: ', osc_total)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer/omega_range_total not found')
        osc_total = None
    try:
        phi=samp_gon['phi']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/phi: ', phi)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/phi not found')
        phi=None
    try:
        phi_end=samp_gon['phi_end']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/phi_end: ', phi_end)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/phi_end not found')
        phi_end=None
    try:
        phi_range_average = samp_gon['phi_range_average']
        if phi_range_average.shape == ():
            phi_range_average = None
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/phi_range_average: ', \
                phi_range_average)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer/phi_range_average not found')
        phi_range_average=None
    try:
        phi_range_total = samp_gon['phi_range_total']
        if phi_range_total.shape == ():
            phi_range_total = None
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/phi_range_total: ', \
                phi_range_total)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer/phi_range_total not found')
        phi_range_total=None

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

if args['thread'] == 0 or args['thread'] ==None:
    thread = 0
else:
    thread = int(args['thread'])
args['thread'] = str(thread)

new_images_buffer=args['out_file']+'_'+args['thread']+"_new_images.h5"
new_images_buffer_manager= \
     HDF5DatasetBuffer(new_images_buffer, max_cache_size=50, max_memory_mb=1024)

print("args['first_image']",args['first_image'])
print("(args['last_image'])+1",(args['last_image'])+1)
print("args['sum_range']",args['sum_range'])
for image in range(args['first_image'],(args['last_image'])+1,args['sum_range']):
    lim_image=image+int(args['sum_range'])
    if lim_image > args['last_image']+1:
        lim_image = args['last_image']+1
    if args['verbose']==True and args['sum_range'] > 1 :
        print('Adding images from ',image,' to ',lim_image)
    prev_out=None
    for cur_image in range(image,lim_image):
        if args['verbose']==True:
            print('image, (block,offset): ',cur_image,\
              conv_image_to_block_offset(cur_image,number_per_block))
        cur_source=conv_image_to_block_offset(cur_image,number_per_block)
        cur_source_img_block='data_'+str(cur_source[0]+block_start).zfill(6)
        cur_source_img_imgno=cur_source[1]
        cur_source=datagroup[cur_source_img_block][cur_source_img_imgno,:,:]
        cur_source[(cur_source  < 0) | (cur_source > 65530)]=0
        if args['verbose']==True:
            print('image input shape ',cur_source.shape)
        print('cur_source_img_block: ', cur_source_img_block)
        print('cur_source_img_imgno: ', cur_source_img_imgno)
        if args['bin_range'] > 1:
            cur_source=bin_array(cur_source,int(args['bin_range']),satval)
        if cur_image > image:
            prev_out = np.clip(prev_out+cur_source,0,satval)
        else:
            prev_out = np.clip(np.asarray(cur_source,dtype='i2'),0,satval)
        del cur_source
    new_nimage = new_nimage+1
    new_images_buffer_manager.create_dataset(str(new_nimage), prev_out)
    del prev_out


if (args['data_block_size'] == None) or (args['data_block_size'] < 2):
    args['data_block_size'] = 1
out_number_per_block = args['data_block_size']
out_number_of_blocks = int(new_nimage+out_number_per_block-1)//out_number_per_block
if args['threads'] > out_number_of_blocks:
    args['threads'] =  out_number_of_blocks
out_max_image=new_nimage
if args['verbose'] == True:
    print('out_number_per_block: ', out_number_per_block)
    print('out_number_of_blocks: ', out_number_of_blocks)
    print('number of threads:    ', args['threads'])
fout={}
if args['out_squash'] != None:
    fout_squash={}

# create the master file
master=0
if args['out_master']==None or args['out_master']=='out_file':
    args['out_master']=args['out_file']+"_master"
if args['out_squash']=='out_file':
    args['out_squash']=args['out_file']+"_squash"
'''
  Cases for master files and data files
    threads == 0 or theads == None
       There is no parallelism and a single thread will
       write both the master file ad all data files
    threads > 0
       Thread 0 will write the master file.
       All other threads (1...) will read the master file and
       write data files in parallel
'''
if args['threads'] == 0 or args['threads'] == None:
    threads = 0
else:
    threads = int(args['threads'])
if threads <= 0 or thread ==0:
    write_master=True
    fout[master] = h5py.File(args['out_master']+".h5",'w')
    fout[master].attrs.create('default',ntstr('entry'),dtype=ntstrdt('entry'))
    fout[master].create_group('entry') 
    fout[master]['entry'].attrs.create('NX_class',ntstr('NXentry'),dtype=ntstrdt('NXentry'))
    fout[master]['entry'].attrs.create('default',ntstr('data'),dtype=ntstrdt('data'))
    fout[master]['entry'].create_group('data') 
    fout[master]['entry']['data'].attrs.create('NX_class',ntstr('NXdata'),dtype=ntstrdt('NXdata'))
    fout[master]['entry']['data'].attrs.create('signal',ntstr('data_000001'),\
        dtype=ntstrdt('data_000001'))
    if top_definition != None:
        fout[master]['entry'].create_dataset('definition',shape=top_definition.shape,\
            dtype=top_definition.dtype)
        fout[master]['entry']['definition'][()]=top_definition[()]
        if 'version' in top_definition.attrs.keys():
            fout[master]['entry']['definition'].attrs.create('version',\
            top_definition.attrs['version'])
    fout[master]['entry'].create_group('instrument') 
    fout[master]['entry']['instrument'].attrs.create('NX_class',ntstr('NXinstrument'),dtype=ntstrdt('NXinstrument'))
    fout[master]['entry'].create_group('sample') 
    fout[master]['entry']['sample'].attrs.create('NX_class',ntstr('NXsample'),dtype=ntstrdt('NXsample')) 
    fout[master]['entry']['sample'].create_group('goniometer') 
    fout[master]['entry']['sample']['goniometer'].attrs.create('NX_class',ntstr('NXtransformations'), \
        dtype=ntstrdt('NXtransformations'))
    fout[master]['entry']['sample'].create_group('transformations') 
    fout[master]['entry']['sample']['transformations'].attrs.create('NX_class',ntstr('NXtransformations'),\
        dtype=ntstrdt('NXtransformations'))  
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
    if depends_on != None:
        fout[master]['entry']['instrument']['detector'].create_dataset(\
            'depends_on',shape=depends_on.shape,dtype=depends_on.dtype)
        fout[master]['entry']['instrument']['detector']['depends_on'][()]=\
            depends_on[()]
    if bit_depth_image != None:
        fout[master]['entry']['instrument']['detector'].create_dataset(\
            'bit_depth_image',shape=bit_depth_image.shape,dtype='u4')
        fout[master]['entry']['instrument']['detector']['bit_depth_image'][()]=\
            np.uint32(16)
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
        fout[master]['entry']['instrument']['detector']['countrate_correction_applied'][()]=\
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
        'units',ntstr(frame_time.attrs['units']),\
        dtype=ntstrdt(frame_time.attrs['units']))
    fout[master]['entry']['instrument']['detector'].create_dataset(\
        'x_pixel_size',shape=pixelsizex.shape,dtype=pixelsizex.dtype)
    fout[master]['entry']['instrument']['detector']['x_pixel_size'][()]=\
        pixelsizex[()]*int(args['sum_range'])
    fout[master]['entry']['instrument']['detector']['x_pixel_size'].attrs.create(\
        'units',ntstr(pixelsizex.attrs['units']),\
        dtype=ntstrdt(pixelsizex.attrs['units']))
    fout[master]['entry']['instrument']['detector'].create_dataset(\
        'y_pixel_size',shape=pixelsizey.shape,dtype=pixelsizey.dtype)
    fout[master]['entry']['instrument']['detector']['y_pixel_size'][()]=\
        pixelsizey[()]*int(args['sum_range'])
    fout[master]['entry']['instrument']['detector']['y_pixel_size'].attrs.create(\
        'units',ntstr(pixelsizey.attrs['units']),\
        dtype=ntstrdt(pixelsizey.attrs['units']))
    if pixel_mask!=None:
        new_pixel_mask=conv_pixel_mask(pixel_mask,int(args['bin_range']))
        fout[master]['entry']['instrument']['detector'].create_dataset(\
            'pixel_mask',shape=new_pixel_mask.shape,dtype='u4',\
            data=new_pixel_mask,\
            **hdf5plugin.Bitshuffle(nelems=0,cname='lz4'))
        del new_pixel_mask
    fout[master]['entry']['instrument']['detector'].create_group(\
        'detectorSpecific')
    fout[master]['entry']['instrument']['detector']['detectorSpecific'].attrs.create(\
        'NX_class',ntstr('NXcollection'))
    new_shape=conv_image_shape((int(ypixels[()]),int(xpixels[()])),int(args['bin_range']))
    fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
        'auto_summation',data=1,dtype='i1')
    print('compression: ',args['compression'])
    fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
        'compression',data=args['compression'])
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
    if satval_not_found:
        fout[master]['entry']['instrument']['detector']['detectorSpecific']['saturation_value'][()]=\
        satval
        fout[master]['entry']['instrument']['detector']['detectorSpecific']['countrate_correction_count_cutoff'][()]=\
        satval
    if saturation_value != None:
        fout[master]['entry']['instrument']['detector'].create_dataset(\
        'saturation_value',shape=saturation_value.shape,dtype=saturation_value.dtype)
        fout[master]['entry']['instrument']['detector']['saturation_value'][()]=saturation_value[()]
    if countrate_correction_count_cutoff != None:
        fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
        'countrate_correction_count_cutoff',shape=countrate_correction_count_cutoff.shape,\
        dtype=countrate_correction_count_cutoff.dtype)
        fout[master]['entry']['instrument']['detector']['detectorSpecific']['countrate_correction_count_cutoff'][()]=\
            countrate_correction_count_cutoff[()]
    if dS_saturation_value != None:
        fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
        'saturation_value',shape=dS_saturation_value.shape,dtype=dS_saturation_value.dtype)
        fout[master]['entry']['instrument']['detector']['detectorSpecific']['saturation_value'][()]=\
        dSsaturation_value[()]
    if det_gon != None:
        fout[master]['entry']['instrument']['detector'].create_group('goniometer')
        fout[master]['entry']['instrument']['detector']['goniometer'].attrs.create('NX_class',\
        ntstr('NXgoniometer'),dtype=ntstrdt('NXgoniometer'))
        if det_gon_two_theta != None:
            newshape = det_gon_two_theta.shape
            if newshape[0]==nimages:
                newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
            fout[master]['entry']['instrument']['detector']['goniometer'].create_dataset(\
            'two_theta',shape=newshape,dtype=det_gon_two_theta.dtype)
            fout[master]['entry']['instrument']['detector']['goniometer']['two_theta'][0:(int(args['last_image'])-\
            int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
            det_gon_two_theta[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
            xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['goniometer']['two_theta'],\
            det_gon_two_theta)
        if det_gon_two_theta_end != None:
            newshape = det_gon_two_theta_end.shape
            if newshape[0]==nimages:
                newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
            fout[master]['entry']['instrument']['detector']['goniometer'].create_dataset(\
            'two_theta_end',shape=newshape,dtype=det_gon_two_theta_end.dtype)
            fout[master]['entry']['instrument']['detector']['goniometer']['two_theta_end'][0:(int(args['last_image'])-\
            int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
            det_gon_two_theta_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
            xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['goniometer']['two_theta_end'],\
            det_gon_two_theta_end)
        if det_gon_two_theta_range_average != None:
            newshape = det_gon_two_theta_range_average.shape
            if newshape[0]==nimages:
                newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
            fout[master]['entry']['instrument']['detector']['goniometer'].create_dataset(\
            'two_theta_range_average',shape=newshape,dtype=det_gon_two_theta_range_average.dtype)
            fout[master]['entry']['instrument']['detector']['goniometer']['two_theta_range_average'][()]=\
            det_gon_two_theta_range_average[()]*int(args['sum_range'])
            xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['goniometer']['two_theta_range_average'],\
            det_gon_two_theta_range_average)
        if det_gon_two_theta_range_total != None:
            newshape = det_gon_two_theta_range_total.shape
            if newshape[0]==nimages:
                newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
            fout[master]['entry']['instrument']['detector']['goniometer'].create_dataset(\
            'two_theta_range_total',shape=newshape,dtype=det_gon_two_theta_range_total.dtype)
            fout[master]['entry']['instrument']['detector']['goniometer']['two_theta_range_total'][()]=\
            det_gon_two_theta_range_total[()]
            xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['goniometer']['two_theta_range_total'],\
            det_gon_two_theta_range_total)
    if det_nxt != None:
        fout[master]['entry']['instrument']['detector'].create_group('transformations')
        fout[master]['entry']['instrument']['detector']['transformations'].attrs.create('NX_class',\
        ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        if det_nxt_translation != None:
            newshape = det_nxt_translation.shape
            if newshape[0]==nimages:
                newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
            fout[master]['entry']['instrument']['detector']['transformations'].create_dataset(\
            'translation',shape=newshape,dtype=det_nxt_translation.dtype)
            fout[master]['entry']['instrument']['detector']['transformations']['translation'][()]=\
            det_nxt_translation[()]
            xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['transformations']['translation'],\
            det_nxt_translation)
        if det_nxt_two_theta != None:
            fout[master]['entry']['instrument']['detector']['transformations'].create_dataset(\
            'two_theta',shape=det_nxt_two_theta.shape,dtype=det_nxt_two_theta.dtype)
            fout[master]['entry']['instrument']['detector']['transformations']['two_theta'][0:(int(args['last_image'])-\
            int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
            det_nxt_two_theta[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
            xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['transformations']['two_theta'],\
            det_nxt_two_theta)
        if det_nxt_two_theta_end != None:
            fout[master]['entry']['instrument']['detector']['transformations'].create_dataset(\
            'two_theta_end',shape=det_nxt_two_theta_end.shape,dtype=det_nxt_two_theta_end.dtype)
            fout[master]['entry']['instrument']['detector']['transformations']['two_theta_end'][0:(int(args['last_image'])-\
            int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
            det_nxt_two_theta_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
            xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['transformations']['two_theta_end'],\
            det_nxt_two_theta_end)
        if det_nxt_two_theta_range_average != None:
            fout[master]['entry']['instrument']['detector']['transformations'].create_dataset(\
            'two_theta_range_average',shape=det_nxt_two_theta_range_average.shape,dtype=det_nxt_two_theta_range_average.dtype)
            fout[master]['entry']['instrument']['detector']['transformations']['two_theta_range_average'][()]=\
            det_nxt_two_theta_range_average[()]*int(args['sum_range'])
            xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['transformations']['two_theta_range_average'],\
            det_nxt_two_theta_range_average)
        if det_nxt_two_theta_range_total != None:
            fout[master]['entry']['instrument']['detector']['transformations'].create_dataset(\
            'two_theta_range_total',shape=det_nxt_two_theta_range_total.shape,dtype=det_nxt_two_theta_range_total.dtype)
            fout[master]['entry']['instrument']['detector']['transformations']['two_theta_range_total'][()]=\
            det_nxt_two_theta_range_total[()]
            xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['transformations']['two_theta_range_total'],\
            det_nxt_two_theta_range_total)
    if mod0_countrate_cutoff != None:
        if not ('detectorModule_000' in fout[master]['entry']['instrument']['detector']['detectorSpecific'].keys()):
            fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_group('detectorModule_000')
        fout[master]['entry']['instrument']['detector']['detectorSpecific']['detectorModule_000'].create_dataset(\
            'countrate_correction_count_cutoff',shape=mod0_countrate_cutoff.shape,dtype=mod0_countrate_cutoff.dtype)
        fout[master]['entry']['instrument']['detector']['detectorSpecific']['detectorModule_000']\
            ['countrate_correction_count_cutoff'][()]=mod0_countrate_cutoff[()]
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
        if 'units' in sample_wavelength.attrs.keys():
            fout[master]['entry']['sample']['beam']['incident_wavelength'].attrs.create('units',\
                sample_wavelength.attrs['units'])
    if instrument_wavelength!=None:
        if not ('beam' in fout[master]['entry']['instrument'].keys()):
            fout[master]['entry']['instrument'].create_group('beam')
            fout[master]['entry']['instrument']['beam'].attrs.create('NX_class',ntstr('NXbeam'),dtype=ntstrdt('NXbeam'))
        if not ('wavelength' in fout[master]['entry']['instrument']['beam'].keys()):
            fout[master]['entry']['instrument']['beam'].create_dataset(\
            'wavelength',shape=instrument_wavelength.shape,dtype=instrument_wavelength.dtype)
        fout[master]['entry']['instrument']['beam']['wavelength'][()]=instrument_wavelength[()]
        if 'units' in instrument_wavelength.attrs.keys():
            fout[master]['entry']['instrument']['beam']['wavelength'].attrs.create('units',\
                instrument_wavelength.attrs['units'])
    if monochromater_wavelength!=None:
        if not ('monochromater' in fout[master]['entry']['instrument'].keys()):
            fout[master]['entry']['instrument'].create_group('monochromater')
            fout[master]['entry']['instrument']['monochromater'].attrs.create('NX_class', \
                ntstr('NXmonochromater'),dtype=ntstrdt('NXmonochromater'))
        if not ('wavelength' in fout[master]['entry']['instrument']['monochromater'].keys()):
            fout[master]['entry']['instrument']['monochromater'].create_dataset(\
                'wavelength',shape=monochromater_wavelength.shape,dtype=monochromater_wavelength.dtype)
        fout[master]['entry']['instrument']['monochromater']['wavelength'][()]=monochromater_wavelength[()]
        if 'units' in monochromater_wavelength.attrs.keys():
            fout[master]['entry']['instrument']['monochromater']['wavelength'].attrs.create('units',\
                monochromater_wavelength.attrs['units'])
    if beam_incident_wavelength!=None:
        if not ('beam' in fout[master]['entry']['instrument'].keys()):
            fout[master]['entry']['instrument'].create_group('beam')
            fout[master]['entry']['instrument']['beam'].attrs.create('NX_class',ntstr('NXbeam'),dtype=ntstrdt('NXbeam'))
        if not ('incident_wavelength' in fout[master]['entry']['instrument']['beam'].keys()):
            fout[master]['entry']['instrument']['beam'].create_dataset(\
            'incident_wavelength',shape=beam_incident_wavelength.shape,dtype=beam_incident_wavelength.dtype)
        fout[master]['entry']['instrument']['beam']['incident_wavelength'][()]=beam_incident_wavelength[()]
        if 'units' in beam_incident_wavelength.attrs.keys():
            fout[master]['entry']['instrument']['beam']['incident_wavelength'].attrs.create('units',\
                beam_incident_wavelength.attrs['units'])
    if chi!=None:
        newshape = chi.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'chi',shape=newshape,dtype=chi.dtype) 
        fout[master]['entry']['sample']['goniometer']['chi'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        chi[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        fout[master]['entry']['sample']['goniometer']['chi'].attrs.create(\
            'units',ntstr(chi.attrs['units']),\
            dtype=ntstrdt(chi.attrs['units']))
    if chi_end!=None:
        newshape = chi_end.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'chi_end',shape=newshape,dtype=chi_end.dtype)
        fout[master]['entry']['sample']['goniometer']['chi_end'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        chi_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        fout[master]['entry']['sample']['goniometer']['chi_end'].attrs.create(\
            'units',ntstr(chi_end.attrs['units']),\
            dtype=ntstrdt(chi_end.attrs['units']))
    if chi_range_average != None:
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'chi_range_average',shape=chi_range_average.shape,dtype=chi_range_average.dtype)
        fout[master]['entry']['sample']['goniometer']['chi_range_average'][()]=\
            chi_range_average[()]*int(args['sum_range'])
        fout[master]['entry']['sample']['goniometer']['chi_range_average'].attrs.create(\
            'units',ntstr(chi_range_average.attrs['units']),\
        dtype=ntstrdt(chi_range_average.attrs['units']))
    if chi_range_total != None:
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'chi_range_total',shape=chi_range_average.shape,dtype=chi_range_total.dtype)
        fout[master]['entry']['sample']['goniometer']['chi_range_total'][()]=\
        chi_range_average[()]
        fout[master]['entry']['sample']['goniometer']['chi_range_total'].attrs.create(\
            'units',ntstr(chi_range_total.attrs['units']),\
            dtype=ntstrdt(chi_range_total.attrs['units']))
    if kappa!=None:
        newshape = kappa.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'kappa',shape=newshape,dtype=kappa.dtype) 
        fout[master]['entry']['sample']['goniometer']['kappa'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        kappa[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        fout[master]['entry']['sample']['goniometer']['kappa'].attrs.create(\
            'units',ntstr(kappa.attrs['units']),\
            dtype=ntstrdt(kappa.attrs['units']))
    if kappa_end!=None:
        newshape = kappa_end.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'kappa_end',shape=newshape,dtype=kappa_end.dtype)
        fout[master]['entry']['sample']['goniometer']['kappa_end'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        kappa_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        fout[master]['entry']['sample']['goniometer']['kappa_end'].attrs.create(\
            'units',ntstr(kappa_end.attrs['units']),\
            dtype=ntstrdt(kappa_end.attrs['units']))
    if kappa_range_average != None:
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'kappa_range_average',shape=kappa_range_average.shape,dtype=kappa_range_average.dtype)
        fout[master]['entry']['sample']['goniometer']['kappa_range_average'][()]=\
        kappa_range_average[()]*int(args['sum_range'])
        fout[master]['entry']['sample']['goniometer']['kappa_range_average'].attrs.create(\
            'units',ntstr(kappa_range_average.attrs['units']),\
            dtype=ntstrdt(kappa_range_average.attrs['units']))
    if kappa_range_total != None:
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'kappa_range_total',shape=kappa_range_total.shape,dtype=kappa_range_total.dtype)
        fout[master]['entry']['sample']['goniometer']['kappa_range_total'][()]=\
        kappa_range_total[()]
        fout[master]['entry']['sample']['goniometer']['kappa_range_total'].attrs.create(\
            'units',ntstr(kappa_range_total.attrs['units']),\
             dtype=ntstrdt(kappa_range_total.attrs['units']))
    if angles != None:
        newshape = angles.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        print('angles.shape: ',angles.shape)
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'omega',shape=newshape,dtype=angles.dtype) 
        fout[master]['entry']['sample']['goniometer']['omega'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        angles[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        fout[master]['entry']['sample']['goniometer']['omega'].attrs.create(\
           'units',ntstr(angles.attrs['units']),\
           dtype=ntstrdt(angles.attrs['units']))
    if angles_end != None:
        newshape = angles_end.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        print('angles_end.shape: ',angles_end.shape)
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'omega_end',shape=newshape,dtype=angles_end.dtype) 
        fout[master]['entry']['sample']['goniometer']['omega_end'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        angles_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        fout[master]['entry']['sample']['goniometer']['omega_end'].attrs.create(\
            'units',ntstr(angles_end.attrs['units']),\
            dtype=ntstrdt(angles_end.attrs['units']))
    if osc_width != None:
        newshape = osc_width.shape
        print('osc_width.shape: ',osc_width.shape)
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
            'omega_range_average',shape=newshape,dtype=osc_width.dtype)
        fout[master]['entry']['sample']['goniometer']['omega_range_average'][()]=\
            osc_width[()]*int(args['sum_range'])
        fout[master]['entry']['sample']['goniometer']['omega_range_average'].attrs.create(\
            'units',ntstr(osc_width.attrs['units']),\
            dtype=ntstrdt(osc_width.attrs['units']))
    if osc_total != None:
        newshape = osc_total.shape
        print('osc_total.shape: ',osc_total.shape)
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
            'omega_range_total',shape=newshape,dtype=osc_total.dtype)
        fout[master]['entry']['sample']['goniometer']['omega_range_total'][()]=\
            osc_total[()]
        fout[master]['entry']['sample']['goniometer']['omega_range_total'].attrs.create(\
            'units',ntstr(osc_total.attrs['units']),\
            dtype=ntstrdt(osc_total.attrs['units']))
    if phi!=None:
        newshape = phi.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'phi',shape=newshape,dtype=phi.dtype) 
        fout[master]['entry']['sample']['goniometer']['phi'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        phi[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        fout[master]['entry']['sample']['goniometer']['phi'].attrs.create(\
            'units',ntstr(phi.attrs['units']),\
            dtype=ntstrdt(phi.attrs['units']))
    if phi_end!=None:
        newshape = phi_end.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'phi_end',shape=newshape,dtype=phi_end.dtype)
        fout[master]['entry']['sample']['goniometer']['phi_end'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        phi_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        fout[master]['entry']['sample']['goniometer']['phi_end'].attrs.create(\
            'units',ntstr(phi_end.attrs['units']),\
            dtype=ntstrdt(phi_end.attrs['units']))
    if phi_range_average != None:
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'phi_range_average',shape=phi_range_average.shape,dtype=phi_range_average.dtype)
        fout[master]['entry']['sample']['goniometer']['phi_range_average'][()]=\
        phi_range_average[()]*int(args['sum_range'])
        fout[master]['entry']['sample']['goniometer']['phi_range_average'].attrs.create(\
            'units',ntstr(phi_range_average.attrs['units']),\
            dtype=ntstrdt(phi_range_average.attrs['units']))
    if phi_range_total != None:
        fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'phi_range_total',shape=phi_range_total.shape,dtype=phi_range_total.dtype)
        fout[master]['entry']['sample']['goniometer']['phi_range_total'][()]=\
        phi_range_total[()]
        fout[master]['entry']['sample']['goniometer']['phi_range_total'].attrs.create(\
            'units',ntstr(phi_range_total.attrs['units']),\
            dtype=ntstrdt(phi_range_total.attrs['units']))
    if nxt_chi!=None:
        if not ('transformations' in  fout[master]['entry']['sample'].keys()):
            fout[master]['entry']['sample'].create_group('transformations')
            fout[master]['entry']['sample']['transformations'].\
                attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        newshape = nxt_chi.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['sample']['transformations'].create_dataset(\
        'chi',shape=newshape,dtype=nxt_chi.dtype)
        fout[master]['entry']['sample']['transformations']['chi'][0:(int(args['last_image'])-\
            int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
            nxt_chi[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        fout[master]['entry']['sample']['transformations']['chi'].attrs.create(\
            'units',ntstr(nxt_chi.attrs['units']),\
            dtype=ntstrdt(nxt_chi.attrs['units']))
    if nxt_chi_end!=None:
        if not ('transformations' in  fout[master]['entry']['sample'].keys()):
            fout[master]['entry']['sample'].create_group('transformations')
            fout[master]['entry']['sample']['transformations'].\
                attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        newshape = nxt_chi_end.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['sample']['transformations'].create_dataset(\
        'chi_end',shape=newshape,dtype=nxt_chi_end.dtype)
        fout[master]['entry']['sample']['transformations']['chi_end'][0:(int(args['last_image'])-\
            int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
            nxt_chi_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        fout[master]['entry']['sample']['transformations']['chi_end'].attrs.create(\
            'units',ntstr(nxt_chi_end.attrs['units']),\
            dtype=ntstrdt(nxt_chi_end.attrs['units']))
    if nxt_chi_range_average != None:
        if not ('transformations' in  fout[master]['entry']['sample'].keys()):
            fout[master]['entry']['sample'].create_group('transformations')
            fout[master]['entry']['sample']['transformations'].\
                attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        fout[master]['entry']['sample']['transformations'].create_dataset(\
        'chi_range_average',shape=nxt_chi_range_average.shape,dtype=nxt_chi_range_average.dtype)
        fout[master]['entry']['sample']['transformations']['chi_range_average'][()]=\
        nxt_chi_range_average[()]*int(args['sum_range'])
        fout[master]['entry']['sample']['transformations']['chi_range_average'].attrs.create(\
            'units',ntstr(nxt_chi_range_average.attrs['units']),\
            dtype=ntstrdt(nxt_chi_range_average.attrs['units']))
    if nxt_chi_range_total != None:
        if not ('transformations' in  fout[master]['entry']['sample'].keys()):
            fout[master]['entry']['sample'].create_group('transformations')
            fout[master]['entry']['sample']['transformations'].\
                attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        fout[master]['entry']['sample']['transformations'].create_dataset(\
        'chi_range_total',shape=nxt_chi_range_total.shape,dtype=nxt_chi_range_total.dtype)
        fout[master]['entry']['sample']['transformations']['chi_range_total'][()]=\
        nxt_chi_range_total[()]
        fout[master]['entry']['sample']['transformations']['chi_range_total'].attrs.create(\
            'units',ntstr(nxt_chi_range_total.attrs['units']),\
            dtype=ntstrdt(nxt_chi_range_total.attrs['units']))
    if nxt_omega!=None:
        if not ('transformations' in  fout[master]['entry']['sample'].keys()):
            fout[master]['entry']['sample'].create_group('transformations')
            fout[master]['entry']['sample']['transformations'].\
                attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        newshape = nxt_omega.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['sample']['transformations'].create_dataset(\
        'omega',shape=newshape,dtype=nxt_omega.dtype)
        fout[master]['entry']['sample']['transformations']['omega'][0:(int(args['last_image'])-\
            int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
            nxt_omega[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        if 'depends_on' in nxt_omega.attrs.keys():
            fout[master]['entry']['sample']['transformations']['omega']\
                .attrs.create('depends_on',nxt_omega.attrs['depends_on'])
        if 'offset' in nxt_omega.attrs.keys():
            fout[master]['entry']['sample']['transformations']['omega']\
                .attrs.create('offset',nxt_omega.attrs['offset'])
        if 'transformation_type' in nxt_omega.attrs.keys():
            fout[master]['entry']['sample']['transformations']['omega']\
                .attrs.create('transformation_type',nxt_omega.attrs['transformation_type'])
        if 'units' in nxt_omega.attrs.keys():
            fout[master]['entry']['sample']['transformations']['omega']\
                .attrs.create('units',nxt_omega.attrs['units'])
        if 'vector' in nxt_omega.attrs.keys():
            fout[master]['entry']['sample']['transformations']['omega']\
                .attrs.create('vector',nxt_omega.attrs['vector'])
    if nxt_omega_end!=None:
        if not ('transformations' in  fout[master]['entry']['sample'].keys()):
            fout[master]['entry']['sample'].create_group('transformations')
            fout[master]['entry']['sample']['transformations'].\
                attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        newshape = nxt_omega_end.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['sample']['transformations'].create_dataset(\
        'omega_end',shape=newshape,dtype=nxt_omega_end.dtype)
        fout[master]['entry']['sample']['transformations']['omega_end'][0:(int(args['last_image'])-\
            int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
            nxt_omega_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        fout[master]['entry']['sample']['transformations']['omega_end'].attrs.create(\
            'units',ntstr(nxt_omega_end.attrs['units']),\
            dtype=ntstrdt(nxt_omega_end.attrs['units']))
    if nxt_omega_range_average != None:
        if not ('transformations' in  fout[master]['entry']['sample'].keys()):
            fout[master]['entry']['sample'].create_group('transformations')
            fout[master]['entry']['sample']['transformations'].\
                attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        fout[master]['entry']['sample']['transformations'].create_dataset(\
        'omega_range_average',shape=nxt_omega_range_average.shape,dtype=nxt_omega_range_average.dtype)
        fout[master]['entry']['sample']['transformations']['omega_range_average'][()]=\
        nxt_omega_range_average[()]*int(args['sum_range'])
        fout[master]['entry']['sample']['transformations']['omega_range_average'].attrs.create(\
            'units',ntstr(nxt_omega_range_average.attrs['units']),\
            dtype=ntstrdt(nxt_omega_range_average.attrs['units']))
    if nxt_omega_range_total != None:
        if not ('transformations' in  fout[master]['entry']['sample'].keys()):
            fout[master]['entry']['sample'].create_group('transformations')
            fout[master]['entry']['sample']['transformations'].\
                attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        fout[master]['entry']['sample']['transformations'].create_dataset(\
        'omega_range_total',shape=nxt_omega_range_total.shape,dtype=nxt_omega_range_total.dtype)
        fout[master]['entry']['sample']['transformations']['omega_range_total'][()]=\
        nxt_omega_range_total[()]
        fout[master]['entry']['sample']['transformations']['omega_range_total'].attrs.create(\
            'units',ntstr(nxt_omega_range_total.attrs['units']),\
            dtype=ntstrdt(nxt_omega_range_total.attrs['units']))
    if nxt_phi_range_average != None:
        if not ('transformations' in  fout[master]['entry']['sample'].keys()):
            fout[master]['entry']['sample'].create_group('transformations')
            fout[master]['entry']['sample']['transformations'].\
                attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        fout[master]['entry']['sample']['transformations'].create_dataset(\
        'phi_range_average',shape=nxt_phi_range_average.shape,dtype=nxt_phi_range_average.dtype)
        fout[master]['entry']['sample']['transformations']['phi_range_average'][()]=\
        nxt_phi_range_average[()]*int(args['sum_range'])
        fout[master]['entry']['sample']['transformations']['phi_range_average'].attrs.create(\
            'units',ntstr(nxt_phi_range_average.attrs['units']),\
            dtype=ntstrdt(nxt_phi_range_average.attrs['units']))
    if nxt_phi!=None:
        if not ('transformations' in  fout[master]['entry']['sample'].keys()):
            fout[master]['entry']['sample'].create_group('transformations')
            fout[master]['entry']['sample']['transformations'].\
                attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        newshape = nxt_phi.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['sample']['transformations'].create_dataset(\
        'phi',shape=newshape,dtype=nxt_phi.dtype)
        fout[master]['entry']['sample']['transformations']['phi'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        nxt_phi[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        fout[master]['entry']['sample']['transformations']['phi'].attrs.create(\
            'units',ntstr(nxt_phi.attrs['units']),\
            dtype=ntstrdt(nxt_phi.attrs['units']))
    if nxt_phi_end!=None:
        if not ('transformations' in  fout[master]['entry']['sample'].keys()):
            fout[master]['entry']['sample'].create_group('transformations')
            fout[master]['entry']['sample']['transformations'].\
                attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        newshape = nxt_phi_end.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['sample']['transformations'].create_dataset(\
        'phi_end',shape=newshape,dtype=nxt_phi_end.dtype)
        fout[master]['entry']['sample']['transformations']['phi_end'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        nxt_phi_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        fout[master]['entry']['sample']['transformations']['phi_end'].attrs.create(\
            'units',ntstr(nxt_phi_end.attrs['units']),\
            dtype=ntstrdt(nxt_phi_end.attrs['units']))
    if nxt_kappa!=None:
        if not ('transformations' in  fout[master]['entry']['sample'].keys()):
            fout[master]['entry']['sample'].create_group('transformations')
            fout[master]['entry']['sample']['transformations'].\
               attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        newshape = nxt_kappa.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['sample']['transformations'].create_dataset(\
        'kappa',shape=nxt_kappa.shape,dtype=nxt_kappa.dtype)
        fout[master]['entry']['sample']['transformations']['kappa'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        nxt_kappa[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        fout[master]['entry']['sample']['transformations']['kappa'].attrs.create(\
            'units',ntstr(nxt_kappa.attrs['units']),\
            dtype=ntstrdt(nxt_kappa.attrs['units']))
    if nxt_kappa_end!=None:
        if not ('transformations' in  fout[master]['entry']['sample'].keys()):
            fout[master]['entry']['sample'].create_group('transformations')
            fout[master]['entry']['sample']['transformations'].\
                attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        newshape = nxt_kappa_end.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['sample']['transformations'].create_dataset(\
        'kappa_end',shape=newshape,dtype=nxt_kappa_end.dtype)
        fout[master]['entry']['sample']['transformations']['kappa_end'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        nxt_kappa_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        fout[master]['entry']['sample']['transformations']['kappa_end'].attrs.create(\
            'units',ntstr(nxt_kappa_end.attrs['units']),\
            dtype=ntstrdt(nxt_kappa_end.attrs['units']))
    if nxt_kappa_range_total != None:
        if not ('transformations' in  fout[master]['entry']['sample'].keys()):
            fout[master]['entry']['sample'].create_group('transformations')
            fout[master]['entry']['sample']['transformations'].\
                attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        fout[master]['entry']['sample']['transformations'].create_dataset(\
        'kappa_range_total',shape=nxt_kappa_range_total.shape,dtype=nxt_kappa_range_total.dtype)
        fout[master]['entry']['sample']['transformations']['kappa_range_total'][()]=\
        nxt_kappa_range_total[()]
        fout[master]['entry']['sample']['transformations']['kappa_range_total'].attrs.create(\
            'units',ntstr(nxt_kappa_range_total.attrs['units']),\
            dtype=ntstrdt(nxt_kappa_range_total.attrs['units']))
    if nxt_translation!=None:
        if not ('transformations' in  fout[master]['entry']['sample'].keys()):
            fout[master]['entry']['sample'].create_group('transformations')
            fout[master]['entry']['sample']['transformations'].\
                attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
        newshape = nxt_translation.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['sample']['transformations'].create_dataset(\
        'translation',shape=newshape,dtype=nxt_translation.dtype)
        fout[master]['entry']['sample']['transformations']['translation'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        nxt_translation[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        if 'depends_on' in nxt_translation.attrs.keys():
            fout[master]['entry']['sample']['transformations']['translation']\
                .attrs.create('depends_on',nxt_translation.attrs['depends_on'])
        if 'offset' in nxt_translation.attrs.keys():
            fout[master]['entry']['sample']['transformations']['translation']\
                .attrs.create('offset',nxt_translation.attrs['offset'])
        if 'transformation_type' in nxt_translation.attrs.keys():
            fout[master]['entry']['sample']['transformations']['translation']\
                .attrs.create('transformation_type',nxt_translation.attrs['transformation_type'])
        if 'units' in nxt_translation.attrs.keys():
            fout[master]['entry']['sample']['transformations']['translation']\
                .attrs.create('units',nxt_translation.attrs['units'])
        if 'vector' in nxt_translation.attrs.keys():
            fout[master]['entry']['sample']['transformations']['translation']\
                .attrs.create('vector',nxt_translation.attrs['vector'])
    for nout_block in range(1,out_number_of_blocks+1):
        fout[master]['entry']['data']["data_"+str(nout_block).zfill(6)] \
            = h5py.ExternalLink(os.path.basename(args['out_file'])+"_"+str(nout_block).zfill(6)+".h5", "/entry/data/data")
        if args['out_squash'] != None:
            fout[master]['entry']['data']["squash_"+str(nout_block).zfill(6)] \
            = h5py.ExternalLink(args['out_squash']+"_"+str(nout_block).zfill(6)+".h5", "/entry/data")
    fout[master].close()
    manager = BackgroundJobManager()
    Jobs = {}
    DoneJobs = {}
    if args['threads'] != None and args['threads'] > 1:
       for mythread in range(1,args['threads']+1):
           print('subprocess.run '+'cctbx.python '+' '.join(sys.argv)+\
               ' '+'-t '+str(mythread)+ ' &')
           Jobs[mythread] = manager.submit('cctbx.python '+' '.join(sys.argv)+\
               ' '+'-t '+str(mythread),args['out_file']+'_thread'+str(mythread)+'_')
           # Check status immediately
           print('Job ', mythread, 'status: ', manager.get_status(Jobs[mythread]))
       nrunning = args['threads']
       while nrunning > 0:
           nrunning = 0
           for mythread in range(1,args['threads']+1):
               mystatus = manager.get_status(Jobs[mythread])
               if mystatus == 'running':
                   nrunning = nrunning+1
                   #print(' thread ', mythread, ' status: ',mystatus)
               else:
                   if DoneJobs.get(mythread) == None:
                       DoneJobs[mythread]=manager.get_result(Jobs[mythread])
           print(' nrunning: ', nrunning)
           if nrunning == 0:
               print(' exiting from base thread ')
               time.sleep(1)
               sys.exit()
               break
           time.sleep(20)
           #for mythread in range(1,args['threads']+1):
           #    print('Job ', mythread, 'status: ', manager.get_status(Jobs[mythread])) 
else:
    fout[master] = h5py.File(args['out_master']+".h5",'r')

print('out_number_of_blocks: ', out_number_of_blocks)
print('threads: ',threads)
print('thread: ',thread)
out_block_start = 1
out_block_limit = out_number_of_blocks+1
out_block_step = 1
if thread > out_number_of_blocks:
    thread = out_number_of_blocks
if threads != None and threads > 1 and thread > 0:
    out_block_start = thread
    out_block_step = threads
if args['verbose'] == True:
    print('nout_block in range(',out_block_start,out_number_of_blocks+1,out_block_step,')')
for nout_block in range(out_block_start,out_number_of_blocks+1,out_block_step):
    if threads == None or threads <= 1:
        if args['verbose'] == True:
            print('nout_block in range(',out_block_start,out_number_of_blocks+1,out_block_step,')')
    elif threads != None and threads > 1 and thread > 0:
        mythread = (nout_block-1)%threads +1
        if mythread != thread:
            continue; 
    nout_image=1+(nout_block-1)*out_number_per_block
    image_nr_low=nout_image
    lim_nout_image = nout_image+out_number_per_block
    if lim_nout_image > out_max_image+1:
        lim_nout_image = out_max_image+1
    image_nr_high = lim_nout_image-1
    new_images[nout_image]= \
        new_images_buffer_manager.get_dataset(str(nout_image))
    nout_data_shape = new_images[nout_image].shape
    print ('nout_data_shape: ',nout_data_shape)
    if args['verbose'] == True:
        print('nout_block: ',nout_block)
        print('image_nr_low: ',image_nr_low)
        print('image_nr_high :',image_nr_high)
    fout[nout_block] = h5py.File(args['out_file']+"_"+str(nout_block).zfill(6)+".h5",'w')
    fout[nout_block].create_group('entry')
    fout[nout_block]['entry'].attrs.create('NX_class',ntstr('NXentry'),dtype=ntstrdt('NXentry'))
    fout[nout_block]['entry'].create_group('data')
    fout[nout_block]['entry']['data'].attrs.create('NX_class',ntstr('NXdata'),dtype=ntstrdt('NXdata'))
    if args['out_squash'] != None:
        fout_squash[nout_block] = h5py.File(args['out_squash']+"_"+str(nout_block).zfill(6)+".h5",'w')
        fout_squash[nout_block].create_group('entry')
        fout_squash[nout_block]['entry'].attrs.create('NX_class',ntstr('NXentry'),dtype=ntstrdt('NXentry'))
        fout_squash[nout_block]['entry'].create_group('data')
        fout_squash[nout_block]['entry']['data'].attrs.create('NX_class',ntstr('NXdata'),dtype=ntstrdt('NXdata'))
    mydata_type=np.uint16
    mydata_float=False
    mydata_int=True
    mydata_signed=False
    mydata_size=2
    if args['ufloat'] !=  None:
        mydata_float=True
        mydata_int=False
        if args['ufloat'] >= 0:
            mydata_signed=False
        else:
            mydata_signed=True
        if args['ufloat'] == 2 or args['ufloat'] == -2:
            mydata_size = 2
            mydata_type = np.float16
        elif args['ufloat'] == 4 or args['ufloat'] == -4:
            mydata_size = 4
            mydata_type = np.float32
        else:
            print("args['ufloat'] =",args['ufloat']," not supported")
            sys.exit(-1)
    elif args['uint'] !=  None:
        mydata_float=False  
        mydata_int=True
        if args['uint'] >= 0:
            mydata_signed=False
        else:
            mydata_signed=True
        if args['uint'] == 2:
            mydata_size = 2
            mydata_type = np.uint16
        elif args['uint'] == -2:
            mydata_size = 2
            mydata_type = np.int16 
        elif args['ufloat'] == 4:
            mydata_size = 4
            mydata_type = np.uint32
        elif args['ufloat'] == -4:
            mydata_size = 4
            mydata_type = np.int32
        else:
            print("args['uint'] =",args['uint']," not supported")
            sys.exit(-1)
    else:
        mydata_float=False  
        mydata_int=True 
        mydata_signed = False
        mydata_size = 2
        mydata_type = np.uint16
    if args['compression']==None:
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype=mydata_type,chunks=(1,nout_data_shape[0],nout_data_shape[1]))
    elif args['compression']=='bshuf' or args['compression']=='BSHUF':
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype=mydata_type,chunks=(1,nout_data_shape[0],nout_data_shape[1]),
            **hdf5plugin.Bitshuffle(nelems=0,cname='none'))
    elif args['compression']=='bslz4' or args['compression']=='BSLZ4':
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype=mydata_type,chunks=(1,nout_data_shape[0],nout_data_shape[1]),
            **hdf5plugin.Bitshuffle(nelems=0,cname='lz4'))
    elif args['compression']=='bszstd' or args['compression']=='BSZSTD':
        if args['compression_level'] == None:
            clevel=3
        elif int(args['compression_level']) > 22:
            clevel=22
        elif int(args['compression_level']) < -2:
            clevel=-2
        else:
            clevel=int(args['compression_level'])
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype=mydata_type,chunks=(1,nout_data_shape[0],nout_data_shape[1]),
            **hdf5plugin.Bitshuffle(nelems=0,cname='zstd',clevel=clevel))
    elif args['compression']=='zstd' or args['compression']=='ZSTD':
        if args['compression_level'] == None:
            clevel=3
        elif int(args['compression_level']) > 22:
            clevel=22
        elif int(args['compression_level']) < -2:
            clevel=-2
        else:
            clevel=int(args['compression_level'])
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype=mydata_type,chunks=(1,nout_data_shape[0],nout_data_shape[1]),
            **hdf5plugin.Blosc(cname='zstd',clevel=clevel,shuffle=hdf5plugin.Blosc.NOSHUFFLE))
    else:
        print('l_bnl_compress.py: unrecognized compression, reverting to bslz4')
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype=mydata_type,chunks=(1,nout_data_shape[0],nout_data_shape[1]),
            **hdf5plugin.Bitshuffle(nelems=0,cname='lz4'))
    fout[nout_block]['entry']['data']['data'].attrs.create('image_nr_low',dtype=np.uint64,
        data=np.uint64(image_nr_low))
    fout[nout_block]['entry']['data']['data'].attrs.create('image_nr_high',dtype=np.uint64,
        data=np.uint64(image_nr_high))
    if args['out_squash'] != None:
        fout_squash[nout_block]['entry']['data'].attrs.create('image_nr_low',dtype=np.uint64,
            data=np.uint64(image_nr_low))
        fout[nout_block]['entry']['data']['data'].attrs.create('image_nr_high',dtype=np.uint64,
            data=np.uint64(image_nr_high))
    print("fout[nout_block]['entry']['data']['data']: ",fout[nout_block]['entry']['data']['data'])
    hcomp_tot_compimgsize=0
    nhcomp_img = 0
    j2k_tot_compimgsize=0
    nj2k_img=0
    my_dtype = mydata_type

    for out_image in range(nout_image,lim_nout_image):
        new_images[out_image]= \
            new_images_buffer_manager.get_dataset(str(out_image))
        try:
            (newresult,new_satval)=scale_with_saturation(np.clip(new_images[out_image][0:nout_data_shape[0],\
                0:nout_data_shape[1]],0,satval), float(args['scale_factor']), satval)
            del new_images[out_image]
        except:
            print('out_image: ',out_image)
            print('nout_data_shape: ',nout_data_shape)
        if args['verbose'] == True:
            print('l_bnl_compress.py satval: ',satval,' new_satval: ',new_satval,' scale_factor: ',float(args['scale_factor']))
        if args['logarithm_base'] != None:
            log_base=args['logarithm_base']
            if log_base == 'e':
                log_base = np.exp(1.)
            else:
                log_base=float(log_base)
            if (log_base < 2.):
                print('l_bnl_compress needs loarithm base >= 2., abort')
                sys.exit(-1)
            newresult=log_transform(newresult,base=log_base,data_signed=mydata_signed,\
                 data_size=mydata_size,data_type=mydata_type)
        if args['exponent'] != None:
             exponent=args['exponent']
             if exponent == 'e':
                 exponent = np.exp(1.)
             else:
                 exponent = float(exponent)
             if (exponent < 2.):
                print('l_bnl_compress needs exponent >= 2., abort')
                sys.exit(-1)
             newresult=inverse_log_transform(newresult,base=exponent,data_signed=mydata_signed,\
                 data_size=mydata_size,data_type=mydata_type)
        if args['hcomp_scale']==None \
            and args['j2k_target_compression_ratio']==None \
            and args['j2k_alt_target_compression_ratio']==None:
            fout[nout_block]['entry']['data']['data'][out_image-nout_image,0:nout_data_shape[0],0:nout_data_shape[1]] \
              =np.clip(newresult,0,new_satval)
        elif args['hcomp_scale']!=None:
            myscale=args['hcomp_scale']
            if myscale < 1 :
                myscale=16
            img16=np.clip(newresult,0,new_satval)
            img16=img16.astype('i2')
            fits_bytes, original_shape = compress_HCarray(img16,new_satval,myscale)
            if args['out_squash'] != None:
                fout_squash[nout_block]['entry']['data'].create_dataset('data_'+str(out_image).zfill(6),data=repr(fits_bytes))
            hdecomp_data=decompress_HCarray(fits_bytes,original_shape,scale=myscale)
            decompressed_data= (np.maximum(hdecomp_data,0).astype(np.uint16)).reshape((nout_data_shape[0],nout_data_shape[1]))
            decompressed_data = np.clip(decompressed_data,0,new_satval)
            hcomp_tot_compimgsize = hcomp_tot_compimgsize+sys.getsizeof(fits_bytes)
            nhcomp_img = nhcomp_img+1
            if args['verbose'] == True:
                print('l_bnl_compress.py: Hcompress sys.getsizeof(): ', sys.getsizeof(fits_bytes))
                print('                   decompressed_data sys.getsizeof: ', sys.getsizeof(decompressed_data))
            fout[nout_block]['entry']['data']['data'][out_image-nout_image, \
                0:nout_data_shape[0],0:nout_data_shape[1]] \
                = np.asarray(decompressed_data,dtype='u2')
            del decompressed_data
            del hdecomp_data
            del fits_bytes
            del img16
        elif args['j2k_target_compression_ratio']!=None:
            mycrat=int(args['j2k_target_compression_ratio'])
            if mycrat < 1:
                mycrat=125
            img16=np.clip(newresult,0,new_satval)
            outtemp=args['out_file']+"_"+str(out_image).zfill(6)+".j2k"
            print("outtemp: ",outtemp)
            xmycrat = [mycrat]
            if mycrat < 4000:
                ycrat = mycrat * 2
                while ycrat < 8000 :
                    xmycrat.insert(0,ycrat)
                    ycrat = ycrat *2
            j2k=glymur.Jp2k(outtemp, data=img16, cratios=xmycrat)
            print ('j2k.dtype', j2k.dtype)
            print ('j2k.shape', j2k.shape)
            jdecomped = glymur.Jp2k(outtemp)
            jdecomped = np.maximum(0,jdecomped[:])
            arr_final = np.array(jdecomped, dtype='u2')
            file_size = os.path.getsize(outtemp)
            if args['out_squash'] != None:
                fout_squash[nout_block]['entry']['data']\
                    .create_dataset('data_'+str(out_image).zfill(6),data=jdecomped)
                if args['verbose'] == True:
                    print (fout_squash[nout_block]['entry']['data']['data_'+str(out_image).zfill(6)])
                fout_squash[nout_block]['entry']['data']['data_'+str(out_image).zfill(6)]\
                .attrs.create('compression',ntstr('j2k'),dtype=ntstrdt('j2k'))
                fout_squash[nout_block]['entry']['data']['data_'+str(out_image).zfill(6)]\
                .attrs.create('compression_level',mycrat,dtype=ntstrdt('i2'))
            j2k_tot_compimgsize = j2k_tot_compimgsize + file_size
            nj2k_img = nj2k_img+1
            if args['verbose'] == True:
                print('l_bnl_compress.py: JPEG-2000 outtemp file_size: ', file_size )
            fout[nout_block]['entry']['data']['data'][out_image-nout_image, \
                0:nout_data_shape[0],0:nout_data_shape[1]] \
                = np.clip(arr_final,0,new_satval)
            del arr_final
            del jdecomped
            del j2k
            os.remove(outtemp)
        else:
            mycrat=int(args['j2k_alt_target_compression_ratio'])
            if mycrat < 1:
                mycrat=125
            img16=np.clip(newresult,0,new_satval)
            outtemp_tif=args['out_file']+"_"+str(out_image).zfill(6)+".tif"
            save_uint16_tiff_simple(img16,outtemp_tif)
            outtemp=args['out_file']+"_"+str(out_image).zfill(6)+".j2k"
            outtemp_outtiff=args['out_file']+"_decompresses_"+str(out_image).zfill(6)+".tif"
            compress_tif_to_jp2(outtemp_tif, outtemp, ratios=crat_list(mycrat))
            j2kimage = Image.open(outtemp)
            j2k=np.array(j2kimage)
            decompress_jp2_to_tif(outtemp, outtemp_outtiff)
            jdecomped = np.clip(load_tiff_to_numpy(outtemp_outtiff),0,new_satval)
            print("outtemp: ",outtemp)
            jdecomped = np.maximum(0,jdecomped[:])
            arr_final = np.array(jdecomped, dtype='u2')
            file_size = os.path.getsize(outtemp)
            if args['out_squash'] != None:
                fout_squash[nout_block]['entry']['data']\
                    .create_dataset('data_'+str(out_image).zfill(6),data=j2k)
                if args['verbose'] == True:
                    print (fout_squash[nout_block]['entry']['data']['data_'+str(out_image).zfill(6)])
                fout_squash[nout_block]['entry']['data']['data_'+str(out_image).zfill(6)]\
                .attrs.create('compression',ntstr('j2k'),dtype=ntstrdt('j2k'))
                fout_squash[nout_block]['entry']['data']['data_'+str(out_image).zfill(6)]\
                .attrs.create('compression_level',mycrat,dtype=ntstrdt('i2'))
            j2k_tot_compimgsize = j2k_tot_compimgsize + file_size
            nj2k_img = nj2k_img+1
            if args['verbose'] == True:
                print('l_bnl_compress.py: JPEG-2000 outtemp file_size: ', file_size )
            fout[nout_block]['entry']['data']['data'][out_image-nout_image, \
                0:nout_data_shape[0],0:nout_data_shape[1]] \
                = np.clip(arr_final,0,new_satval)
            del arr_final
            del jdecomped
            del j2k
            os.remove(outtemp)
            os.remove(outtemp_tif)
            os.remove(outtemp_outtiff)
    del fout[nout_block]
    if nhcomp_img > 0:
        print('l_bnl_compress.py: hcomp avg compressed image size: ', int(.5+hcomp_tot_compimgsize/nhcomp_img))
    if nj2k_img > 0:
        print('l_bnl_compress.py: j2k avg compressed imgage size: ', int(.5+j2k_tot_compimgsize/nj2k_img))
os.remove(new_images_buffer)
sys.exit()
