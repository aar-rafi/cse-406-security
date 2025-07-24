#!/usr/bin/env python3
"""
Test Parallel vs. Non-Parallel CTR File Encryption/Decryption

This script benchmarks and verifies correctness of parallel vs. non-parallel CTR mode
file encryption and decryption using file_crypto.py.
"""
import os
import time
import multiprocessing
from file_crypto import encrypt_file, decrypt_file, calculate_file_hash, FileEncryptionError

# Test file sizes in KB
FILE_SIZES = [10, 100, 1000]  # 10KB, 100KB, 1MB
KEY = "ParallelTestKey123"

def create_test_file(filename, size_kb):
    """Create a test file of the specified size in KB."""
    if os.path.exists(filename) and os.path.getsize(filename) >= size_kb * 1024:
        print(f"Test file {filename} already exists with sufficient size.")
        return
    print(f"Creating test file of {size_kb} KB...")
    with open(filename, 'wb') as f:
        f.write(os.urandom(size_kb * 1024))
    print(f"Created {filename} ({os.path.getsize(filename)/1024:.2f} KB)")

def test_encrypt_decrypt(file_size_kb, parallel):
    """Test encryption and decryption for a specific file size and mode."""
    test_file = f"test_file_{file_size_kb}kb.bin"
    enc_file = f"test_file_{file_size_kb}kb.{'parallel' if parallel else 'serial'}.enc"
    dec_file = f"test_file_{file_size_kb}kb.{'parallel' if parallel else 'serial'}.dec"
    
    # Create test file if it doesn't exist
    create_test_file(test_file, file_size_kb)
    
    mode = "Parallel" if parallel else "Serial"
    print(f"[{mode}] Testing with {file_size_kb} KB file...")
    
    # Encryption test
    start = time.time()
    encrypt_file(test_file, enc_file, KEY, mode='ctr', parallel=parallel)
    enc_time = time.time() - start
    
    # Decryption test
    start = time.time()
    decrypt_file(enc_file, dec_file, KEY, parallel=parallel)
    dec_time = time.time() - start
    
    # Verify correctness
    orig_hash = calculate_file_hash(test_file)
    dec_hash = calculate_file_hash(dec_file)
    is_correct = orig_hash == dec_hash
    
    if not is_correct:
        print(f"WARNING: {mode} decryption did not match original for {file_size_kb} KB!")
    
    return enc_time, dec_time, is_correct

def run_benchmarks():
    """Run benchmarks for all file sizes with both serial and parallel modes."""
    results = {}
    
    for size_kb in FILE_SIZES:
        results[size_kb] = {}
        
        # Test serial mode
        serial_enc, serial_dec, serial_correct = test_encrypt_decrypt(size_kb, parallel=False)
        results[size_kb]['serial'] = {
            'encrypt_time': serial_enc,
            'decrypt_time': serial_dec,
            'correct': serial_correct
        }
        
        # Test parallel mode
        parallel_enc, parallel_dec, parallel_correct = test_encrypt_decrypt(size_kb, parallel=True)
        results[size_kb]['parallel'] = {
            'encrypt_time': parallel_enc,
            'decrypt_time': parallel_dec,
            'correct': parallel_correct
        }
    
    return results

def display_system_info():
    """Display information about the system's CPU cores."""
    logical_cores = os.cpu_count()
    # Try to get physical cores if possible (requires psutil which may not be installed)
    physical_cores = None
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False)
    except (ImportError, AttributeError):
        pass
    
    print("=== System Information ===")
    print(f"Logical CPU Cores/Threads: {logical_cores}")
    if physical_cores:
        print(f"Physical CPU Cores: {physical_cores}")
    print(f"Threads used for parallel processing: {os.cpu_count()}")
    print("=" * 70)

def display_results_chart(results):
    """Display benchmark results in a chart format."""
    print("\n" + "=" * 70)
    print("CTR Mode Encryption/Decryption Performance Benchmark")
    print("=" * 70)
    
    # Print header
    print(f"{'File Size':^10} | {'Mode':^10} | {'Encrypt Time (s)':^16} | {'Decrypt Time (s)':^16} | {'Speedup':^10}")
    print("-" * 70)
    
    # Print rows
    for size_kb in FILE_SIZES:
        serial_enc = results[size_kb]['serial']['encrypt_time']
        serial_dec = results[size_kb]['serial']['decrypt_time']
        parallel_enc = results[size_kb]['parallel']['encrypt_time']
        parallel_dec = results[size_kb]['parallel']['decrypt_time']
        
        # Calculate speedup factors
        enc_speedup = serial_enc / parallel_enc if parallel_enc > 0 else 0
        dec_speedup = serial_dec / parallel_dec if parallel_dec > 0 else 0
        
        # Print serial row
        print(f"{size_kb:^10} | {'Serial':^10} | {serial_enc:^16.4f} | {serial_dec:^16.4f} | {'-':^10}")
        
        # Print parallel row
        print(f"{' ':^10} | {'Parallel':^10} | {parallel_enc:^16.4f} | {parallel_dec:^16.4f} | {enc_speedup:^10.2f}x")
        
        # Add separator between file sizes
        if size_kb != FILE_SIZES[-1]:
            print("-" * 70)
    
    print("=" * 70)
    print(f"Note: Speedup is Serial time / Parallel time. Higher is better.")
    print("=" * 70)

def main():
    display_system_info()
    print("\nRunning CTR mode benchmarks for different file sizes...")
    results = run_benchmarks()
    display_results_chart(results)
    print("\nBenchmark completed.")

if __name__ == "__main__":
    main() 