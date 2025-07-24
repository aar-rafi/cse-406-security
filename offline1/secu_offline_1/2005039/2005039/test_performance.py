#!/usr/bin/env python3
"""
Performance Testing for Cryptographic Operations

This module tests the performance of various cryptographic operations
including AES encryption/decryption and ECDH key exchange.
"""

import os
import time
import random
import argparse
from typing import List, Dict, Any, Tuple

# Import our modules
from aes import encrypt_text, decrypt_text, encrypt_cbc, decrypt_cbc, key_expansion, generate_iv
from ecc import EllipticCurve, ECDH, generate_curve_params
from secure_socket import SecureClient, SecureServer
from perf_metrics import (
    Timer, MemoryMonitor, MetricsCollector, PerformanceReport,
    timed, measure_memory, global_metrics
)

# Constants
DEFAULT_ITERATIONS = 20
DEFAULT_KEY_SIZES_BITS = [128, 192, 256]
DEFAULT_DATA_SIZES_BYTES = [1024, 10240, 102400]  # 1KB, 10KB, 100KB
DEFAULT_FIELD_BITS = [112, 128, 192, 256]  # For ECDH curves

# Check if matplotlib is available for plotting
MATPLOTLIB_AVAILABLE = False
try:
    import numpy as np
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: Matplotlib not available. Plotting functions will be disabled.")

def generate_random_data(size_bytes: int) -> bytes:
    """
    Generate random data of the specified size.
    
    Args:
        size_bytes: Size of data to generate in bytes
        
    Returns:
        Random data as bytes
    """
    return os.urandom(size_bytes)

@timed("aes_key_expansion")
def test_aes_key_expansion(key: bytes) -> None:
    """
    Test AES key expansion performance.
    
    Args:
        key: AES key
    """
    key_expansion(key, len(key))

@timed("aes_encrypt")
def test_aes_encryption(plaintext: bytes, key: bytes) -> Tuple[bytes, bytes]:
    """
    Test AES encryption performance.
    
    Args:
        plaintext: Data to encrypt
        key: AES key
        
    Returns:
        Tuple of (encrypted_iv, ciphertext)
    """
    iv = generate_iv()
    return encrypt_cbc(plaintext, key, iv)

@timed("aes_decrypt")
def test_aes_decryption(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    """
    Test AES decryption performance.
    
    Args:
        ciphertext: Encrypted data
        key: AES key
        iv: Initialization vector
        
    Returns:
        Decrypted plaintext
    """
    return decrypt_cbc(ciphertext, key, iv)

@timed("ecdh_curve_generation")
def test_ecdh_curve_generation(field_bits: int) -> EllipticCurve:
    """
    Test ECDH curve generation performance.
    
    Args:
        field_bits: Size of field in bits
        
    Returns:
        Generated elliptic curve
    """
    params = generate_curve_params(field_bits)
    return EllipticCurve(params['a'], params['b'], params['p'], params['g'], params['n'])

@timed("ecdh_key_generation")
def test_ecdh_key_generation(curve: EllipticCurve) -> Tuple[int, Tuple[int, int]]:
    """
    Test ECDH key generation performance.
    
    Args:
        curve: Elliptic curve
        
    Returns:
        Generated keypair
    """
    ecdh = ECDH(curve)
    return ecdh.generate_keypair()

@timed("ecdh_shared_secret")
def test_ecdh_shared_secret(curve: EllipticCurve, private_key: int, public_key: Tuple[int, int]) -> bytes:
    """
    Test ECDH shared secret computation performance.
    
    Args:
        curve: Elliptic curve
        private_key: Private key
        public_key: Public key
        
    Returns:
        Computed shared secret
    """
    ecdh = ECDH(curve)
    ecdh.private_key = private_key
    return ecdh.compute_shared_secret(public_key)

def test_aes_performance(
    key_sizes_bits: List[int],
    data_sizes_bytes: List[int],
    iterations: int
) -> None:
    """
    Comprehensive testing of AES performance with different key and data sizes.
    
    Args:
        key_sizes_bits: List of key sizes in bits to test
        data_sizes_bytes: List of data sizes in bytes to test
        iterations: Number of iterations for each test
    """
    print(f"\n===== AES Performance Testing =====")
    print(f"Key sizes: {key_sizes_bits} bits")
    print(f"Data sizes: {[size/1024 for size in data_sizes_bytes]} KB")
    print(f"Iterations: {iterations}")
    
    # Convert key sizes from bits to bytes
    key_sizes_bytes = [size // 8 for size in key_sizes_bits]
    
    # Test each combination of key size and data size
    for key_size_bytes in key_sizes_bytes:
        key_size_bits = key_size_bytes * 8
        
        print(f"\n----- Testing AES-{key_size_bits} -----")
        key = os.urandom(key_size_bytes)
        
        # Test key expansion
        print("\nKey Expansion:")
        for _ in range(iterations):
            test_aes_key_expansion(key)
        
        # Record key size in metrics
        key_exp_stats = global_metrics.get_timing_stats("aes_key_expansion")
        if "aes_key_expansion" in key_exp_stats:
            global_metrics.record_key_size_metric(
                key_size_bits, 
                "key_expansion", 
                key_exp_stats["aes_key_expansion"]["mean"]
            )
        
        for data_size in data_sizes_bytes:
            plaintext = generate_random_data(data_size)
            
            print(f"\nData Size: {data_size/1024:.1f} KB")
            
            # Encryption tests
            enc_times = []
            dec_times = []
            
            for i in range(iterations):
                # Test encryption
                with Timer("aes_encrypt", global_metrics) as timer:
                    iv, ciphertext = encrypt_cbc(plaintext, key, generate_iv())
                enc_times.append(timer.elapsed_ms)
                
                # Test decryption
                with Timer("aes_decrypt", global_metrics) as timer:
                    decrypted = decrypt_cbc(ciphertext, key, iv)
                dec_times.append(timer.elapsed_ms)
                
                # Record timing for key size metrics
                global_metrics.record_key_size_metric(key_size_bits, "encryption", enc_times[-1])
                global_metrics.record_key_size_metric(key_size_bits, "decryption", dec_times[-1])
                
                # Record timing for data size metrics
                global_metrics.record_data_size_metric(data_size, "encryption", enc_times[-1])
                global_metrics.record_data_size_metric(data_size, "decryption", dec_times[-1])
                
                # Calculate throughput
                enc_throughput = (data_size / 1024 / 1024) / (enc_times[-1] / 1000)  # MB/s
                dec_throughput = (data_size / 1024 / 1024) / (dec_times[-1] / 1000)  # MB/s
                
                global_metrics.record_custom_metric("throughput", f"encryption_{key_size_bits}_{data_size}", enc_throughput)
                global_metrics.record_custom_metric("throughput", f"decryption_{key_size_bits}_{data_size}", dec_throughput)
            
            # Print summary for this data size
            avg_enc_time = sum(enc_times) / len(enc_times)
            avg_dec_time = sum(dec_times) / len(dec_times)
            
            print(f"  Encryption: {avg_enc_time:.3f} ms (avg) - Throughput: {(data_size/1024/1024)/(avg_enc_time/1000):.2f} MB/s")
            print(f"  Decryption: {avg_dec_time:.3f} ms (avg) - Throughput: {(data_size/1024/1024)/(avg_dec_time/1000):.2f} MB/s")

def test_ecdh_performance(
    field_bits_list: List[int],
    iterations: int
) -> None:
    """
    Comprehensive testing of ECDH performance with different field sizes.
    
    Args:
        field_bits_list: List of field sizes in bits to test
        iterations: Number of iterations for each test
    """
    print(f"\n===== ECDH Performance Testing =====")
    print(f"Field sizes: {field_bits_list} bits")
    print(f"Iterations: {iterations}")
    
    # Test each field size
    for field_bits in field_bits_list:
        print(f"\n----- Testing ECDH with {field_bits}-bit field -----")
        
        # Generate curve
        curve_gen_times = []
        for _ in range(iterations):
            with Timer("ecdh_curve_generation", global_metrics) as timer:
                curve = test_ecdh_curve_generation(field_bits)
            curve_gen_times.append(timer.elapsed_ms)
        
        avg_curve_gen_time = sum(curve_gen_times) / len(curve_gen_times)
        print(f"Curve Generation: {avg_curve_gen_time:.3f} ms (avg)")
        
        # Generate keypairs
        key_gen_times = []
        alice_keys = []
        bob_keys = []
        
        for _ in range(iterations):
            # Alice generates a keypair
            with Timer("ecdh_key_generation", global_metrics) as timer:
                alice_private, alice_public = test_ecdh_key_generation(curve)
            key_gen_times.append(timer.elapsed_ms)
            
            alice_keys.append((alice_private, alice_public))
            
            # Bob generates a keypair
            with Timer("ecdh_key_generation", global_metrics) as timer:
                bob_private, bob_public = test_ecdh_key_generation(curve)
            key_gen_times.append(timer.elapsed_ms)
            
            bob_keys.append((bob_private, bob_public))
        
        avg_key_gen_time = sum(key_gen_times) / len(key_gen_times)
        print(f"Key Generation: {avg_key_gen_time:.3f} ms (avg)")
        
        # Compute shared secrets
        shared_secret_times = []
        
        for i in range(iterations):
            alice_private, alice_public = alice_keys[i]
            bob_private, bob_public = bob_keys[i]
            
            # Alice computes shared secret
            with Timer("ecdh_shared_secret", global_metrics) as timer:
                alice_secret = test_ecdh_shared_secret(curve, alice_private, bob_public)
            shared_secret_times.append(timer.elapsed_ms)
            
            # Bob computes shared secret
            with Timer("ecdh_shared_secret", global_metrics) as timer:
                bob_secret = test_ecdh_shared_secret(curve, bob_private, alice_public)
            shared_secret_times.append(timer.elapsed_ms)
            
            # Record key size metrics
            global_metrics.record_key_size_metric(field_bits, "shared_secret", shared_secret_times[-2])
            global_metrics.record_key_size_metric(field_bits, "shared_secret", shared_secret_times[-1])
            
            # Verify that the shared secrets match
            if alice_secret != bob_secret:
                print(f"ERROR: Shared secrets don't match in iteration {i}!")
        
        avg_shared_secret_time = sum(shared_secret_times) / len(shared_secret_times)
        print(f"Shared Secret: {avg_shared_secret_time:.3f} ms (avg)")

@measure_memory
def test_secure_socket_memory():
    """Test memory usage during secure socket communication."""
    print(f"\n===== Secure Socket Memory Usage Testing =====")
    
    # Start a secure server
    server = SecureServer(port=12346)
    
    if server.start():
        print("Secure server started.")
        
        try:
            # Wait a moment for the server to initialize
            time.sleep(0.5)
            
            # Create and connect a secure client
            client = SecureClient(port=12346)
            
            # Connect to the server and perform key exchange
            if client.connect_secure():
                print("Secure connection established.")
                
                # Send a series of messages with increasing size
                for i in range(5):
                    message_size = 1024 * (i + 1)
                    message = "X" * message_size
                    
                    if client.send_encrypted(message):
                        print(f"Sent encrypted message of size {message_size} bytes")
                    
                    # Wait for a moment to receive any responses
                    time.sleep(0.5)
                
                # Disconnect the client
                client.disconnect()
                print("Client disconnected.")
            
            else:
                print("Failed to establish secure connection.")
        
        finally:
            # Stop the server
            server.stop()
            print("Server stopped.")
    
    else:
        print("Failed to start secure server.")

def plot_key_size_performance():
    """Generate a plot of key size vs. performance metrics."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plot generation.")
        return
        
    stats = global_metrics.get_key_size_stats()
    
    if not stats:
        print("No key size performance data available.")
        return
    
    # Prepare data for plotting
    key_sizes = sorted(stats.keys())
    operations = set()
    
    for size_stats in stats.values():
        operations.update(size_stats.keys())
    
    operations = sorted(operations)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    for operation in operations:
        means = []
        
        for key_size in key_sizes:
            if operation in stats[key_size]:
                means.append(stats[key_size][operation]['mean'])
            else:
                means.append(0)
        
        plt.plot(key_sizes, means, 'o-', label=operation)
    
    plt.xlabel('Key Size (bits)')
    plt.ylabel('Average Time (ms)')
    plt.title('Performance Impact of Key Size')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig('key_size_performance.png')
    print("\nKey size performance plot saved as 'key_size_performance.png'")

def plot_data_size_performance():
    """Generate a plot of data size vs. performance metrics."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plot generation.")
        return
        
    stats = global_metrics.get_data_size_stats()
    
    if not stats:
        print("No data size performance data available.")
        return
    
    # Prepare data for plotting
    data_sizes = sorted(stats.keys())
    operations = set()
    
    for size_stats in stats.values():
        operations.update(size_stats.keys())
    
    operations = sorted(operations)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    for operation in operations:
        means = []
        
        for data_size in data_sizes:
            if operation in stats[data_size]:
                means.append(stats[data_size][operation]['mean'])
            else:
                means.append(0)
        
        plt.plot([size/1024 for size in data_sizes], means, 'o-', label=operation)
    
    plt.xlabel('Data Size (KB)')
    plt.ylabel('Average Time (ms)')
    plt.title('Performance Impact of Data Size')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig('data_size_performance.png')
    print("\nData size performance plot saved as 'data_size_performance.png'")

def main():
    """Run performance tests and generate reports."""
    parser = argparse.ArgumentParser(description='Performance testing for cryptographic operations')
    parser.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS,
                        help=f'Number of iterations (default: {DEFAULT_ITERATIONS})')
    parser.add_argument('--key-sizes', type=int, nargs='+', default=DEFAULT_KEY_SIZES_BITS,
                        help=f'Key sizes in bits (default: {DEFAULT_KEY_SIZES_BITS})')
    parser.add_argument('--data-sizes', type=int, nargs='+', default=DEFAULT_DATA_SIZES_BYTES,
                        help=f'Data sizes in bytes (default: {DEFAULT_DATA_SIZES_BYTES})')
    parser.add_argument('--field-bits', type=int, nargs='+', default=DEFAULT_FIELD_BITS,
                        help=f'Field sizes in bits for ECDH (default: {DEFAULT_FIELD_BITS})')
    parser.add_argument('--test-aes', action='store_true', help='Run AES performance tests')
    parser.add_argument('--test-ecdh', action='store_true', help='Run ECDH performance tests')
    parser.add_argument('--test-memory', action='store_true', help='Run memory usage tests')
    parser.add_argument('--plot', action='store_true', help='Generate performance plots')
    parser.add_argument('--output', type=str, default='performance_report.json',
                        help='Output file for performance report (default: performance_report.json)')
    
    args = parser.parse_args()
    
    # If no specific tests are selected, run all tests
    if not (args.test_aes or args.test_ecdh or args.test_memory):
        args.test_aes = True
        args.test_ecdh = True
        args.test_memory = True
    
    # Run AES performance tests
    if args.test_aes:
        test_aes_performance(args.key_sizes, args.data_sizes, args.iterations)
    
    # Run ECDH performance tests
    if args.test_ecdh:
        test_ecdh_performance(args.field_bits, args.iterations)
    
    # Run memory usage tests
    if args.test_memory:
        test_secure_socket_memory()
    
    # Generate and save performance report
    report = PerformanceReport.generate_report(global_metrics)
    PerformanceReport.save_report(report, args.output)
    print(f"\nPerformance report saved to {args.output}")
    
    # Print the report summary
    PerformanceReport.print_report(report)
    
    # Generate plots if requested
    if args.plot:
        if MATPLOTLIB_AVAILABLE:
            plot_key_size_performance()
            plot_data_size_performance()
        else:
            print("Error: Matplotlib not available. Plots could not be generated.")

if __name__ == "__main__":
    main() 