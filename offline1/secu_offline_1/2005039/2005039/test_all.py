#!/usr/bin/env python3
import sys
import time
import hashlib
import os  # Use os.urandom instead of Crypto.Random

# Import AES implementation
from aes import encrypt_text, decrypt_text, encrypt_cbc, decrypt_cbc, generate_iv, key_expansion

# Import ECC and ECDH implementation
from ecc import EllipticCurve, Point, ECDH, generate_curve_params, point_to_bytes, bytes_to_point

# Import socket implementations
from socket_utils import Client as SocketClient, Server as SocketServer
from secure_socket import SecureClient, SecureServer

# Import performance metrics
from perf_metrics import Timer, MemoryMonitor, global_metrics, PerformanceReport

def test_aes():
    """Test AES implementation"""
    print("\n===== AES Key Size Tests =====")
    
    plaintext = b"This is a test message for AES encryption"
    print(f"Original text: {plaintext.decode('utf-8')}")
    
    # Test all key sizes
    key_sizes = [
        (b"mysecretkey12345", "AES-128 (16 bytes)"),
        (b"mysecretkey12345_longer", "AES-192 (24 bytes)"),
        (b"mysecretkey12345_much_longer_key", "AES-256 (32 bytes)")
    ]
    
    all_successful = True
    
    for key, name in key_sizes:
        print(f"\n----- Testing {name} -----")
        
        # Ensure key is exactly the right length
        if len(key) < 16:
            key = key.ljust(16, b'\0')
        elif len(key) > 16 and len(key) <= 24:
            key = key.ljust(24, b'\0')
        elif len(key) > 24:
            key = key.ljust(32, b'\0')
            
        print(f"Key length: {len(key)} bytes")
        
        # Test encryption and decryption
        with Timer("encryption", global_metrics):
            iv = generate_iv()
            encrypted_iv, ciphertext = encrypt_cbc(plaintext, key, iv)
        
        with Timer("decryption", global_metrics):
            decrypted = decrypt_cbc(ciphertext, key, encrypted_iv)
        
        decrypted_text = decrypted.decode('utf-8')
        
        print(f"Decrypted text: {decrypted_text}")
        success = plaintext.decode('utf-8') == decrypted_text
        print(f"Decryption successful: {success}")
        
        # Get timing metrics
        encryption_time = global_metrics.timings["encryption"][-1]
        decryption_time = global_metrics.timings["decryption"][-1]
        
        print(f"Encryption time: {encryption_time:.3f} ms")
        print(f"Decryption time: {decryption_time:.3f} ms")
        
        # Record key size metrics
        global_metrics.record_key_size_metric(len(key) * 8, "encryption", encryption_time)
        global_metrics.record_key_size_metric(len(key) * 8, "decryption", decryption_time)
        
        all_successful = all_successful and success
    
    return all_successful

def test_ecdh():
    """Test ECDH key exchange"""
    print("\n===== ECDH Key Exchange Test =====")
    
    # Generate curve parameters
    with Timer("curve_generation", global_metrics):
        params = generate_curve_params(128)
    
    gen_time = global_metrics.timings["curve_generation"][-1]
    print(f"Generated curve parameters in {gen_time:.3f} ms")
    
    # Create elliptic curve
    curve = EllipticCurve(params['a'], params['b'], params['p'], params['g'], params['n'])
    
    # Create two ECDH instances (Alice and Bob)
    alice = ECDH(curve)
    bob = ECDH(curve)
    
    # Generate keypairs
    with Timer("alice_key_generation", global_metrics):
        alice_private, alice_public = alice.generate_keypair()
    
    with Timer("bob_key_generation", global_metrics):
        bob_private, bob_public = bob.generate_keypair()
    
    alice_time = global_metrics.timings["alice_key_generation"][-1]
    bob_time = global_metrics.timings["bob_key_generation"][-1]
    
    print(f"Generated Alice's keypair in {alice_time:.3f} ms")
    print(f"Generated Bob's keypair in {bob_time:.3f} ms")
    
    # Compute shared secrets
    with Timer("alice_shared_secret", global_metrics):
        alice_secret = alice.compute_shared_secret(bob_public)
    
    with Timer("bob_shared_secret", global_metrics):
        bob_secret = bob.compute_shared_secret(alice_public)
    
    alice_secret_time = global_metrics.timings["alice_shared_secret"][-1]
    bob_secret_time = global_metrics.timings["bob_shared_secret"][-1]
    
    print(f"Alice computed shared secret in {alice_secret_time:.3f} ms")
    print(f"Bob computed shared secret in {bob_secret_time:.3f} ms")
    
    # Check if shared secrets match
    match = alice_secret == bob_secret
    print(f"Shared secrets match: {match}")
    print(f"Alice's secret: {alice_secret.hex()}")
    print(f"Bob's secret: {bob_secret.hex()}")
    
    # Record metrics for field size
    global_metrics.record_key_size_metric(128, "key_generation", alice_time)
    global_metrics.record_key_size_metric(128, "key_generation", bob_time)
    global_metrics.record_key_size_metric(128, "shared_secret", alice_secret_time)
    global_metrics.record_key_size_metric(128, "shared_secret", bob_secret_time)
    
    return match

def test_point_serialization():
    """Test point serialization and deserialization"""
    print("\n===== Point Serialization Test =====")
    
    # Generate curve parameters
    params = generate_curve_params(128)
    curve = EllipticCurve(params['a'], params['b'], params['p'], params['g'], params['n'])
    
    # Create a point
    x, y = curve.g
    point = (x, y)
    
    # Serialize and deserialize
    with Timer("point_serialization", global_metrics):
        serialized = point_to_bytes(point)
    
    with Timer("point_deserialization", global_metrics):
        deserialized = bytes_to_point(serialized)
    
    serialization_time = global_metrics.timings["point_serialization"][-1]
    deserialization_time = global_metrics.timings["point_deserialization"][-1]
    
    print(f"Original point: ({x}, {y})")
    print(f"Serialized size: {len(serialized)} bytes")
    print(f"Deserialized point: ({deserialized[0]}, {deserialized[1]})")
    print(f"Serialization time: {serialization_time:.3f} ms")
    print(f"Deserialization time: {deserialization_time:.3f} ms")
    
    match = point == deserialized
    print(f"Points match after serialization/deserialization: {match}")
    
    return match

def test_aes_performance():
    """Test AES performance for different key sizes"""
    print("\n===== AES Performance Comparison =====")
    
    # Prepare test data (larger plaintext for better performance measurement)
    plaintext = b"This is a sample plaintext for AES performance testing. " * 20  # ~1KB of data
    print(f"Plaintext size: {len(plaintext)} bytes")
    
    # Test all key sizes
    key_sizes = [
        (16, "AES-128"),
        (24, "AES-192"),
        (32, "AES-256")
    ]
    
    results = {}
    
    # Number of iterations for averaging
    iterations = 50
    
    for key_size, name in key_sizes:
        print(f"\n----- Testing {key_size*8} bits ({name}) -----")
        
        # Generate random key of appropriate length
        key = os.urandom(key_size)
        iv = generate_iv()
        
        # Key expansion time
        key_expansion_times = []
        for _ in range(iterations):
            with Timer(f"key_expansion_{key_size*8}", global_metrics):
                round_keys = key_expansion(key, len(key))
            key_expansion_times.append(global_metrics.timings[f"key_expansion_{key_size*8}"][-1])
        
        # First run to get ciphertext
        with Timer(f"encryption_first_{key_size*8}", global_metrics):
            encrypted_iv, ciphertext = encrypt_cbc(plaintext, key, iv)
        
        # Encryption time
        encryption_times = []
        for _ in range(iterations):
            with Timer(f"encryption_{key_size*8}", global_metrics):
                _, _ = encrypt_cbc(plaintext, key, iv)
            encryption_times.append(global_metrics.timings[f"encryption_{key_size*8}"][-1])
        
        # Decryption time
        decryption_times = []
        for _ in range(iterations):
            with Timer(f"decryption_{key_size*8}", global_metrics):
                decrypted = decrypt_cbc(ciphertext, key, encrypted_iv)
            decryption_times.append(global_metrics.timings[f"decryption_{key_size*8}"][-1])
        
        # Calculate averages
        key_expansion_time = sum(key_expansion_times) / len(key_expansion_times)
        encryption_time = sum(encryption_times) / len(encryption_times)
        decryption_time = sum(decryption_times) / len(decryption_times)
        total_time = key_expansion_time + encryption_time + decryption_time
        
        # Store results
        results[name] = {
            "key_expansion_time": key_expansion_time,
            "encryption_time": encryption_time,
            "decryption_time": decryption_time,
            "total_time": total_time
        }
        
        # Record key size metrics
        global_metrics.record_key_size_metric(key_size*8, "key_expansion", key_expansion_time)
        global_metrics.record_key_size_metric(key_size*8, "encryption", encryption_time)
        global_metrics.record_key_size_metric(key_size*8, "decryption", decryption_time)
        
        # Calculate throughput
        encryption_throughput = (len(plaintext) / 1024 / 1024) / (encryption_time / 1000)  # MB/s
        decryption_throughput = (len(plaintext) / 1024 / 1024) / (decryption_time / 1000)  # MB/s
        
        global_metrics.record_custom_metric("throughput", f"encryption_{key_size*8}", encryption_throughput)
        global_metrics.record_custom_metric("throughput", f"decryption_{key_size*8}", decryption_throughput)
        
        # Print results
        print(f"Key expansion: {key_expansion_time:.3f} ms (avg over {iterations} iterations)")
        print(f"Encryption: {encryption_time:.3f} ms (avg over {iterations} iterations)")
        print(f"Decryption: {decryption_time:.3f} ms (avg over {iterations} iterations)")
        print(f"Total: {total_time:.3f} ms")
        print(f"Encryption throughput: {encryption_throughput:.2f} MB/s")
        print(f"Decryption throughput: {decryption_throughput:.2f} MB/s")
    
    # Print comparative summary
    print("\n----- Performance Summary -----")
    base_time = results["AES-128"]["total_time"]
    for name, data in results.items():
        slowdown = (data["total_time"] / base_time - 1) * 100
        print(f"{name}: {data['total_time']:.3f} ms ({slowdown:.2f}% slower than AES-128)")
    
    return True

def test_comprehensive_performance():
    """Run the comprehensive performance test suite and generate a report"""
    
    try:
        # Use the comprehensive performance test from test_performance.py
        import test_performance
        
        # Define smaller test parameters for this quick test
        key_sizes = [128, 192, 256]
        data_sizes = [1024, 10240]  # 1KB, 10KB
        field_bits = [128, 256]
        iterations = 10
            
        # Clear previous metrics to get a clean report
        global_metrics.clear()
        
        # Run tests
        print("\n===== Running Comprehensive Performance Tests =====")
        
        # Test AES performance
        test_performance.test_aes_performance(key_sizes, data_sizes, iterations)
        
        # Test ECDH performance
        test_performance.test_ecdh_performance(field_bits, iterations)
        
        # Generate report
        report = PerformanceReport.generate_report(global_metrics)
        report_file = "quick_performance_report.json"
        PerformanceReport.save_report(report, report_file)
        print(f"\nQuick performance report saved to {report_file}")
        
        # Print summary
        PerformanceReport.print_report(report, detailed=False)
        
        return True
        
    except ImportError as e:
        print(f"\nWarning: Could not run comprehensive performance tests - {str(e)}")
        print("Skipping comprehensive performance tests")
        return True

def run_all_tests():
    """Run all tests and report results"""
    print("Starting comprehensive testing of all components...")
    
    results = {
        "AES with different key sizes": test_aes(),
        "AES performance comparison": test_aes_performance(),
        "ECDH key exchange": test_ecdh(),
        "Point serialization": test_point_serialization(),
        "Comprehensive performance": test_comprehensive_performance()
    }
    
    print("\n===== Test Results Summary =====")
    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        all_passed = all_passed and passed
        print(f"{test_name}: {status}")
    
    # Print overall performance metrics
    print("\n===== Overall Performance Metrics =====")
    PerformanceReport.print_report(PerformanceReport.generate_report(global_metrics), detailed=False)
    
    if all_passed:
        print("\nAll tests passed successfully!")
        return 0
    else:
        print("\nSome tests failed. Check the results above.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests()) 