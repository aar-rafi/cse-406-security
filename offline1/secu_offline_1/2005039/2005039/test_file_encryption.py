#!/usr/bin/env python3
"""
Test File Encryption and Decryption

This script tests the file encryption and decryption functionality
with various file types and sizes.
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path

from file_crypto import (
    encrypt_file, decrypt_file, verify_file_integrity,
    get_encryption_info, calculate_file_hash, FileEncryptionError
)

# Directory for test files
TEST_DIR = "test_files"
ENCRYPTED_DIR = "encrypted_files"
DECRYPTED_DIR = "decrypted_files"

def ensure_directories():
    """Create test directories if they don't exist."""
    for directory in [TEST_DIR, ENCRYPTED_DIR, DECRYPTED_DIR]:
        os.makedirs(directory, exist_ok=True)

def create_test_text_file(filename, size_kb):
    """Create a text file of approximately the specified size."""
    filepath = os.path.join(TEST_DIR, filename)
    
    # Generate random text
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 \n"
    text = ""
    
    # Approximate the number of characters needed
    target_size = size_kb * 1024
    chunk_size = min(target_size, 10240)  # 10KB chunks
    
    with open(filepath, 'w') as f:
        while f.tell() < target_size:
            # Generate a chunk of random text
            chunk = ''.join(random.choice(chars) for _ in range(chunk_size))
            f.write(chunk)
    
    print(f"Created text file: {filepath} ({os.path.getsize(filepath) / 1024:.2f} KB)")
    return filepath

def create_test_binary_file(filename, size_kb):
    """Create a binary file of approximately the specified size."""
    filepath = os.path.join(TEST_DIR, filename)
    
    # Generate random binary data
    with open(filepath, 'wb') as f:
        remaining = size_kb * 1024
        chunk_size = min(remaining, 10240)  # 10KB chunks
        
        while remaining > 0:
            # Generate a chunk of random binary data
            chunk = os.urandom(min(chunk_size, remaining))
            f.write(chunk)
            remaining -= len(chunk)
    
    print(f"Created binary file: {filepath} ({os.path.getsize(filepath) / 1024:.2f} KB)")
    return filepath

def test_file_encryption(input_file, key):
    """Test encryption and decryption of a file in both CBC and CTR modes."""
    filename = os.path.basename(input_file)
    file_size_kb = os.path.getsize(input_file) / 1024
    for mode in ["cbc", "ctr"]:
        encrypted_file = os.path.join(ENCRYPTED_DIR, f"{filename}.{mode}.enc")
        decrypted_file = os.path.join(DECRYPTED_DIR, f"dec_{filename}.{mode}")
        print(f"\nTesting encryption of {filename} ({file_size_kb:.2f} KB) in {mode.upper()} mode")
        # Encrypt the file
        print(f"Encrypting...")
        start_time = time.time()
        try:
            metadata = encrypt_file(input_file, encrypted_file, key, mode=mode)
            encrypt_time = time.time() - start_time
            encrypted_size_kb = os.path.getsize(encrypted_file) / 1024
            print(f"File encrypted in {encrypt_time:.2f} seconds")
            print(f"Original size: {file_size_kb:.2f} KB, Encrypted size: {encrypted_size_kb:.2f} KB")
            print(f"Encryption speed: {file_size_kb / encrypt_time:.2f} KB/s")
            print("Metadata stored with encrypted file:")
            for k, value in metadata.items():
                if k == "iv" or k == "nonce":
                    print(f"  {k}: {value[:16]}...")
                elif isinstance(value, str) and len(value) > 60:
                    print(f"  {k}: {value[:60]}...")
                else:
                    print(f"  {k}: {value}")
            # Decrypt the file
            print(f"\nDecrypting...")
            start_time = time.time()
            decrypted_metadata = decrypt_file(encrypted_file, decrypted_file, key)
            decrypt_time = time.time() - start_time
            print(f"File decrypted in {decrypt_time:.2f} seconds")
            print(f"Decryption speed: {file_size_kb / decrypt_time:.2f} KB/s")
            # Verify integrity
            original_hash = calculate_file_hash(input_file)
            decrypted_hash = calculate_file_hash(decrypted_file)
            print(f"\nIntegrity check:")
            print(f"Original hash: {original_hash[:16]}...")
            print(f"Decrypted hash: {decrypted_hash[:16]}...")
            if original_hash == decrypted_hash:
                print("✅ Files match - Successful decryption!")
            else:
                print("❌ Files differ - Decryption failed!")
        except FileEncryptionError as e:
            print(f"Error: {str(e)}")

def main():
    """Run the file encryption test suite."""
    parser = argparse.ArgumentParser(description="Test file encryption with various file types and sizes")
    parser.add_argument("--key", default="TestSecretKey123", help="Encryption key to use")
    parser.add_argument("--small", action="store_true", help="Run only with small files (quick test)")
    parser.add_argument("--large", action="store_true", help="Include large file tests (can be slow)")
    args = parser.parse_args()
    
    # Ensure test directories exist
    ensure_directories()
    
    # Define file sizes to test (in KB)
    if args.small:
        text_sizes = [10, 100]
        binary_sizes = [10, 100]
    elif args.large:
        text_sizes = [10, 100, 1000, 10000]
        binary_sizes = [10, 100, 1000, 10000]
    else:
        text_sizes = [10, 100, 1000]
        binary_sizes = [10, 100, 1000]
    
    print(f"Running file encryption tests with {args.key=}")
    
    # Test text files
    for size_kb in text_sizes:
        filename = f"text_{size_kb}kb.txt"
        input_file = create_test_text_file(filename, size_kb)
        test_file_encryption(input_file, args.key)
    
    # Test binary files
    for size_kb in binary_sizes:
        filename = f"binary_{size_kb}kb.bin"
        input_file = create_test_binary_file(filename, size_kb)
        test_file_encryption(input_file, args.key)
    
    # Test a real-world scenario with an image if available
    sample_image = "sample.jpg"
    if os.path.exists(sample_image):
        copied_image = os.path.join(TEST_DIR, "sample.jpg")
        with open(sample_image, "rb") as src, open(copied_image, "wb") as dst:
            dst.write(src.read())
        test_file_encryption(copied_image, args.key)

if __name__ == "__main__":
    main() 