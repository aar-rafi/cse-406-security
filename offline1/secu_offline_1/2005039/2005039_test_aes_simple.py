#!/usr/bin/env python3
import os
import binascii
import time
from aes import encrypt_cbc, decrypt_cbc, generate_iv, key_expansion

def test_simple_aes():
    print("==== Simple AES Test ====")
    
    plaintext = b"This is a test message. for AES encryption. 128, 192, 256 bit keys."
    print(f"Plaintext: {plaintext!r}")
    print(f"Plaintext length: {len(plaintext)} bytes")
    
    # a fixed IV
    fixed_iv = bytes([i for i in range(16)])
    
    # each key size with a fixed key 
    # key_128 = bytes([i for i in range(16)])
    # key_192 = bytes([i for i in range(24)])
    # key_256 = bytes([i for i in range(32)])
    key_128 = b'BJET CSE20 Batch'
    key_192 = b'BJET CSE20 BatchExtended'
    key_256 = b'BJET CSE20 Batch#1234567^ Secure'
    
    for key, size_name in [
        (key_128, "AES-128"),
        (key_192, "AES-192"),
        (key_256, "AES-256")
    ]:
        print(f"\n----- Testing {size_name} -----")
        print(f"Key length: {len(key)} bytes")
        print(f"Key: {binascii.hexlify(key).decode()}")
        
        # Test key expansion
        rounds = 10 if len(key) == 16 else (12 if len(key) == 24 else 14)
        round_keys = key_expansion(key, len(key))
        print(f"Number of rounds: {rounds}")
        print(f"Number of round keys: {len(round_keys)}")
        
        # Test encryption
        iv, encrypted = encrypt_cbc(plaintext, key, fixed_iv)
        print(f"IV: {binascii.hexlify(iv).decode()}")
        print(f"Ciphertext: {binascii.hexlify(encrypted).decode()}")
        print(f"Ciphertext length: {len(encrypted)} bytes")
        
        try:
            # Test decryption
            decrypted = decrypt_cbc(encrypted, key, iv)
            print(f"Decrypted: {decrypted!r}")
            print(f"Decryption successful: {plaintext == decrypted}")
        except Exception as e:
            print(f"Decryption error: {e}")
            
            # Debug padding
            if len(encrypted) >= 16:
                last_block = encrypted[-16:]
                decrypted_last_block = decrypt_cbc(last_block, key, bytes(16))  # Zero IV for debugging
                print(f"Last block decrypted: {decrypted_last_block!r}")
                print(f"Last byte (potential padding length): {decrypted_last_block[-1]}")

if __name__ == "__main__":
    test_simple_aes() 