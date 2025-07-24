"""
AES-128 Implementation with CBC Mode and PKCS#7 Padding

Usage:
    python 2005039_aes.py --encrypt --input 'Hello, World!' --key 'BJET CSE20 Batch' --verbose
    python 2005039_aes.py --decrypt --input 'c0 05 42907053f44e2ba6330b6d319f4175c3ac608efab473bcc91a3608b9c873' --key 'BJET CSE20 Batch'
"""

import sys
import time
import argparse
import binascii
import os

# Import our AES implementation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from aes import (encrypt_text, decrypt_text, pkcs7_pad, pkcs7_unpad, 
                key_expansion, AES_BLOCK_SIZE)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AES-128 Implementation with CBC Mode and PKCS#7 Padding"
    )
    
    # Action group (encrypt or decrypt)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--encrypt", action="store_true", help="Encrypt the input text")
    action.add_argument("--decrypt", action="store_true", help="Decrypt the input text")
    
    # Required arguments
    parser.add_argument("--input", required=True, help="Input text to encrypt/decrypt")
    parser.add_argument("--key", required=True, help="Encryption/decryption key (16 chars for AES-128)")
    
    # Optional arguments
    parser.add_argument("--iv", help="Initialization vector for CBC mode (hexadecimal, 32 chars). Only used for decryption")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    return parser.parse_args()

def encrypt(input_text, key, verbose=False):
    """
    Encrypt input text using AES-CBC with PKCS#7 padding.
    
    Args:
        input_text: Plain text to encrypt
        key: Encryption key
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (IV, ciphertext) in hexadecimal format
    """
    if verbose:
        print("Encrypting text...")
        print(f"Input text: '{input_text}'")
        print(f"Key: '{key}'")
    
    # Print key in ASCII and HEX
    key_bytes = key.encode('utf-8')
    print("Key:")
    print(f"In ASCII: {key}")
    print(f"In HEX: {' '.join(f'{b:02X}' for b in key_bytes)}")
    
    # Print plaintext in ASCII and HEX
    plaintext_bytes = input_text.encode('utf-8')
    print("Plain Text:")
    print(f"In ASCII: {input_text}")
    print(f"In HEX: {' '.join(f'{b:02x}' for b in plaintext_bytes)}")
    
    # Print padded plaintext
    padded_plaintext = pkcs7_pad(plaintext_bytes)
    print(f"In ASCII (After Padding): {padded_plaintext.decode('utf-8', errors='replace')}")
    print(f"In HEX (After Padding): {' '.join(f'{b:02x}' for b in padded_plaintext)}")
    
    # Perform encryption with timing
    start_key_time = time.time()
    round_keys = key_expansion(key_bytes[:16])
    key_time = (time.time() - start_key_time) * 1000
    
    start_enc_time = time.time()
    iv, ciphertext = encrypt_text(input_text, key)
    enc_time = (time.time() - start_enc_time) * 1000
    
    # Print ciphertext (IV + actual ciphertext)
    full_ciphertext = iv + ciphertext
    print("Ciphered Text:")
    print(f"In HEX: {' '.join(f'{b:02x}' for b in full_ciphertext)}")
    print(f"In ASCII: {full_ciphertext.decode('latin1', errors='replace')}")
    
    # Print timing information
    print("Execution Time Details:")
    print(f"Key Schedule Time: {key_time:.15f} ms")
    print(f"Encryption Time: {enc_time:.15f} ms")
    
    return iv, ciphertext

def decrypt(input_hex, key, iv=None, verbose=False):
    """
    Decrypt input ciphertext using AES-CBC with PKCS#7 padding.
    
    Args:
        input_hex: Hex string of ciphertext (can include IV as first 16 bytes)
        key: Decryption key
        iv: Optional IV in hex format. If None, first 16 bytes of input are used as IV.
        verbose: Whether to print verbose output
        
    Returns:
        Decrypted plaintext
    """
    if verbose:
        print("Decrypting text...")
        print(f"Input (hex): {input_hex}")
        print(f"Key: '{key}'")
    
    # Convert input hex string to bytes
    # Remove spaces if present
    input_hex = input_hex.replace(" ", "")
    try:
        input_bytes = bytes.fromhex(input_hex)
    except ValueError:
        print("Error: Input must be a valid hexadecimal string")
        sys.exit(1)
    
    # Check if IV was provided, otherwise extract from input
    if iv is None:
        # First 16 bytes are IV, rest is ciphertext
        if len(input_bytes) < AES_BLOCK_SIZE * 2:
            print("Error: Input must be at least 32 bytes (16 for IV + 16 for ciphertext)")
            sys.exit(1)
        iv = input_bytes[:AES_BLOCK_SIZE]
        ciphertext = input_bytes[AES_BLOCK_SIZE:]
    else:
        # IV was provided separately
        try:
            iv = bytes.fromhex(iv.replace(" ", ""))
            ciphertext = input_bytes
        except ValueError:
            print("Error: IV must be a valid hexadecimal string")
            sys.exit(1)
    
    # Print key in ASCII and HEX
    key_bytes = key.encode('utf-8')
    print("Key:")
    print(f"In ASCII: {key}")
    print(f"In HEX: {' '.join(f'{b:02X}' for b in key_bytes)}")
    
    # Perform decryption with timing
    start_dec_time = time.time()
    decrypted_text = decrypt_text(ciphertext, key, iv)
    dec_time = (time.time() - start_dec_time) * 1000
    
    # Print decrypted text
    print("Deciphered Text:")
    print("Before Unpadding:")
    padded_bytes = pkcs7_pad(decrypted_text.encode('utf-8'))
    print(f"In HEX: {' '.join(f'{b:02x}' for b in padded_bytes)}")
    print(f"In ASCII: {padded_bytes.decode('utf-8', errors='replace')}")
    
    print("After Unpadding:")
    print(f"In ASCII: {decrypted_text}")
    print(f"In HEX: {' '.join(f'{b:02x}' for b in decrypted_text.encode('utf-8'))}")
    
    # Print timing information
    print("Execution Time Details:")
    print(f"Decryption Time: {dec_time:.15f} ms")
    
    return decrypted_text

def main():
    """Main function."""
    args = parse_args()
    
    if args.encrypt:
        # Encrypt the input text
        encrypt(args.input, args.key, args.verbose)
    
    elif args.decrypt:
        # Decrypt the input hex
        decrypt(args.input, args.key, args.iv, args.verbose)

if __name__ == "__main__":
    main() 