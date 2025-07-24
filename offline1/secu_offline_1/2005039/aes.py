#!/usr/bin/env python3
"""
AES-128 Implementation with CBC Mode and PKCS#7 Padding
"""

import time
import os
import binascii
from typing import List, Tuple, Union, Optional

# AES Block size in bytes (always 16 for AES)
AES_BLOCK_SIZE = 16

# Number of rounds for different key sizes
ROUNDS_BY_KEY_SIZE = {
    16: 10,  # 128-bit key
    24: 12,  # 192-bit key
    32: 14   # 256-bit key
}

# Pre-computed S-box and Inverse S-box tables
# Values from FIPS-197: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197.pdf
SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

INV_SBOX = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
]

RCON = [
    0x00000000, # Not used (index 0)
    0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000,
    0x20000000, 0x40000000, 0x80000000, 0x1b000000, 0x36000000
]

# Byte manipulation utility functions

def bytes_to_matrix(byte_array: bytes) -> List[List[int]]:

    matrix = [list(byte_array[i:i+4]) for i in range(0, len(byte_array), 4)]
    return matrix

def matrix_to_bytes(matrix: List[List[int]]) -> bytes:

    result = bytearray(AES_BLOCK_SIZE)
    for i in range(4):
        for j in range(4):
            result[i*4 + j] = matrix[i][j]
    return bytes(result)

def xor_bytes(a: bytes, b: bytes) -> bytes:

    return bytes(x ^ y for x, y in zip(a, b))

# State manipulation helper functions

def sub_bytes(state: List[List[int]]) -> List[List[int]]:

    for i in range(4):
        for j in range(4):
            state[i][j] = SBOX[state[i][j]]
    return state

def inv_sub_bytes(state: List[List[int]]) -> List[List[int]]:

    for i in range(4):
        for j in range(4):
            state[i][j] = INV_SBOX[state[i][j]]
    return state

def shift_rows(state: List[List[int]]) -> List[List[int]]:

    # Create a copy of the state to avoid in-place modification
    result = [row[:] for row in state]
    
    # Row 0: No shift
    # Row 1: Shift left by 1
    result[1] = state[1][1:] + state[1][:1]
    # Row 2: Shift left by 2
    result[2] = state[2][2:] + state[2][:2]
    # Row 3: Shift left by 3
    result[3] = state[3][3:] + state[3][:3]
    
    return result

def inv_shift_rows(state: List[List[int]]) -> List[List[int]]:

    # Create a copy of the state to avoid in-place modification
    result = [row[:] for row in state]
    
    # Row 0: No shift
    # Row 1: Shift right by 1
    result[1] = state[1][-1:] + state[1][:-1]
    # Row 2: Shift right by 2
    result[2] = state[2][-2:] + state[2][:-2]
    # Row 3: Shift right by 3
    result[3] = state[3][-3:] + state[3][:-3]
    
    return result

# Galois Field operations for MixColumns
def gf_multiply(a: int, b: int) -> int:

    product = 0
    for _ in range(8):
        if b & 1:
            product ^= a
        a <<= 1
        if a & 0x100:
            a ^= 0x11b  # x^8 + x^4 + x^3 + x + 1 (irreducible polynomial for AES)
        b >>= 1
    return product & 0xFF

# Tools for testing and debugging
def print_hex(data: bytes) -> None:

    print(' '.join(f'{byte:02x}' for byte in data))

def print_state(state: List[List[int]]) -> None:

    for row in state:
        print(' '.join(f'{byte:02x}' for byte in row))
    print()


def mix_columns(state: List[List[int]]) -> List[List[int]]:

    result = [row[:] for row in state]
    
    for i in range(4):
        s0 = state[0][i]
        s1 = state[1][i]
        s2 = state[2][i]
        s3 = state[3][i]
        
        # Perform the matrix multiplication for MixColumns
        result[0][i] = gf_multiply(0x02, s0) ^ gf_multiply(0x03, s1) ^ s2 ^ s3
        result[1][i] = s0 ^ gf_multiply(0x02, s1) ^ gf_multiply(0x03, s2) ^ s3
        result[2][i] = s0 ^ s1 ^ gf_multiply(0x02, s2) ^ gf_multiply(0x03, s3)
        result[3][i] = gf_multiply(0x03, s0) ^ s1 ^ s2 ^ gf_multiply(0x02, s3)
    
    return result

def inv_mix_columns(state: List[List[int]]) -> List[List[int]]:

    result = [row[:] for row in state]
    
    for i in range(4):
        s0 = state[0][i]
        s1 = state[1][i]
        s2 = state[2][i]
        s3 = state[3][i]
        
        # Perform the matrix multiplication for InvMixColumns
        result[0][i] = gf_multiply(0x0e, s0) ^ gf_multiply(0x0b, s1) ^ gf_multiply(0x0d, s2) ^ gf_multiply(0x09, s3)
        result[1][i] = gf_multiply(0x09, s0) ^ gf_multiply(0x0e, s1) ^ gf_multiply(0x0b, s2) ^ gf_multiply(0x0d, s3)
        result[2][i] = gf_multiply(0x0d, s0) ^ gf_multiply(0x09, s1) ^ gf_multiply(0x0e, s2) ^ gf_multiply(0x0b, s3)
        result[3][i] = gf_multiply(0x0b, s0) ^ gf_multiply(0x0d, s1) ^ gf_multiply(0x09, s2) ^ gf_multiply(0x0e, s3)
    
    return result

def add_round_key(state: List[List[int]], round_key: List[List[int]]) -> List[List[int]]:

    result = [row[:] for row in state]
    
    for i in range(4):
        for j in range(4):
            result[i][j] ^= round_key[i][j]
    
    return result

def rot_word(word: List[int]) -> List[int]:

    return word[1:] + word[:1]

def sub_word(word: List[int]) -> List[int]:

    return [SBOX[byte] for byte in word]

def key_expansion(key: bytes, key_length: int = 16) -> List[List[List[int]]]:

    # Determine the number of rounds based on key length
    num_rounds = ROUNDS_BY_KEY_SIZE[key_length]
    
    # Get the number of 32-bit words in the key
    num_words = key_length // 4
    
    # Initialize the expanded key schedule
    w = [[] for _ in range(4 * (num_rounds + 1))]
    
    # For the first num_words entries, convert key bytes to words
    for i in range(num_words):
        w[i] = [key[4*i], key[4*i+1], key[4*i+2], key[4*i+3]]
    
    # Generate the remaining words
    for i in range(num_words, 4 * (num_rounds + 1)):
        temp = w[i-1][:]  # Copy the previous word
        
        if i % num_words == 0:
            # Apply RotWord, SubWord, and XOR with Rcon
            temp = sub_word(rot_word(temp))
            temp[0] ^= (RCON[i // num_words] >> 24) & 0xFF  # Extract first byte of Rcon
        elif num_words > 6 and i % num_words == 4:
            # Additional transformation for AES-256
            temp = sub_word(temp)
        
        # XOR with the word num_words positions earlier
        w[i] = [temp[j] ^ w[i - num_words][j] for j in range(4)]
    
    # Convert flat list of words to a list of round keys (each a 4x4 matrix)
    round_keys = []
    for i in range(0, len(w), 4):
        # Transpose 4 consecutive words to form a round key matrix
        round_key = [[0 for _ in range(4)] for _ in range(4)]
        for j in range(4):
            for k in range(4):
                round_key[k][j] = w[i + j][k]
        round_keys.append(round_key)
    
    return round_keys

def encrypt_block(block: bytes, round_keys: List[List[List[int]]]) -> bytes:

    # Convert block to state matrix
    state = bytes_to_matrix(block)
    
    # Round 0: Just add round key
    state = add_round_key(state, round_keys[0])
    
    # Main rounds (1 to num_rounds-1)
    for i in range(1, len(round_keys) - 1):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, round_keys[i])
    
    # Final round (no mix columns)
    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, round_keys[-1])
    
    # Convert state back to bytes
    return matrix_to_bytes(state)

def decrypt_block(block: bytes, round_keys: List[List[List[int]]]) -> bytes:
    # Convert block to state matrix
    state = bytes_to_matrix(block)
    
    # Start with final round key
    state = add_round_key(state, round_keys[-1])
    
    # Main rounds (reverse order)
    for i in range(len(round_keys) - 2, 0, -1):
        state = inv_shift_rows(state)
        state = inv_sub_bytes(state)
        state = add_round_key(state, round_keys[i])
        state = inv_mix_columns(state)
    
    # Final round (just inverse of initial round)
    state = inv_shift_rows(state)
    state = inv_sub_bytes(state)
    state = add_round_key(state, round_keys[0])
    
    # Convert state back to bytes
    return matrix_to_bytes(state)

def pkcs7_pad(data: bytes, block_size: int = AES_BLOCK_SIZE) -> bytes:

    pad_len = block_size - (len(data) % block_size)
    padding = bytes([pad_len] * pad_len)
    return data + padding

def pkcs7_unpad(data: bytes) -> bytes:

    if not data:
        return data
    
    pad_len = data[-1]
    
    # Validate padding
    if pad_len == 0 or pad_len > AES_BLOCK_SIZE:
        raise ValueError("Invalid PKCS#7 padding")
    
    # Check that all padding bytes are correct
    for i in range(1, pad_len + 1):
        if data[-i] != pad_len:
            raise ValueError("Invalid PKCS#7 padding")
    
    return data[:-pad_len]

def generate_iv() -> bytes:

    return os.urandom(AES_BLOCK_SIZE)

def encrypt_cbc(data: bytes, key: bytes, iv: Optional[bytes] = None) -> Tuple[bytes, bytes]:

    if iv is None:
        iv = generate_iv()
    
    # Pad the data to a multiple of the block size
    padded_data = pkcs7_pad(data)
    
    # Generate round keys from the encryption key
    round_keys = key_expansion(key, len(key))
    
    # Initialize result with IV
    ciphertext = bytearray()
    
    # Initialize previous block for CBC
    prev_block = iv
    
    # Process each block
    for i in range(0, len(padded_data), AES_BLOCK_SIZE):
        block = padded_data[i:i+AES_BLOCK_SIZE]
        
        # XOR with previous ciphertext block (or IV for first block)
        block = xor_bytes(block, prev_block)
        
        # Encrypt the block
        encrypted_block = encrypt_block(block, round_keys)
        
        # Add to ciphertext
        ciphertext.extend(encrypted_block)
        
        # Update previous block for next iteration
        prev_block = encrypted_block
    
    return iv, bytes(ciphertext)

def decrypt_cbc(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:

    # Validate inputs
    if len(ciphertext) % AES_BLOCK_SIZE != 0:
        raise ValueError("Ciphertext length must be a multiple of the block size")
    
    # Generate round keys from the encryption key
    round_keys = key_expansion(key, len(key))
    
    # Initialize result
    plaintext = bytearray()
    
    # Initialize previous block for CBC
    prev_block = iv
    
    # Process each block
    for i in range(0, len(ciphertext), AES_BLOCK_SIZE):
        block = ciphertext[i:i+AES_BLOCK_SIZE]
        
        # Decrypt the block
        decrypted_block = decrypt_block(block, round_keys)
        
        # XOR with previous ciphertext block (or IV for first block)
        decrypted_block = xor_bytes(decrypted_block, prev_block)
        
        # Add to plaintext
        plaintext.extend(decrypted_block)
        
        # Update previous block for next iteration
        prev_block = block
    
    # Remove padding
    return pkcs7_unpad(plaintext)

# Add a helper function to detect key size
def detect_key_size(key_bytes: bytes) -> int:

    key_len = len(key_bytes)
    
    if key_len <= 16:
        return 16  # AES-128
    elif key_len <= 24:
        return 24  # AES-192
    else:
        return 32  # AES-256

def encrypt_text(text: str, key: str, key_size: int = None) -> Tuple[bytes, bytes]:

    # Convert text to bytes
    data = text.encode('utf-8')
    
    # Process key
    key_bytes = key.encode('utf-8')
    
    # Determine key size if not specified
    if key_size is None:
        key_size = detect_key_size(key_bytes)
    
    # Validate key size
    if key_size not in [16, 24, 32]:
        raise ValueError("Key size must be 16, 24, or 32 bytes (128, 192, or 256 bits)")
    
    # Handle key of any length by padding or truncating
    if len(key_bytes) < key_size:
        # Pad short keys with zeros
        key_bytes = key_bytes + b'\x00' * (key_size - len(key_bytes))
    elif len(key_bytes) > key_size:
        # Truncate long keys
        key_bytes = key_bytes[:key_size]
    
    # Encrypt data
    return encrypt_cbc(data, key_bytes)

def decrypt_text(ciphertext: bytes, key: str, iv: bytes, key_size: int = None) -> str:

    # Process key
    key_bytes = key.encode('utf-8')
    
    # Determine key size if not specified
    if key_size is None:
        key_size = detect_key_size(key_bytes)
    
    # Validate key size
    if key_size not in [16, 24, 32]:
        raise ValueError("Key size must be 16, 24, or 32 bytes (128, 192, or 256 bits)")
    
    # Handle key of any length by padding or truncating
    if len(key_bytes) < key_size:
        # Pad short keys with zeros
        key_bytes = key_bytes + b'\x00' * (key_size - len(key_bytes))
    elif len(key_bytes) > key_size:
        # Truncate long keys
        key_bytes = key_bytes[:key_size]
    
    # Decrypt data
    plaintext_bytes = decrypt_cbc(ciphertext, key_bytes, iv)
    
    # Convert back to string
    return plaintext_bytes.decode('utf-8')

def encrypt_ctr(data: bytes, key: bytes, nonce: Optional[bytes] = None) -> Tuple[bytes, bytes]:

    if nonce is None:
        nonce = os.urandom(8)  # 8 bytes for nonce, 8 for counter
    if len(nonce) != 8:
        raise ValueError("Nonce must be 8 bytes for AES-CTR")
    
    round_keys = key_expansion(key, len(key))
    ciphertext = bytearray()
    block_size = AES_BLOCK_SIZE
    counter = 0
    
    for i in range(0, len(data), block_size):
        block = data[i:i+block_size]
        # Construct counter block: nonce (8 bytes) + counter (8 bytes, big endian)
        counter_block = nonce + counter.to_bytes(8, byteorder='big')
        keystream = encrypt_block(counter_block, round_keys)
        ciphertext.extend(xor_bytes(block, keystream[:len(block)]))
        counter += 1
    return nonce, bytes(ciphertext)

def decrypt_ctr(ciphertext: bytes, key: bytes, nonce: bytes) -> bytes:

    if len(nonce) != 8:
        raise ValueError("Nonce must be 8 bytes for AES-CTR")
    round_keys = key_expansion(key, len(key))
    plaintext = bytearray()
    block_size = AES_BLOCK_SIZE
    counter = 0
    for i in range(0, len(ciphertext), block_size):
        block = ciphertext[i:i+block_size]
        counter_block = nonce + counter.to_bytes(8, byteorder='big')
        keystream = encrypt_block(counter_block, round_keys)
        plaintext.extend(xor_bytes(block, keystream[:len(block)]))
        counter += 1
    return bytes(plaintext)

if __name__ == "__main__":
    # Simple test of the utility functions
    test_data = bytes.fromhex('00112233445566778899aabbccddeeff')
    print("Original bytes:")
    print_hex(test_data)
    
    # Convert to state matrix
    state = bytes_to_matrix(test_data)
    print("\nAs state matrix:")
    print_state(state)
    
    # Apply SubBytes
    state = sub_bytes(state)
    print("After SubBytes:")
    print_state(state)
    
    # Apply ShiftRows
    state = shift_rows(state)
    print("After ShiftRows:")
    print_state(state)
    
    # Apply MixColumns
    state = mix_columns(state)
    print("After MixColumns:")
    print_state(state)
    
    # Test key expansion
    test_key = bytes.fromhex('000102030405060708090a0b0c0d0e0f')
    round_keys = key_expansion(test_key)
    
    print("\nKey expansion test (AES-128):")
    print("Original key:")
    print_state(bytes_to_matrix(test_key))
    
    print("Round 0 key (initial key):")
    print_state(round_keys[0])
    
    print("Round 1 key:")
    print_state(round_keys[1])
    
    print("Final round key:")
    print_state(round_keys[-1])
    
    # Test encryption and decryption of a single block
    original_block = bytes.fromhex('00112233445566778899aabbccddeeff')
    print("\nEncryption/Decryption Test:")
    print("Original block:")
    print_hex(original_block)
    
    # Encrypt block
    encrypted_block = encrypt_block(original_block, round_keys)
    print("Encrypted block:")
    print_hex(encrypted_block)
    
    # Decrypt block
    decrypted_block = decrypt_block(encrypted_block, round_keys)
    print("Decrypted block:")
    print_hex(decrypted_block)
    
    # Test PKCS#7 padding
    test_data = b'Hello, World!'
    padded_data = pkcs7_pad(test_data)
    print("\nPKCS#7 Padding Test:")
    print(f"Original data ({len(test_data)} bytes): {test_data}")
    print(f"Padded data ({len(padded_data)} bytes): {padded_data}")
    print(f"Padded data (hex): {padded_data.hex()}")
    
    unpadded_data = pkcs7_unpad(padded_data)
    print(f"Unpadded data ({len(unpadded_data)} bytes): {unpadded_data}")
    
    # Test CBC mode encryption and decryption
    test_data = b'This is a test message for AES-CBC encryption with PKCS#7 padding.'
    test_key = b'ABCDEFGHIJKLMNOP'  # 16-byte key (AES-128)
    test_iv = bytes.fromhex('00112233445566778899aabbccddeeff')
    
    print("\nAES-CBC Mode Test:")
    print(f"Original data: {test_data}")
    
    start_time = time.time()
    iv, encrypted_data = encrypt_cbc(test_data, test_key, test_iv)
    encryption_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    print(f"IV: {iv.hex()}")
    print(f"Encrypted data: {encrypted_data.hex()}")
    
    start_time = time.time()
    decrypted_data = decrypt_cbc(encrypted_data, test_key, iv)
    decryption_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    print(f"Decrypted data: {decrypted_data}")
    
    print(f"Encryption time: {encryption_time:.6f} ms")
    print(f"Decryption time: {decryption_time:.6f} ms")
    
    # Test text encryption and decryption
    test_text = "We need picnic"
    test_key = "BJET CSE20 Batch"
    
    print("\nText Encryption Test:")
    print(f"Original text: {test_text}")
    print(f"Key: {test_key}")
    
    # Print key in ASCII and HEX
    key_bytes = test_key.encode('utf-8')
    print("Key:")
    print(f"In ASCII: {test_key}")
    print(f"In HEX: {' '.join(f'{b:02X}' for b in key_bytes)}")
    
    # Print plaintext in ASCII and HEX
    plaintext_bytes = test_text.encode('utf-8')
    print("Plain Text:")
    print(f"In ASCII: {test_text}")
    print(f"In HEX: {' '.join(f'{b:02x}' for b in plaintext_bytes)}")
    
    # Pad the plaintext
    padded_plaintext = pkcs7_pad(plaintext_bytes)
    print(f"In ASCII (After Padding): {padded_plaintext.decode('utf-8', errors='replace')}")
    print(f"In HEX (After Padding): {' '.join(f'{b:02x}' for b in padded_plaintext)}")
    
    # Encrypt
    start_key_time = time.time()
    round_keys = key_expansion(key_bytes[:16])
    key_time = (time.time() - start_key_time) * 1000
    
    start_enc_time = time.time()
    iv, ciphertext = encrypt_text(test_text, test_key)
    enc_time = (time.time() - start_enc_time) * 1000
    
    # Print the full ciphertext (IV + actual ciphertext)
    full_ciphertext = iv + ciphertext
    print("Ciphered Text:")
    print(f"In HEX: {' '.join(f'{b:02x}' for b in full_ciphertext)}")
    print(f"In ASCII: {full_ciphertext.decode('latin1', errors='replace')}")
    
    # Decrypt
    start_dec_time = time.time()
    decrypted_text = decrypt_text(ciphertext, test_key, iv)
    dec_time = (time.time() - start_dec_time) * 1000
    
    print("Deciphered Text:")
    print("Before Unpadding:")
    padded_bytes = pkcs7_pad(test_text.encode('utf-8'))
    print(f"In HEX: {' '.join(f'{b:02x}' for b in padded_bytes)}")
    print(f"In ASCII: {padded_bytes.decode('utf-8', errors='replace')}")
    
    print("After Unpadding:")
    print(f"In ASCII: {decrypted_text}")
    print(f"In HEX: {' '.join(f'{b:02x}' for b in decrypted_text.encode('utf-8'))}")
    
    print("Execution Time Details:")
    print(f"Key Schedule Time: {key_time:.15f} ms")
    print(f"Encryption Time: {enc_time:.15f} ms")
    print(f"Decryption Time: {dec_time:.15f} ms") 