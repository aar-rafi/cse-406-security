import os
import sys
import json
import hashlib
import binascii
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Generator, BinaryIO
import concurrent.futures

from aes import (
    encrypt_cbc, decrypt_cbc, pkcs7_pad, pkcs7_unpad, 
    generate_iv, detect_key_size, AES_BLOCK_SIZE,
    encrypt_ctr, decrypt_ctr, key_expansion, encrypt_block, xor_bytes
)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB

# Metadata keys
META_FILENAME = "filename"
META_ORIGINAL_SIZE = "original_size"
META_MIME_TYPE = "mime_type"
META_HASH = "sha256"
META_TIMESTAMP = "timestamp"
META_IV = "iv"

class FileEncryptionError(Exception):
    pass

def calculate_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()
    chunk_size = 8192  # 8KB chunks
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        raise FileEncryptionError(f"Error calculating file hash: {str(e)}")

def get_file_metadata(file_path: str) -> Dict[str, Any]:
    try:
        path = Path(file_path)
        
        metadata = {
            META_FILENAME: path.name,
            META_ORIGINAL_SIZE: path.stat().st_size,
            META_TIMESTAMP: int(path.stat().st_mtime),
        }
        
        mime_type, _ = mimetypes.guess_type(file_path)
        metadata[META_MIME_TYPE] = mime_type or "application/octet-stream"

        metadata[META_HASH] = calculate_file_hash(file_path)
        
        return metadata
    except Exception as e:
        raise FileEncryptionError(f"Error getting file metadata: {str(e)}")

def read_in_chunks(file_obj: BinaryIO, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Generator[bytes, None, None]:
    while True:
        data = file_obj.read(chunk_size)
        if not data:
            break
        yield data

def _ctr_parallel_worker(args):
    """Worker for parallel CTR block encryption/decryption."""
    (block_index, block_data, key_bytes, nonce, is_encrypt) = args
    # For CTR, we need to generate the keystream for this block
    round_keys = key_expansion(key_bytes, len(key_bytes))
    counter_block = nonce + block_index.to_bytes(8, byteorder='big')
    keystream = encrypt_block(counter_block, round_keys)
    result = xor_bytes(block_data, keystream[:len(block_data)])
    return (block_index, result)

def encrypt_file(
    input_file: str, 
    output_file: str, 
    key: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    show_progress: bool = True,
    mode: str = 'cbc',
    parallel: bool = False
) -> Dict[str, Any]:

    metadata = get_file_metadata(input_file)
    metadata['mode'] = mode
    try:
        key_bytes = key.encode('utf-8') if isinstance(key, str) else key
        key_size = detect_key_size(key_bytes)
        if len(key_bytes) < key_size:
            key_bytes = key_bytes + b'\x00' * (key_size - len(key_bytes))
        elif len(key_bytes) > key_size:
            key_bytes = key_bytes[:key_size]
        with open(input_file, 'rb') as f:
            data = f.read()
        if mode == 'cbc':
            iv, encrypted_data = encrypt_cbc(data, key_bytes)
            metadata[META_IV] = iv.hex()
        elif mode == 'ctr':
            nonce = os.urandom(8)
            if parallel and len(data) > 1024*1024:  # Only parallelize for files >1MB
                block_size = AES_BLOCK_SIZE
                num_blocks = (len(data) + block_size - 1) // block_size
                args_list = []
                for i in range(num_blocks):
                    block = data[i*block_size:(i+1)*block_size]
                    args_list.append((i, block, key_bytes, nonce, True))
                encrypted_blocks = [None] * num_blocks
                with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                    for block_index, result in executor.map(_ctr_parallel_worker, args_list):
                        encrypted_blocks[block_index] = result
                encrypted_data = b''.join(encrypted_blocks)
            else:
                nonce, encrypted_data = encrypt_ctr(data, key_bytes, nonce)
            metadata['nonce'] = nonce.hex()
        else:
            raise FileEncryptionError(f"Unsupported mode: {mode}")
        with open(output_file, 'wb') as f:
            metadata_bytes = json.dumps(metadata).encode('utf-8')
            metadata_length_bytes = len(metadata_bytes).to_bytes(4, byteorder='big')
            f.write(metadata_length_bytes)
            f.write(metadata_bytes)
            f.write(encrypted_data)
        return metadata
    except Exception as e:
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
        except:
            pass
        raise FileEncryptionError(f"Error encrypting file: {str(e)}")

def decrypt_file(
    input_file: str, 
    output_file: str, 
    key: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    show_progress: bool = True,
    parallel: bool = False
) -> Dict[str, Any]:

    try:
        key_bytes = key.encode('utf-8') if isinstance(key, str) else key
        key_size = detect_key_size(key_bytes)
        if len(key_bytes) < key_size:
            key_bytes = key_bytes + b'\x00' * (key_size - len(key_bytes))
        elif len(key_bytes) > key_size:
            key_bytes = key_bytes[:key_size]
        with open(input_file, 'rb') as f:
            metadata_length_bytes = f.read(4)
            if len(metadata_length_bytes) != 4:
                raise FileEncryptionError("Invalid encrypted file format: missing metadata length")
            metadata_length = int.from_bytes(metadata_length_bytes, byteorder='big')
            metadata_bytes = f.read(metadata_length)
            if len(metadata_bytes) != metadata_length:
                raise FileEncryptionError("Invalid encrypted file format: incomplete metadata")
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            mode = metadata.get('mode', 'cbc')
            if mode == 'cbc':
                if META_IV not in metadata:
                    raise FileEncryptionError("Invalid encrypted file format: missing IV in metadata")
                iv = bytes.fromhex(metadata[META_IV])
                encrypted_data = f.read()
                decrypted_data = decrypt_cbc(encrypted_data, key_bytes, iv)
            elif mode == 'ctr':
                if 'nonce' not in metadata:
                    raise FileEncryptionError("Invalid encrypted file format: missing nonce in metadata for CTR mode")
                nonce = bytes.fromhex(metadata['nonce'])
                encrypted_data = f.read()
                if parallel and len(encrypted_data) > 1024*1024:
                    block_size = AES_BLOCK_SIZE
                    num_blocks = (len(encrypted_data) + block_size - 1) // block_size
                    args_list = []
                    for i in range(num_blocks):
                        block = encrypted_data[i*block_size:(i+1)*block_size]
                        args_list.append((i, block, key_bytes, nonce, False))
                    decrypted_blocks = [None] * num_blocks
                    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                        for block_index, result in executor.map(_ctr_parallel_worker, args_list):
                            decrypted_blocks[block_index] = result
                    decrypted_data = b''.join(decrypted_blocks)
                else:
                    decrypted_data = decrypt_ctr(encrypted_data, key_bytes, nonce)
            else:
                raise FileEncryptionError(f"Unsupported mode: {mode}")
        with open(output_file, 'wb') as f:
            f.write(decrypted_data)
        if META_HASH in metadata:
            decrypted_hash = calculate_file_hash(output_file)
            if decrypted_hash != metadata[META_HASH]:
                os.remove(output_file)
                raise FileEncryptionError("File integrity check failed: the decrypted file doesn't match the original hash")
        return metadata
    except Exception as e:
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
        except:
            pass
        raise FileEncryptionError(f"Error decrypting file: {str(e)}")

def verify_file_integrity(file_path: str, expected_hash: str) -> bool:
    try:
        actual_hash = calculate_file_hash(file_path)
        return actual_hash == expected_hash
    except Exception:
        return False

def get_encryption_info(encrypted_file: str) -> Dict[str, Any]:

    try:
        with open(encrypted_file, 'rb') as f:

            metadata_length_bytes = f.read(4)
            if len(metadata_length_bytes) != 4:
                raise FileEncryptionError("Invalid encrypted file format")
                
            metadata_length = int.from_bytes(metadata_length_bytes, byteorder='big')
            

            metadata_bytes = f.read(metadata_length)
            if len(metadata_bytes) != metadata_length:
                raise FileEncryptionError("Invalid encrypted file format")
                
            return json.loads(metadata_bytes.decode('utf-8'))
    except Exception as e:
        raise FileEncryptionError(f"Error reading encryption info: {str(e)}")

if __name__ == "__main__":

    import time
    
    def test_file_encryption():
        # Create a test file
        test_file = "test_file.txt"
        encrypted_file = "test_file.enc"
        decrypted_file = "test_file_decrypted.txt"
        
        # Write test data
        with open(test_file, "w") as f:
            f.write("This is a test file for encryption and decryption.\n" * 1000)
        
        # Encrypt the file
        print(f"Encrypting {test_file}...")
        key = "TestSecretKey123"
        start_time = time.time()
        metadata = encrypt_file(test_file, encrypted_file, key)
        encrypt_time = time.time() - start_time
        
        print(f"File encrypted in {encrypt_time:.2f} seconds.")
        print(f"Metadata: {json.dumps(metadata, indent=2)}")
        
        # Decrypt the file
        print(f"\nDecrypting {encrypted_file}...")
        start_time = time.time()
        decrypted_metadata = decrypt_file(encrypted_file, decrypted_file, key)
        decrypt_time = time.time() - start_time
        
        print(f"File decrypted in {decrypt_time:.2f} seconds.")
        
        # Verify integrity
        original_hash = calculate_file_hash(test_file)
        decrypted_hash = calculate_file_hash(decrypted_file)
        
        print(f"\nIntegrity check:")
        print(f"Original file hash: {original_hash}")
        print(f"Decrypted file hash: {decrypted_hash}")
        print(f"Files match: {original_hash == decrypted_hash}")
        
        # Clean up
        #os.remove(test_file)
        #os.remove(encrypted_file)
        #os.remove(decrypted_file)
    
    test_file_encryption() 