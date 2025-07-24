#!/usr/bin/env python3
"""
File Transfer Server

Receives an encrypted file over TCP, saves it, decrypts it using file_crypto.py, and saves the decrypted file.
"""
import socket
import argparse
import os
from pathlib import Path
from file_crypto import decrypt_file, FileEncryptionError

BUFFER_SIZE = 4096

def receive_file(conn, output_dir, filename_hint="received_file.enc"):
    """Receive an encrypted file and save it to output_dir. Return the file path."""
    enc_path = os.path.join(output_dir, filename_hint)
    with open(enc_path, 'wb') as f:
        while True:
            chunk = conn.recv(BUFFER_SIZE)
            if not chunk:
                break
            f.write(chunk)
    return enc_path

def main():
    parser = argparse.ArgumentParser(description="File Transfer Server (receives encrypted file and decrypts it)")
    parser.add_argument('-p', '--port', type=int, default=5001, help='Port to listen on (default: 5001)')
    parser.add_argument('-k', '--key', required=True, help='Decryption key')
    parser.add_argument('-o', '--output-dir', default='received_files', help='Directory to save received and decrypted files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Listening on port {args.port}... (output dir: {args.output_dir})")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', args.port))
        s.listen(1)
        while True:
            print("Waiting for a connection...")
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                try:
                    enc_path = receive_file(conn, args.output_dir)
                    print(f"Received encrypted file: {enc_path}")
                    # Decrypt the file
                    dec_filename = f"decrypted_{Path(enc_path).name}"
                    dec_path = os.path.join(args.output_dir, dec_filename)
                    decrypt_file(enc_path, dec_path, args.key)
                    print(f"Decrypted file saved as: {dec_path}")
                except FileEncryptionError as e:
                    print(f"Decryption error: {e}")
                except Exception as e:
                    print(f"Error: {e}")
            print("Connection closed. Ready for next file.")

if __name__ == "__main__":
    main() 