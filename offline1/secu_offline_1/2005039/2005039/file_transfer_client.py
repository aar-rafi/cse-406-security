#!/usr/bin/env python3
"""
File Transfer Client

Encrypts a file using file_crypto.py and sends it to the server over TCP.
"""
import socket
import argparse
import os
from file_crypto import encrypt_file, FileEncryptionError

BUFFER_SIZE = 4096

def send_file(sock, input_file, key):
    """Encrypt a file and send it to the server (send the full encrypted file as-is)."""
    # Encrypt the file to a temp file
    enc_file = input_file + ".enc_tmp"
    encrypt_file(input_file, enc_file, key, show_progress=False)
    # Send the entire encrypted file as a binary stream
    with open(enc_file, 'rb') as f:
        while True:
            chunk = f.read(BUFFER_SIZE)
            if not chunk:
                break
            sock.sendall(chunk)
    os.remove(enc_file)
    print(f"Sent encrypted file: {input_file}")

def main():
    parser = argparse.ArgumentParser(description="File Transfer Client (encrypts and sends file)")
    parser.add_argument('input_file', help='File to send')
    parser.add_argument('-k', '--key', required=True, help='Encryption key')
    parser.add_argument('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
    parser.add_argument('-p', '--port', type=int, default=5001, help='Server port (default: 5001)')
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Input file not found: {args.input_file}")
        return
    print(f"Connecting to {args.host}:{args.port}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((args.host, args.port))
        try:
            send_file(sock, args.input_file, args.key)
        except FileEncryptionError as e:
            print(f"Encryption error: {e}")
        except Exception as e:
            print(f"Error: {e}")
    print("File transfer complete.")

if __name__ == "__main__":
    main() 