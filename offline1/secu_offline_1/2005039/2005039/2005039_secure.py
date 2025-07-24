#!/usr/bin/env python3
"""
Secure Communication Demo with ECDH Key Exchange

This script demonstrates secure communication using ECDH key exchange
for establishing a shared AES key and encrypted messaging.

Usage:
    # Start a secure server
    python 2005039_secure.py --server

    # Start a secure client
    python 2005039_secure.py --client
"""

import sys
import os
import time
import threading
import argparse
import logging
from typing import Dict, Any

# Import our custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from secure_socket import SecureServer, SecureClient

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Secure Communication Demo with ECDH Key Exchange"
    )
    
    # Server or client mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--server", action="store_true", help="Run in server mode")
    mode.add_argument("--client", action="store_true", help="Run in client mode")
    
    # Optional arguments
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=12345, help="Port number (default: 12345)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--key-bits", type=int, default=256, 
                         help="Key size in bits for ECDH (default: 256)")
    
    return parser.parse_args()

def run_server(host, port, key_bits, debug):
    """
    Run the secure server.
    
    Args:
        host: Host address to bind to
        port: Port number to listen on
        key_bits: Key size in bits for ECDH
        debug: Whether to enable debug output
    """
    # Configure logging
    if debug:
        logging.getLogger('secure_socket').setLevel(logging.DEBUG)
    
    print(f"Starting secure server on {host}:{port}")
    print(f"Using {key_bits}-bit elliptic curve")
    
    # Create and start the secure server
    server = SecureServer(host, port)
    
    if server.start():
        # Initialize the curve
        curve = server.initialize_curve(key_bits)
        print(f"Initialized elliptic curve: y^2 = x^3 + {curve.a}x + {curve.b} (mod {curve.p})")
        print("Server started. Waiting for clients to connect...")
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\nStopping server...")
        
        finally:
            server.stop()
            print("Server stopped.")
    
    else:
        print("Failed to start server.")

def run_client(host, port, debug):
    """
    Run the secure client.
    
    Args:
        host: Server host address to connect to
        port: Server port to connect to
        debug: Whether to enable debug output
    """
    # Configure logging
    if debug:
        logging.getLogger('secure_socket').setLevel(logging.DEBUG)
    
    print(f"Connecting to secure server at {host}:{port}")
    
    # Create a secure client
    client = SecureClient(host, port)
    
    # Message received callback
    def on_message(message):
        print(f"\n{host}:{port}: {message}")
        print("> ", end="", flush=True)
    
    # Error callback
    def on_error(error):
        print(f"\nError: {error}")
        print("> ", end="", flush=True)
    
    # Register callbacks
    client.register_message_callback(on_message)
    client.register_error_callback(on_error)
    
    # Connect to the server and perform key exchange
    print("Connecting and performing key exchange...")
    if client.connect_secure():
        print("Secure connection established!")
        print("Type messages and press Enter to send.")
        print("Type 'exit' to quit.")
        
        try:
            # Message input loop
            while True:
                try:
                    message = input("> ")
                    
                    # Check for exit command
                    if message.lower() == 'exit':
                        break
                    
                    # Send the encrypted message
                    if client.send_encrypted(message):
                        # Message sent successfully
                        pass
                    else:
                        print("Failed to send message.")
                
                except EOFError:
                    # Handle Ctrl+D
                    break
            
            print("\nDisconnecting from server...")
        
        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\nDisconnecting from server...")
        
        finally:
            client.disconnect()
            print("Disconnected.")
    
    else:
        print("Failed to establish secure connection.")

def main():
    """Main function."""
    args = parse_args()
    
    if args.server:
        run_server(args.host, args.port, args.key_bits, args.debug)
    
    elif args.client:
        run_client(args.host, args.port, args.debug)

if __name__ == "__main__":
    main() 