#!/usr/bin/env python3
"""
Socket Communication Demo

This script demonstrates the socket communication framework with a simple
messaging application that allows sending and receiving text messages.

Usage:
    # Start a server
    python 2005039_socket.py --server

    # Start a client and connect to the server
    python 2005039_socket.py --client
"""

import sys
import os
import time
import threading
import argparse
from typing import Dict, Any

# Import our socket utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from socket_utils import Server, Client, measure_latency, measure_throughput

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Socket Communication Demo"
    )
    
    # Server or client mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--server", action="store_true", help="Run in server mode")
    mode.add_argument("--client", action="store_true", help="Run in client mode")
    
    # Optional arguments
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=12345, help="Port number (default: 12345)")
    parser.add_argument("--measure", action="store_true", help="Measure performance metrics")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    return parser.parse_args()

def run_server(host, port, debug):
    """
    Run the server component.
    
    Args:
        host: Host address to bind to
        port: Port number to listen on
        debug: Whether to enable debug output
    """
    print(f"Starting server on {host}:{port}")
    
    # Create a server
    server = Server(host, port)
    
    # Register message handlers
    def handle_message(client_id, message):
        """Handle regular messages from clients."""
        text = message.get('text', '')
        print(f"Message from {client_id}: {text}")
        
        # Broadcast the message to all clients (including the sender)
        server.broadcast({
            'type': 'message',
            'sender': client_id,
            'text': text,
            'timestamp': time.time()
        })
    
    def handle_ping(client_id, message):
        """Handle ping messages for latency measurement."""
        # Send a pong response
        server.send_to_client(client_id, {
            'type': 'pong',
            'id': message.get('id'),
            'timestamp': time.time()
        })
    
    def handle_throughput_test(client_id, message):
        """Handle throughput test messages."""
        # Just acknowledge receipt
        server.send_to_client(client_id, {
            'type': 'throughput_ack',
            'id': message.get('id')
        })
    
    # Register the handlers
    server.register_handler('message', handle_message)
    server.register_handler('ping', handle_ping)
    server.register_handler('throughput_test', handle_throughput_test)
    
    # Start the server
    if server.start():
        print("Server started successfully. Press Ctrl+C to stop.")
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\nStopping server...")
        
        finally:
            server.stop()
    
    else:
        print("Failed to start server.")

def run_client(host, port, measure, debug):
    """
    Run the client component.
    
    Args:
        host: Server host address to connect to
        port: Server port to connect to
        measure: Whether to measure performance metrics
        debug: Whether to enable debug output
    """
    print(f"Connecting to server at {host}:{port}")
    
    # Create a client
    client = Client(host, port)
    
    # Flag to control the message receiving loop
    running = True
    
    # Register message handlers
    def handle_message(message):
        """Handle messages from the server."""
        sender = message.get('sender', 'Unknown')
        text = message.get('text', '')
        
        # Don't print our own messages again
        if sender != f"{host}:{port}":
            print(f"\n{sender}: {text}")
            print("> ", end="", flush=True)
    
    def handle_pong(message):
        """Handle pong messages for latency measurement."""
        # This is handled by the measure_latency function
        pass
    
    def handle_throughput_ack(message):
        """Handle throughput acknowledgement messages."""
        # This is handled by the measure_throughput function
        pass
    
    # Register the handlers
    client.register_handler('message', handle_message)
    client.register_handler('pong', handle_pong)
    client.register_handler('throughput_ack', handle_throughput_ack)
    
    # Connect to the server
    if client.connect():
        print("Connected to server. Type messages and press Enter to send.")
        print("Type 'exit' to quit.")
        
        if measure:
            # Measure performance
            print("\nMeasuring connection performance...")
            
            # Measure latency
            min_latency, max_latency, avg_latency = measure_latency(client, num_messages=50)
            print(f"Latency (ms): min={min_latency:.2f}, max={max_latency:.2f}, avg={avg_latency:.2f}")
            
            # Measure throughput
            size = 1000  # Use 1KB messages
            throughput = measure_throughput(client, size, duration=1.0)
            print(f"Throughput with {size} byte messages: {throughput:.2f} msgs/sec")
            print()
        
        try:
            # Message input loop
            while running:
                try:
                    message_text = input("> ")
                    
                    # Check for exit command
                    if message_text.lower() == 'exit':
                        break
                    
                    # Send the message to the server
                    client.send({
                        'type': 'message',
                        'text': message_text,
                        'timestamp': time.time()
                    })
                
                except EOFError:
                    # Handle Ctrl+D
                    break
            
            print("\nDisconnecting from server...")
        
        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\nDisconnecting from server...")
        
        finally:
            client.disconnect()
    
    else:
        print("Failed to connect to the server.")

def main():
    """Main function."""
    args = parse_args()
    
    if args.server:
        run_server(args.host, args.port, args.debug)
    
    elif args.client:
        run_client(args.host, args.port, args.measure, args.debug)

if __name__ == "__main__":
    main() 