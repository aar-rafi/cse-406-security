#!/usr/bin/env python3
"""
Socket Communication Framework

This module provides the foundation for socket-based communication between
sender and receiver for the secure messaging application.
"""

import socket
import threading
import time
import json
import struct
import logging
from typing import Callable, Dict, Any, Optional, Tuple, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('socket_utils')

# Constants
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 12345
BUFFER_SIZE = 4096
MESSAGE_HEADER_SIZE = 4  # 4 bytes for message length (uint32)

class MessageHandler:
    """
    Base class for message handling functionality.
    Provides methods for serializing and deserializing messages.
    """
    
    @staticmethod
    def serialize_message(message: Dict[str, Any]) -> bytes:
        """
        Serialize a message dictionary to bytes with a length prefix.
        
        Args:
            message: Dictionary containing the message data
            
        Returns:
            Serialized message with length header
        """
        # Convert the message to JSON
        json_message = json.dumps(message).encode('utf-8')
        
        # Create a length prefix (4 bytes, uint32, network byte order)
        length_prefix = struct.pack('!I', len(json_message))
        
        # Return the length prefix followed by the JSON message
        return length_prefix + json_message
    
    @staticmethod
    def deserialize_message(data: bytes) -> Dict[str, Any]:
        """
        Deserialize a message from bytes to a dictionary.
        
        Args:
            data: Serialized message bytes (without length header)
            
        Returns:
            Deserialized message dictionary
        """
        # Convert the JSON message back to a dictionary
        return json.loads(data.decode('utf-8'))
    
    @staticmethod
    def receive_message(sock: socket.socket) -> Optional[Dict[str, Any]]:
        """
        Receive a length-prefixed message from a socket.
        
        Args:
            sock: Socket to receive data from
            
        Returns:
            Deserialized message or None if an error occurred
        """
        try:
            # First, receive the 4-byte length prefix
            length_data = b''
            while len(length_data) < MESSAGE_HEADER_SIZE:
                chunk = sock.recv(MESSAGE_HEADER_SIZE - len(length_data))
                if not chunk:
                    logger.error("Connection closed while receiving message header")
                    return None
                length_data += chunk
            
            # Unpack the length prefix to get the message length
            message_length = struct.unpack('!I', length_data)[0]
            
            # Now receive the actual message data
            message_data = b''
            while len(message_data) < message_length:
                chunk = sock.recv(min(BUFFER_SIZE, message_length - len(message_data)))
                if not chunk:
                    logger.error("Connection closed while receiving message data")
                    return None
                message_data += chunk
            
            # Deserialize and return the message
            return MessageHandler.deserialize_message(message_data)
        
        except (socket.error, json.JSONDecodeError, struct.error) as e:
            logger.error(f"Error receiving message: {str(e)}")
            return None
    
    @staticmethod
    def send_message(sock: socket.socket, message: Dict[str, Any]) -> bool:
        """
        Send a message over a socket with a length prefix.
        
        Args:
            sock: Socket to send data to
            message: Message dictionary to send
            
        Returns:
            True if the message was sent successfully, False otherwise
        """
        try:
            # Serialize the message with a length prefix
            serialized_message = MessageHandler.serialize_message(message)
            
            # Send the serialized message
            sock.sendall(serialized_message)
            return True
        
        except socket.error as e:
            logger.error(f"Error sending message: {str(e)}")
            return False

class Server(MessageHandler):
    """
    Server component that listens for incoming connections and handles
    message exchange with connected clients.
    """
    
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """
        Initialize the server with host and port.
        
        Args:
            host: Host address to bind to (default: 127.0.0.1)
            port: Port to listen on (default: 12345)
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.clients = {}  # Dictionary to store client connections
        self.message_handlers = {}  # Dictionary to store message handlers by type
    
    def register_handler(self, message_type: str, handler: Callable):
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Function to call when a message of this type is received
        """
        self.message_handlers[message_type] = handler
    
    def start(self):
        """
        Start the server and listen for incoming connections.
        """
        try:
            # Create a TCP socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to the host and port
            self.server_socket.bind((self.host, self.port))
            
            # Start listening for connections
            self.server_socket.listen(5)
            self.running = True
            
            logger.info(f"Server started on {self.host}:{self.port}")
            
            # Start a thread to accept connections
            accept_thread = threading.Thread(target=self._accept_connections)
            accept_thread.daemon = True
            accept_thread.start()
            
            return True
        
        except socket.error as e:
            logger.error(f"Failed to start server: {str(e)}")
            return False
    
    def stop(self):
        """
        Stop the server and close all connections.
        """
        self.running = False
        
        # Close all client connections
        for client_id, client_info in list(self.clients.items()):
            try:
                client_info['socket'].close()
            except socket.error:
                pass
        
        self.clients.clear()
        
        # Close the server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except socket.error:
                pass
        
        logger.info("Server stopped")
    
    def _accept_connections(self):
        """
        Accept incoming connections and start a thread to handle each client.
        """
        while self.running:
            try:
                # Accept a connection
                client_socket, client_address = self.server_socket.accept()
                
                # Generate a unique client ID
                client_id = f"{client_address[0]}:{client_address[1]}"
                
                # Store client information
                self.clients[client_id] = {
                    'socket': client_socket,
                    'address': client_address,
                    'connected_at': time.time()
                }
                
                logger.info(f"New connection from {client_id}")
                
                # Start a thread to handle this client
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_id, client_socket)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except socket.error as e:
                if self.running:
                    logger.error(f"Error accepting connection: {str(e)}")
                    time.sleep(0.1)  # Prevent CPU spinning on repeated errors
    
    def _handle_client(self, client_id: str, client_socket: socket.socket):
        """
        Handle communication with a connected client.
        
        Args:
            client_id: Unique identifier for the client
            client_socket: Socket for communication with the client
        """
        while self.running and client_id in self.clients:
            try:
                # Receive a message from the client
                message = self.receive_message(client_socket)
                
                if message is None:
                    # Client disconnected or error occurred
                    break
                
                # Log the received message
                logger.debug(f"Received message from {client_id}: {message}")
                
                # Handle the message based on its type
                if 'type' in message and message['type'] in self.message_handlers:
                    # Call the registered handler for this message type
                    self.message_handlers[message['type']](client_id, message)
                else:
                    logger.warning(f"Unknown message type from {client_id}: {message.get('type', 'unknown')}")
            
            except socket.error as e:
                logger.error(f"Error handling client {client_id}: {str(e)}")
                break
        
        # Client disconnected or error occurred
        self._disconnect_client(client_id)
    
    def _disconnect_client(self, client_id: str):
        """
        Handle client disconnection.
        
        Args:
            client_id: Unique identifier for the client
        """
        if client_id in self.clients:
            try:
                self.clients[client_id]['socket'].close()
            except socket.error:
                pass
            
            logger.info(f"Client {client_id} disconnected")
            del self.clients[client_id]
    
    def send_to_client(self, client_id: str, message: Dict[str, Any]) -> bool:
        """
        Send a message to a specific client.
        
        Args:
            client_id: Unique identifier for the client
            message: Message dictionary to send
            
        Returns:
            True if the message was sent successfully, False otherwise
        """
        if client_id not in self.clients:
            logger.warning(f"Cannot send message to unknown client: {client_id}")
            return False
        
        return self.send_message(self.clients[client_id]['socket'], message)
    
    def broadcast(self, message: Dict[str, Any], exclude: Optional[List[str]] = None):
        """
        Send a message to all connected clients.
        
        Args:
            message: Message dictionary to broadcast
            exclude: List of client IDs to exclude from broadcast
        """
        if exclude is None:
            exclude = []
        
        for client_id in list(self.clients.keys()):
            if client_id not in exclude:
                self.send_to_client(client_id, message)

class Client(MessageHandler):
    """
    Client component that connects to a server and exchanges messages.
    """
    
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """
        Initialize the client with server host and port.
        
        Args:
            host: Server host address to connect to (default: 127.0.0.1)
            port: Server port to connect to (default: 12345)
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.receive_thread = None
        self.message_handlers = {}  # Dictionary to store message handlers by type
    
    def register_handler(self, message_type: str, handler: Callable):
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Function to call when a message of this type is received
        """
        self.message_handlers[message_type] = handler
    
    def connect(self, timeout: float = 5.0) -> bool:
        """
        Connect to the server.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if connected successfully, False otherwise
        """
        if self.connected:
            logger.warning("Already connected to server")
            return True
        
        try:
            # Create a TCP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            
            # Connect to the server
            start_time = time.time()
            self.socket.connect((self.host, self.port))
            connection_time = (time.time() - start_time) * 1000  # in milliseconds
            
            # Set back to blocking mode
            self.socket.settimeout(None)
            
            self.connected = True
            
            logger.info(f"Connected to server at {self.host}:{self.port} in {connection_time:.2f} ms")
            
            # Start receiving messages
            self.receive_thread = threading.Thread(target=self._receive_messages)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            return True
        
        except socket.error as e:
            logger.error(f"Failed to connect to server: {str(e)}")
            if self.socket:
                self.socket.close()
                self.socket = None
            return False
    
    def disconnect(self):
        """
        Disconnect from the server.
        """
        self.connected = False
        
        if self.socket:
            try:
                self.socket.close()
            except socket.error:
                pass
            self.socket = None
        
        logger.info("Disconnected from server")
    
    def _receive_messages(self):
        """
        Continuously receive and handle messages from the server.
        """
        while self.connected:
            try:
                # Receive a message from the server
                message = self.receive_message(self.socket)
                
                if message is None:
                    # Server disconnected or error occurred
                    break
                
                # Log the received message
                logger.debug(f"Received message from server: {message}")
                
                # Handle the message based on its type
                if 'type' in message and message['type'] in self.message_handlers:
                    # Call the registered handler for this message type
                    self.message_handlers[message['type']](message)
                else:
                    logger.warning(f"Unknown message type from server: {message.get('type', 'unknown')}")
            
            except socket.error as e:
                if self.connected:
                    logger.error(f"Error receiving message from server: {str(e)}")
                break
        
        # Server disconnected or error occurred
        self.disconnect()
    
    def send(self, message: Dict[str, Any]) -> bool:
        """
        Send a message to the server.
        
        Args:
            message: Message dictionary to send
            
        Returns:
            True if the message was sent successfully, False otherwise
        """
        if not self.connected or not self.socket:
            logger.warning("Cannot send message: not connected to server")
            return False
        
        return self.send_message(self.socket, message)

def measure_latency(client: Client, num_messages: int = 100) -> Tuple[float, float, float]:
    """
    Measure the round-trip latency of messages.
    
    Args:
        client: Connected client to use for measurement
        num_messages: Number of messages to send for measurement
    
    Returns:
        Tuple containing (min_latency, max_latency, avg_latency) in milliseconds
    """
    if not client.connected:
        raise ValueError("Client must be connected to measure latency")
    
    latencies = []
    
    # Create events to signal when a pong is received
    pong_events = {}
    
    # Register a handler for pong messages
    def pong_handler(message):
        message_id = message.get('id')
        if message_id in pong_events:
            pong_events[message_id].set()
    
    client.register_handler('pong', pong_handler)
    
    # Send ping messages and measure round-trip time
    for i in range(num_messages):
        message_id = f"ping-{i}"
        event = threading.Event()
        pong_events[message_id] = event
        
        # Send the ping
        start_time = time.time()
        success = client.send({
            'type': 'ping',
            'id': message_id,
            'timestamp': start_time
        })
        
        if success:
            # Wait for the pong response
            if event.wait(timeout=2.0):
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)
            else:
                logger.warning(f"Timeout waiting for pong response to {message_id}")
        
        # Sleep to avoid overwhelming the server
        time.sleep(0.01)
    
    # Clean up
    for event in pong_events.values():
        event.clear()
    
    # Calculate statistics
    if latencies:
        min_latency = min(latencies)
        max_latency = max(latencies)
        avg_latency = sum(latencies) / len(latencies)
        return min_latency, max_latency, avg_latency
    else:
        return 0, 0, 0

def measure_throughput(client: Client, message_size: int, duration: float = 5.0) -> float:
    """
    Measure the throughput of the connection.
    
    Args:
        client: Connected client to use for measurement
        message_size: Size of each message in bytes
        duration: Duration of the test in seconds
    
    Returns:
        Throughput in messages per second
    """
    if not client.connected:
        raise ValueError("Client must be connected to measure throughput")
    
    # Create a dummy payload of the desired size
    payload = "X" * message_size
    
    # Counter for sent messages
    sent_messages = 0
    
    # Send messages as fast as possible for the specified duration
    start_time = time.time()
    end_time = start_time + duration
    
    while time.time() < end_time:
        success = client.send({
            'type': 'throughput_test',
            'id': f"msg-{sent_messages}",
            'payload': payload
        })
        
        if success:
            sent_messages += 1
        
        # Small sleep to avoid overwhelming the CPU
        time.sleep(0.001)
    
    actual_duration = time.time() - start_time
    
    # Calculate throughput
    if actual_duration > 0:
        return sent_messages / actual_duration
    else:
        return 0

if __name__ == "__main__":
    # Simple demo of the socket utilities
    print("Socket Communication Framework Demo")
    print("=" * 50)
    
    # Create a server
    server = Server()
    
    # Register a ping handler
    def handle_ping(client_id, message):
        print(f"Received ping from {client_id}: {message}")
        server.send_to_client(client_id, {
            'type': 'pong',
            'id': message.get('id'),
            'timestamp': time.time()
        })
    
    def handle_throughput_test(client_id, message):
        # Just acknowledge receipt
        server.send_to_client(client_id, {
            'type': 'throughput_ack',
            'id': message.get('id')
        })
    
    server.register_handler('ping', handle_ping)
    server.register_handler('throughput_test', handle_throughput_test)
    
    # Start the server
    if server.start():
        try:
            # Create a client
            client = Client()
            
            # Connect to the server
            if client.connect():
                # Measure latency
                print("\nMeasuring latency...")
                min_latency, max_latency, avg_latency = measure_latency(client, num_messages=50)
                print(f"Latency (ms): min={min_latency:.2f}, max={max_latency:.2f}, avg={avg_latency:.2f}")
                
                # Measure throughput
                print("\nMeasuring throughput...")
                for size in [100, 1000, 10000]:
                    throughput = measure_throughput(client, size, duration=1.0)
                    print(f"Throughput with {size} byte messages: {throughput:.2f} msgs/sec")
                
                # Disconnect the client
                client.disconnect()
            
        except KeyboardInterrupt:
            print("\nExiting demo...")
        
        finally:
            # Stop the server
            server.stop() 