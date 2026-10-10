"""
Example usage of PAIR pattern in zpipe module
Demonstrates 1:1 bidirectional communication using ZMQ PAIR sockets
"""

import sys
import pathlib
import time
import json
import threading
import signal

# Add parent directory to Python path to find modules
ROOT_PATH = pathlib.Path(__file__).parent.parent
sys.path.append(ROOT_PATH.as_posix())

from common.zpipe import zpipe_create_pipe, zpipe_destroy_pipe, AsyncZSocket

# Global shutdown event
shutdown_event = threading.Event()

def signal_handler(signum, frame):
    """Handle Ctrl+C signal"""
    print("\nReceived interrupt signal. Shutting down gracefully...")
    shutdown_event.set()

def message_handler(multipart_data):
    """Callback function to handle received messages"""
    print(f"Received {len(multipart_data)} parts:")
    for i, part in enumerate(multipart_data):
        print(f"  Part {i}: {part.decode('utf-8')}")

def pair_server_example():
    """Example of PAIR socket acting as server (bind)"""
    print("Creating PAIR server socket...")
    
    # Step 1: Get ZPipe instance
    zpipe = zpipe_create_pipe(io_threads=1)
    
    # Step 2: Create AsyncZSocket with pair pattern
    pair_socket = AsyncZSocket("pair_server", "pair")
    
    # Step 3: Create socket (this registers it with ZPipe)
    if pair_socket.create(pipeline=zpipe):
        # Step 4: Join socket as server (bind)
        if pair_socket.join("tcp", "localhost", 9002, is_server=True):
            print("PAIR server socket created and bound successfully")
            
            # Step 5: Set callback for receiving messages
            pair_socket.set_message_callback(message_handler)
            
            # Step 6: Send some messages to client
            for i in range(3):
                if shutdown_event.is_set():
                    break
                message = {
                    "from": "server",
                    "message": f"Hello from server {i}",
                    "timestamp": time.time()
                }
                pair_socket.dispatch([json.dumps(message)])
                print(f"Server sent message {i}")
                time.sleep(2)
            
            # Step 7: Keep running until shutdown
            while not shutdown_event.is_set():
                time.sleep(0.1)
            
            # Step 8: Destroy socket
            pair_socket.destroy_socket()
            print("PAIR server socket destroyed")
        else:
            print("Failed to bind PAIR server socket")
    else:
        print("Failed to create PAIR server socket")

def pair_client_example():
    """Example of PAIR socket acting as client (connect)"""
    print("Creating PAIR client socket...")
    
    # Step 1: Get ZPipe instance
    zpipe = zpipe_create_pipe(io_threads=1)
    
    # Step 2: Create AsyncZSocket with pair pattern
    pair_socket = AsyncZSocket("pair_client", "pair")
    
    # Step 3: Create socket (this registers it with ZPipe)
    if pair_socket.create(pipeline=zpipe):
        # Step 4: Join socket as client (connect)
        if pair_socket.join("tcp", "localhost", 9002, is_server=False):
            print("PAIR client socket created and connected successfully")
            
            # Step 5: Set callback for receiving messages
            pair_socket.set_message_callback(message_handler)
            
            # Step 6: Send some messages to server
            for i in range(3):
                if shutdown_event.is_set():
                    break
                message = {
                    "from": "client",
                    "message": f"Hello from client {i}",
                    "timestamp": time.time()
                }
                pair_socket.dispatch([json.dumps(message)])
                print(f"Client sent message {i}")
                time.sleep(2)
            
            # Step 7: Keep running until shutdown
            while not shutdown_event.is_set():
                time.sleep(0.1)
            
            # Step 8: Destroy socket
            pair_socket.destroy_socket()
            print("PAIR client socket destroyed")
        else:
            print("Failed to connect PAIR client socket")
    else:
        print("Failed to create PAIR client socket")

if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Step 1: Create singleton zpipe instance with context
        print("Creating ZPipe singleton...")
        pipe_instance = zpipe_create_pipe(io_threads=2)
        print("ZPipe singleton created successfully")
        
        # Step 2: Start server in a separate thread
        server_thread = threading.Thread(target=pair_server_example, daemon=True)
        server_thread.start()
        
        # Wait a bit for server to start
        time.sleep(1)
        
        # Step 3: Start client in a separate thread
        client_thread = threading.Thread(target=pair_client_example, daemon=True)
        client_thread.start()
        
        # Step 4: Wait for threads to finish or shutdown signal
        while (server_thread.is_alive() or client_thread.is_alive()) and not shutdown_event.is_set():
            time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught. Shutting down...")
        shutdown_event.set()
    finally:
        # Step 5: Destroy singleton and cleanup all resources
        print("Destroying ZPipe singleton...")
        zpipe_destroy_pipe()
        print("ZPipe destroyed successfully")
