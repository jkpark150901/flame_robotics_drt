"""
Example usage of zpipe module (New ZSocket Pattern)
Demonstrates how to create, join, and destroy sockets using the new API
"""

import sys
import os
import pathlib
import time
import json
import threading
import signal

# Add parent directory to Python path to find modules
ROOT_PATH = pathlib.Path(__file__).parent.parent
sys.path.append(ROOT_PATH.as_posix())

from common.zpipe import zpipe_create_pipe, zpipe_destroy_pipe, AsyncZSocket

def message_handler(multipart_data):
    """Callback function to handle received messages"""
    print(f"Received {len(multipart_data)} parts:")
    for i, part in enumerate(multipart_data):
        print(f"  Part {i}: {part.decode('utf-8')}")

def publisher_example():
    """Example of creating a publisher socket"""
    print("Creating publisher socket...")
    
    # Step 1: Create ZPipe instance
    zpipe = zpipe_create_pipe(io_threads=1)
    
    # Step 2: Create AsyncZSocket with publish pattern
    pub_socket = AsyncZSocket("publisher", "publish")
    
    # Step 3: Create socket (this registers it with ZPipe)
    if pub_socket.create(pipeline=zpipe):
        # Step 4: Join socket (auto-binds as server for publish pattern)
        if pub_socket.join("tcp", "localhost", 9001):
            print("Publisher socket created and joined successfully")
            
            # Step 5: Send some test messages
            for i in range(5):
                message = {
                    "function": "test_function",
                    "kwargs": {"param1": f"value_{i}", "param2": i}
                }
                pub_socket.dispatch(["call", json.dumps(message)])
                print(f"Sent message {i}")
                time.sleep(1)
            
            # Step 6: Destroy socket (this unregisters from ZPipe)
            pub_socket.destroy_socket()
            print("Publisher socket destroyed")
        else:
            print("Failed to join publisher socket")
    else:
        print("Failed to create publisher socket")

# Global shutdown event
shutdown_event = threading.Event()

def signal_handler(signum, frame):
    """Handle Ctrl+C signal"""
    print("\nReceived interrupt signal. Shutting down gracefully...")
    shutdown_event.set()

def subscriber_example():
    """Example of creating a subscriber socket"""
    print("Creating subscriber socket...")
    
    # Step 1: Get ZPipe instance (already created)
    zpipe = zpipe_create_pipe(io_threads=1)
    
    # Step 2: Create AsyncZSocket with subscribe pattern
    sub_socket = AsyncZSocket("subscriber", "subscribe")
    
    # Step 3: Create socket (this registers it with ZPipe)
    if sub_socket.create(pipeline=zpipe):
        # Step 4: Join socket (auto-connects as client for subscribe pattern)
        if sub_socket.join("tcp", "localhost", 9001):
            print("Subscriber socket created and joined successfully")
            
            # Step 5: Configure subscription and callback
            sub_socket.subscribe("call")  # Subscribe to "call" topic
            sub_socket.set_message_callback(message_handler)
            
            # Step 6: Keep running until shutdown signal
            while not shutdown_event.is_set():
                time.sleep(0.1)  # Check shutdown event every 100ms
            
            # Step 7: Destroy socket (this unregisters from ZPipe)
            sub_socket.destroy_socket()
            print("Subscriber socket destroyed")
        else:
            print("Failed to join subscriber socket")
    else:
        print("Failed to create subscriber socket")

if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Step 1: Create singleton zpipe instance with context
        print("Creating ZPipe singleton...")
        pipe_instance = zpipe_create_pipe(io_threads=2)
        print("ZPipe singleton created successfully")
        
        # Step 2: Start subscriber in a separate thread
        subscriber_thread = threading.Thread(target=subscriber_example, daemon=True)
        subscriber_thread.start()
        
        # Wait a bit for subscriber to start
        time.sleep(2)
        
        # Step 3: Start publisher
        if not shutdown_event.is_set():
            publisher_example()
        
        # Step 4: Wait for subscriber to finish or shutdown signal
        while subscriber_thread.is_alive() and not shutdown_event.is_set():
            time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught. Shutting down...")
        shutdown_event.set()
    finally:
        # Step 5: Destroy singleton and cleanup all resources
        print("Destroying ZPipe singleton...")
        zpipe_destroy_pipe()
        print("ZPipe destroyed successfully")
