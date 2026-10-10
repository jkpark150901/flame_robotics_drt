import sys
import os
import pathlib
import time
import signal
import zmq

# Add project root to path
ROOT_PATH = pathlib.Path(__file__).parent
sys.path.append((ROOT_PATH / "python").as_posix())

from common.zpipe import zpipe_create_pipe, zpipe_destroy_pipe, AsyncZSocket
from util.logger.console import ConsoleLogger

def signal_handler(sig, frame):
    print("\nCtrl+C detected. Exiting...")
    zpipe_destroy_pipe()
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    console = ConsoleLogger.get_logger()
    console.info("Starting SimTool Router...")
    
    # Create ZPipe
    zpipe = zpipe_create_pipe(io_threads=1)
    
    # Create Router Socket
    router = AsyncZSocket("Router", "router")
    router.create(pipeline=zpipe)
    
    # Callback to handle messages
    def on_message(multipart_data):
        try:
            console.info("------------------------------------------------")
            console.info(f"Received message with {len(multipart_data)} frames")
            
            # Print frame contents
            for i, frame in enumerate(multipart_data):
                try:
                    decoded = frame.decode('utf-8')
                    console.info(f"Frame {i}: {decoded}")
                except:
                    console.info(f"Frame {i}: <binary data> ({len(frame)} bytes)")

            # Expected format from Dealer -> Router:
            # [identity, socket_name, function, kwargs]
            if len(multipart_data) >= 4:
                identity = multipart_data[0] # Binary identity
                socket_name = multipart_data[1].decode('utf-8')
                function_name = multipart_data[2].decode('utf-8')
                kwargs = multipart_data[3].decode('utf-8')
                
                if function_name == "load_spool":
                    console.info(f">> Detected 'load_spool' from {socket_name}")
                    console.info(f"   kwargs: {kwargs}")

                    # Send ACK reply
                    # Router needs to send [identity, socket_name, function, kwargs]
                    # We use "router" as our socket name
                    reply = [
                        identity,
                        b"router",
                        b"ack",
                        b"{}"
                    ]
                    console.info(f"Sending ACK to {socket_name} (Identity: {identity})")
                    router.dispatch(reply)
            else:
                 console.warning("Received message with unexpected frame count")

        except Exception as e:
            console.error(f"Error parsing message: {e}")

    # Set callback BEFORE joining to ensure receiver thread starts
    router.set_message_callback(on_message)
    
    # Bind to IPC address
    ipc_address = "/tmp/viewervedo" 
    
    if router.join("ipc", ipc_address):
        console.info(f"Router bound to ipc://{ipc_address}")
    else:
        console.error(f"Failed to bind to ipc://{ipc_address}")
        return
    
    console.info("Router is running. Waiting for messages... (Press Ctrl+C to stop)")
    
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
