
import sys
import os
import pathlib
import time
import signal

# Add project root to path
ROOT_PATH = pathlib.Path(__file__).parent
sys.path.append((ROOT_PATH / "python").as_posix())

from common.zpipe import zpipe_create_pipe, zpipe_destroy_pipe
from viewervedo.zapi import Zapi

def signal_handler(sig, frame):
    print("\nCtrl+C detected. Exiting...")
    zpipe_destroy_pipe()
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Starting ViewerVedo ZAPI test...")
    zpipe = zpipe_create_pipe(io_threads=1)
    
    # Init Zapi (Router)
    zapi = Zapi(config={}, zpipe=zpipe)
    zapi.run()
    
    print("ViewerVedo ZAPI running. Waiting for messages...")
    
    while True:
        # Simulate processing queue
        cmd = zapi.pop_from_queue()
        if cmd:
            print(f"Processed command from queue: {cmd}")
        time.sleep(0.1)

if __name__ == "__main__":
    main()
