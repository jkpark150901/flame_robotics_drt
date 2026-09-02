"""
Zapi Test Script for Viewervedo
@note
- Acts as a DEALER client to test the API provided by Viewervedo (ROUTER server).
- Sends ping and load_spool commands.
- Verifies replies.
"""

import sys
import os
import pathlib
import time
import json
import threading

# Add root path to sys.path
ROOT_PATH = pathlib.Path(__file__).parent.parent
sys.path.append(ROOT_PATH.as_posix())

from common.zpipe import zpipe_create_pipe, zpipe_destroy_pipe, AsyncZSocket
from common.zapi import ZAPIBase
from util.logger.console import ConsoleLogger

class ConfigurableZapi(ZAPIBase):
    pass

def test_zapi():
    console = ConsoleLogger.get_logger()
    console.info("Starting Zapi Test (Dealer Client)")

    # 1. Create ZPipe
    zpipe = zpipe_create_pipe()

    # 2. Create DEALER socket
    # Note: Viewervedo Zapi (Router) uses IPC
    # So Dealer must connect to /tmp/viewervedo
    dealer = AsyncZSocket("TestDealer", "dealer")
    
    if not dealer.create(pipeline=zpipe):
        console.error("Failed to create dealer socket")
        return

    # Use a unique identity for this dealer
    # In AsyncZSocket/zmq, if we don't set identity explicitly, it's auto-generated.
    # Dealer pattern handles it.

    if not dealer.join("ipc", "/tmp/viewervedo"):
        console.error("Failed to connect to Viewervedo")
        return

    console.info("Connected to Viewervedo on ipc:///tmp/viewervedo")

    # Queue for received messages
    received_msgs = []
    
    def on_message(multipart_data):
        # Dealer receives [message] (or [empty, message] if REQ/REP quirks, but Dealer-Router usually just [message])
        # Let's see what we get.
        console.info(f"Received reply: {multipart_data}")
        received_msgs.append(multipart_data)

    dealer.set_message_callback(on_message)

    # Instantiate ZAPIBase to use call method
    zapi = ConfigurableZapi()

    # Allow connection time
    time.sleep(1.0)

    # 3. Test Ping
    console.info("Sending PING...")
    zapi.call(dealer, "ping", {})

    # Wait for Pong
    start_wait = time.time()
    pong_received = False
    while time.time() - start_wait < 5.0:
        if received_msgs:
            # Check for pong
            for msg_parts in received_msgs:
                # msg_parts could be [identity, socket_name, function, json_kwargs] for ROUTER reply?
                # Or just [socket_name, function, json_kwargs] since Dealer is client?
                # Viewervedo logic: dispatch([identity, socket_name, func, kwargs])
                # AsyncZSocket DEALER receiving: identity is peeled off by ZMQ? No, Dealer doesn't peel unless REQ.
                # Actually, Dealer-Router:
                # Server sends [identity, data...] -> Client receives [data...]
                # So client receives [socket_name, function, json_kwargs]
                
                # Check parts
                if len(msg_parts) >= 3:
                     # part 0: socket_name
                     # part 1: function
                     # part 2: json_kwargs
                     func_part = msg_parts[1]
                     func_str = func_part.decode('utf-8') if isinstance(func_part, bytes) else func_part
                     
                     if func_str == "pong":
                         console.info("SUCCESS: Received PONG")
                         pong_received = True
                         break
                
                # Falback check for json decode on last part
                for part in msg_parts:
                    try:
                        decoded = json.loads(part.decode('utf-8'))
                        if decoded.get("function") == "pong":
                             # If wrapped in old style or something
                             console.info("SUCCESS: Received PONG (Old Style Match?)")
                             pong_received = True
                             break
                    except:
                        pass
                if pong_received: break
            
            if pong_received:
                received_msgs.clear() # Clear for next test
                break
        time.sleep(0.1)
    
    if not pong_received:
        console.error("FAILURE: Did not receive PONG")

    # 4. Test Load Spool (Mock)
    console.info("Sending LOAD_SPOOL (dummy path)...")
    zapi.call(dealer, "load_spool", {"path": "/dummy/path/to/spool.obj"})

    # Wait for Reply
    start_wait = time.time()
    reply_received = False
    while time.time() - start_wait < 5.0:
        if received_msgs:
            for msg_parts in received_msgs:
                # Expected [socket_name, "reply_load_spool", json_kwargs]
                if len(msg_parts) >= 3:
                     func_part = msg_parts[1]
                     func_str = func_part.decode('utf-8') if isinstance(func_part, bytes) else func_part
                     
                     if func_str == "reply_load_spool":
                         kwargs_part = msg_parts[2]
                         try:
                             kwargs_str = kwargs_part.decode('utf-8') if isinstance(kwargs_part, bytes) else kwargs_part
                             kwargs = json.loads(kwargs_str)
                             status = kwargs.get("status")
                             console.info(f"SUCCESS: Received reply_load_spool with status: {status}")
                             reply_received = True
                             break
                         except:
                             pass
                             
                for part in msg_parts:
                     # Fallback
                    try:
                        decoded = json.loads(part.decode('utf-8'))
                        if decoded.get("function") == "reply_load_spool":
                            kwargs = decoded.get("kwargs", {})
                            status = kwargs.get("status")
                            console.info(f"SUCCESS: Received reply_load_spool with status: {status}")
                            reply_received = True
                            break
                    except:
                        pass
                if reply_received: break
            
            if reply_received:
                received_msgs.clear()
                break
        time.sleep(0.1)
    
    if not reply_received:
        console.error("FAILURE: Did not receive reply_load_spool")

    # Cleanup
    dealer.destroy_socket()
    zpipe_destroy_pipe()
    console.info("Test Finished")

if __name__ == "__main__":
    test_zapi()
