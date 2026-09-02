"""
ZAPI (ZeroMQ API) Base Class
Provides common definitions and helper functions for system-level signaling
"""

import json
from common.zpipe import AsyncZSocket

class ZAPIBase:
    def __init__(self):
        pass

    def call(self, socket: AsyncZSocket, function: str, kwargs: dict) -> bool:
        """
        Call a remote function via Zpipe socket using standardized multipart message format.
        Format: [socket_name, function, json_kwargs]
        
        Args:
            socket: AsyncZSocket instance
            function: Name of the function to call (str)
            kwargs: Dictionary of arguments
            
        Returns:
            bool: True if dispatched successfully
        """
        try:
            if socket:                
                socket_name = socket.socket_id
                
                parts = [
                    socket_name.encode('utf-8'),
                    function.encode('utf-8'),
                    json.dumps(kwargs).encode('utf-8')
                ]
                
                socket.dispatch(parts)
                return True
            else:
                return False
        except Exception:
            return False

    def call_raw(self, socket: AsyncZSocket, function: str, payload: bytes,
                 identity: bytes = None) -> bool:
        """
        Same multipart layout as call(), but the last frame carries a pre-encoded
        binary payload (e.g. pickle) instead of JSON. Used when arguments contain
        non-JSON data such as numpy arrays.
        Format: [(identity,) socket_name, function, payload]

        Args:
            socket: AsyncZSocket instance
            function: Name of the function to call (str)
            payload: Pre-encoded argument frame (bytes)
            identity: Peer identity, required when replying from a router socket

        Returns:
            bool: True if dispatched successfully
        """
        try:
            if socket:
                parts = [
                    socket.socket_id.encode('utf-8'),
                    function.encode('utf-8'),
                    payload
                ]
                if identity is not None:
                    parts.insert(0, identity)

                return socket.dispatch(parts)
            else:
                return False
        except Exception:
            return False
