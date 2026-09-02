"""
ZMQ Pipeline Module (named ZPipe)
@author Byunghun Hwang <bh.hwang@iae.re.kr>
"""

import zmq
import threading
import time
from typing import Callable, Optional, List, Union, Dict
from util.logger.console import ConsoleLogger


class AsyncZSocket:
    """Generalized socket wrapper for push, pull, publish, subscribe patterns"""
    
    def __init__(self, socket_id: str, pattern: str):
        self.socket_id = socket_id
        self.pattern = pattern
        self.socket = None
        self.transport = None
        self.address = None
        self.port = None
        self.is_server = False
        self.collect_callback = None
        self.stop_event = None
        self.worker_thread = None
        self.is_created = False
        self.is_joined = False
        self.console = ConsoleLogger.get_logger()
        
        # Validate pattern
        valid_patterns = ['publish', 'subscribe', 'push', 'pull', 'dealer', 'router', 'server_pair', 'client_pair']
        if pattern not in valid_patterns:
            raise ValueError(f"Invalid pattern: {pattern}. Valid options: {valid_patterns}")
    
    def create(self, pipeline:Optional['ZPipe']) -> bool:
        """Create ZMQ socket with the given context and register with ZPipe"""
        if self.is_created:
            self.console.warning(f"Socket {self.socket_id} already created")
            return True
            
        try:
            # Create socket based on pattern
            socket_types = {
                'publish': zmq.PUB,
                'subscribe': zmq.SUB,
                'push': zmq.PUSH,
                'pull': zmq.PULL,
                'router': zmq.ROUTER,
                'dealer': zmq.DEALER,
                'server_pair': zmq.PAIR,
                'client_pair': zmq.PAIR
            }

            self.socket = pipeline.get_context().socket(socket_types[self.pattern])
            
            # Configure socket options
            self.socket.setsockopt(zmq.LINGER, 0)
            
            if self.pattern in ['subscribe', 'pull', 'dealer', 'server_pair']:
                self.socket.setsockopt(zmq.RCVHWM, 100)
                self.socket.setsockopt(zmq.RCVTIMEO, 100)
                self.socket.setsockopt(zmq.LINGER, 0)
                self.socket.setsockopt(zmq.RECONNECT_IVL, 500)
                
                if self.pattern == 'dealer':
                    # Send an empty message to the ROUTER upon connection
                    self.socket.setsockopt(zmq.PROBE_ROUTER, 1)
                
            if self.pattern in ['publish', 'push', 'router', 'client_pair']:
                self.socket.setsockopt(zmq.SNDHWM, 100)
                self.socket.setsockopt(zmq.SNDTIMEO, 100)
                self.socket.setsockopt(zmq.LINGER, 0)
                self.socket.setsockopt(zmq.RECONNECT_IVL, 500)
                
            if self.pattern == 'router':
                # Enable ROUTER to receive events when peers connect/disconnect
                self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
                if hasattr(zmq, 'MAP_ROUTER_MAC'):
                    self.socket.setsockopt(zmq.MAP_ROUTER_MAC, 1) # Sometimes needed
                
            # Register this socket with ZPipe for management
            pipeline.register_socket(self)
            
            self.is_created = True
            self.console.debug(f"Created socket {self.socket_id} with pattern {self.pattern}")
            return True
            
        except Exception as e:
            self.console.error(f"Failed to create socket {self.socket_id}: {e}")
            return False
    
    def join(self, transport: str, address: str = "localhost", port: int = 5555) -> bool:
        """Join socket by connecting or binding to address (auto-determined by pattern)"""
        if not self.is_created or not self.socket:
            self.console.error(f"Socket {self.socket_id} not created. Call create_socket first.")
            return False
            
        if self.is_joined:
            self.console.warning(f"Socket {self.socket_id} already joined")
            return True
            
        try:
            # Validate transport
            valid_transports = ['tcp', 'inproc', 'ipc', 'pgm', 'epgm']
            if transport not in valid_transports:
                self.console.error(f"Invalid transport: {transport}. Valid options: {valid_transports}")
                return False
            
            # Determine if socket should bind (server) or connect (client) based on pattern
            server_patterns = ['publish', 'pull', 'router', 'server_pair']  # These patterns typically bind
            client_patterns = ['subscribe', 'push', 'dealer', 'client_pair']  # These patterns typically connect
                
            if self.pattern in server_patterns:
                self.is_server = True
            elif self.pattern in client_patterns:
                self.is_server = False
            else:
                self.console.error(f"Unknown pattern behavior for {self.pattern}")
                return False
            
            # Store connection parameters
            self.transport = transport
            self.address = address
            self.port = port
            
            # Build connection string
            if transport in ['inproc', 'ipc']:
                conn_str = f"{transport}://{address}"
            else:  # tcp, pgm, epgm
                # For binding, address='*' is valid relative to transport (tcp://*:port)
                # For connecting, address='*' is invalid.
                if self.is_server and address in ["*", "0.0.0.0"]:
                     conn_str = f"{transport}://*:{port}"
                else:
                     conn_str = f"{transport}://{address}:{port}"
            
            # Bind or connect based on pattern
            if self.is_server:
                self.socket.bind(conn_str)
                self.console.debug(f"Socket {self.socket_id} ({self.pattern}) bound to: {conn_str}")
            else:
                self.socket.connect(conn_str)
                self.console.debug(f"Socket {self.socket_id} ({self.pattern}) connected to: {conn_str}")
            
            self.is_joined = True
            
            # Start receiver thread for subscriber and pull patterns if callback is set
            if self.pattern in ['subscribe', 'pull', 'router', 'dealer', 'client_pair', 'server_pair'] and self.collect_callback:
                self._start_receiver_thread()
            
            return True
            
        except Exception as e:
            self.console.error(f"Failed to join socket {self.socket_id}: {e}")
            return False
    
    def destroy_socket(self):
        """Destroy ZMQ socket and cleanup resources"""
        # Stop receiver thread
        if self.stop_event:
            self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)
        
        # Unregister from ZPipe if still registered
        if ZPipe._instance and self.socket_id in ZPipe._sockets:
            ZPipe.unregister_socket(self.socket_id)
        
        # Close socket
        if self.socket:
            self.socket.close()
            self.socket = None
        
        self.is_created = False
        self.is_joined = False
        self.console.debug(f"Destroyed socket {self.socket_id}")
    
    def set_message_callback(self, callback: Callable[[List[bytes]], None]) -> bool:
        """Set callback function for receiving multipart data"""
        if not callable(callback):
            self.console.error("Callback must be callable")
            return False
            
        self.collect_callback = callback
            
        return True
    
    def dispatch(self, data: List[Union[str, bytes]]) -> bool:
        """Send multipart data through the socket"""
        if not self.socket or not self.is_joined:
            self.console.error("Socket not joined")
            return False
            
        if self.pattern not in ['publish', 'push', 'server_pair', 'client_pair', 'dealer', 'router']:
            self.console.error("dispatch() can only be called on publish, push, or pair patterns")
            return False
            
        try:
            # Convert strings to bytes
            byte_data = []
            for item in data:
                if isinstance(item, str):
                    byte_data.append(item.encode('utf-8'))
                else:
                    byte_data.append(item)
            
            # Send multipart message
            self.socket.send_multipart(byte_data)
            return True
            
        except zmq.Again:
            self.console.warning("Send would block - message not sent")
            return False
        except Exception as e:
            self.console.error(f"Failed to dispatch data: {e}")
            return False
    
    def subscribe(self, topic: Union[str, bytes] = "") -> bool:
        """Subscribe to specific topic (only for subscribe pattern)"""
        if self.pattern != 'subscribe':
            self.console.error("subscribe() can only be called on subscribe pattern")
            return False
            
        try:
            if isinstance(topic, str):
                topic = topic.encode('utf-8')
            self.socket.setsockopt(zmq.SUBSCRIBE, topic)
            self.console.debug(f"Subscribed to topic: {topic}")

            if self.is_joined:
                self._start_receiver_thread()

            return True
        except Exception as e:
            self.console.error(f"Failed to subscribe to topic: {e}")
            return False
    
    def unsubscribe(self, topic: Union[str, bytes] = "") -> bool:
        """Unsubscribe from specific topic (only for subscribe pattern)"""
        if self.pattern != 'subscribe':
            self.console.error("unsubscribe() can only be called on subscribe pattern")
            return False
            
        try:
            if isinstance(topic, str):
                topic = topic.encode('utf-8')
            self.socket.setsockopt(zmq.UNSUBSCRIBE, topic)
            self.console.debug(f"Unsubscribed from topic: {topic}")
            return True
        except Exception as e:
            self.console.error(f"Failed to unsubscribe from topic: {e}")
            return False
    
    def _start_receiver_thread(self):
        """Start the receiver thread for handling incoming messages"""
        if self.worker_thread and self.worker_thread.is_alive():
            return  # Thread already running
            
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(
            target=self._receiver_worker,
            args=(self.stop_event,),
            daemon=True
        )
        self.worker_thread.start()
        self.console.debug(f"Receiver thread started for socket {self.socket_id}")
    
    def _receiver_worker(self, stop_event: threading.Event):
        """Worker thread for receiving messages"""
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        
        while not stop_event.is_set():
            try:
                events = dict(poller.poll(1000))  # Poll with 1 second timeout
                
                if self.socket in events and events[self.socket] == zmq.POLLIN:
                    # Receive multipart message
                    multipart_data = self.socket.recv_multipart(zmq.NOBLOCK)
                    
                    # Call the callback with received data
                    if self.collect_callback:
                        try:
                            self.collect_callback(multipart_data)
                        except Exception as e:
                            self.console.error(f"Error in collect callback: {e}")
                            
            except zmq.Again:
                continue  # Timeout, continue polling
            except zmq.ContextTerminated:
                # Context was terminated, exit gracefully
                self.console.debug(f"Context terminated, stopping receiver thread for {self.socket_id}")
                break
            except Exception as e:
                self.console.error(f"Error in receiver worker: {e}")
                break
        
        # Cleanup
        try:
            poller.unregister(self.socket)
        except (zmq.ContextTerminated, AttributeError):
            pass  # Context already terminated or socket is None
        self.console.debug(f"Receiver thread stopped for socket {self.socket_id}")
    
    def close(self):
        """Close the socket and cleanup resources (alias for destroy_socket)"""
        self.destroy_socket()


class ZPipe:
    """Singleton ZMQ Pipeline manager - manages ZMQ context and registered sockets"""
    
    _instance = None
    _context = None
    _sockets: Dict[str, AsyncZSocket] = {}
    _lock = threading.RLock()
    _console = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ZPipe, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def create_pipe(cls, io_threads: int = 1) -> 'ZPipe':
        """Create or get the singleton ZPipe instance"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                cls._context = zmq.Context(io_threads=io_threads)
                cls._console = ConsoleLogger.get_logger()
                cls._console.debug(f"Created ZPipe with {io_threads} IO threads")
        return cls._instance

    @classmethod
    def get_context(cls) -> Optional['zmq.Context']:
        """Get the ZMQ context"""
        return cls._context
    
    @classmethod
    def register_socket(cls, socket: AsyncZSocket) -> bool:
        """Register a socket with ZPipe for management"""
        if cls._instance is None:
            cls._console.error("ZPipe not initialized. Call create_pipe() first.")
            return False
            
        with cls._lock:
            if socket.socket_id in cls._sockets:
                cls._console.warning(f"Socket {socket.socket_id} already registered")
                return False
            
            cls._sockets[socket.socket_id] = socket
            cls._console.debug(f"Registered socket {socket.socket_id}")
            return True
    
    @classmethod
    def unregister_socket(cls, socket_id: str) -> bool:
        """Unregister a socket from ZPipe management"""
        with cls._lock:
            if socket_id in cls._sockets:
                del cls._sockets[socket_id]
                cls._console.debug(f"Unregistered socket {socket_id}")
                return True
            return False
    
    @classmethod
    def get_socket(cls, socket_id: str) -> Optional['AsyncZSocket']:
        """Get a registered socket by its ID"""
        return cls._sockets.get(socket_id)
    
    @classmethod
    def destroy_pipe(cls):
        """Destroy the singleton instance and close all registered sockets"""
        with cls._lock:
            # Properly destroy all registered sockets (this stops receiver threads first)
            for socket_id, zsocket in list(cls._sockets.items()):
                zsocket.destroy_socket()
            cls._sockets.clear()
            
            # Terminate context after all threads are stopped
            if cls._context:
                cls._context.term()
                cls._context = None
            
            # Reset instance
            cls._instance = None
            
            if cls._console:
                cls._console.debug("ZPipe destroyed - all connections closed and context terminated")
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if ZPipe is initialized"""
        return cls._instance is not None and cls._context is not None
    
    @classmethod
    def get_socket_count(cls) -> int:
        """Get the number of registered sockets"""
        return len(cls._sockets)
    
    @classmethod
    def list_sockets(cls) -> List[str]:
        """Get list of registered socket IDs"""
        return list(cls._sockets.keys())

# Global functions for singleton access
def zpipe_create_pipe(io_threads: int = 1) -> 'ZPipe':
    """
    Create the singleton ZPipe instance with specified IO threads
    
    Args:
        io_threads: Number of IO threads for ZMQ context (default: 1)
        
    Returns:
        ZPipe singleton instance
        
    Raises:
        RuntimeError: If ZPipe is already initialized
    """
    # Allow getting existing instance if already initialized
    if ZPipe.is_initialized():
        return ZPipe._instance
    
    return ZPipe.create_pipe(io_threads)


def zpipe_destroy_pipe():
    """
    Destroy the singleton ZPipe instance and close all connections
    """
    ZPipe.destroy_pipe()


def get_socket(socket_id: str) -> Optional[AsyncZSocket]:
    """Get a registered socket by its ID"""
    return ZPipe.get_socket(socket_id)


def is_initialized() -> bool:
    """Check if ZPipe is initialized"""
    return ZPipe.is_initialized()


def get_socket_count() -> int:
    """Get the number of registered sockets"""
    return ZPipe.get_socket_count()


def list_sockets() -> List[str]:
    """Get list of registered socket IDs"""
    return ZPipe.list_sockets()
