import platform
import subprocess

try:
    import torch
except ImportError:
    torch = None

class GraphicDevice:
    def __init__(self):
        self._device_type = "cpu"
        self._device_name = "CPU"
        self._check_device()

    def _check_device(self):
        system = platform.system()
        processor = platform.processor()

        # Check for CUDA (NVIDIA)
        if torch and torch.cuda.is_available():
            self._device_type = "cuda"
            self._device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
            return

        # Default to CPU
        self._device_type = "cpu"
        self._device_name = "CPU"

    def get_device_type(self):
        """Returns 'cpu', 'cuda', or 'mps'"""
        return self._device_type

    def get_device_name(self):
        """Returns a human-readable name of the device"""
        return self._device_name

    def is_accelerated(self):
        return self._device_type != "cpu"
