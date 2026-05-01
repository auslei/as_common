"""
Common ML utility functions.
"""

import os
import logging

# Fallback logger if main logger isn't available
logger = logging.getLogger("ml_utils")

def is_running_in_container() -> bool:
    """Detect if code is running inside a container."""
    if os.path.exists('/.dockerenv'):
        return True
    if os.path.exists('/run/.containerenv'):
        return True
    try:
        with open('/proc/1/cgroup', 'r') as f:
            content = f.read()
            if 'docker' in content or 'kubepods' in content or 'containerd' in content:
                return True
    except (FileNotFoundError, PermissionError):
        pass
    return False

def get_device() -> str:
    """
    Return 'cuda' if NVIDIA GPU is available and working, 'mps' if Apple Silicon is available, else 'cpu'.
    Can be overridden by setting the FORCE_DEVICE environment variable.
    
    Note: For ONNX Runtime (FastEmbed), this is used as a hint. 
    Actual acceleration depends on the installed execution providers.
    """
    # Allow manual override via environment variable
    forced_device = os.getenv("FORCE_DEVICE")
    if forced_device:
        print(f"ML_UTILS: Forced device to {forced_device}")
        return forced_device

    # FastEmbed / ONNX Runtime primarily relies on internal provider detection.
    # We still provide a hint for logic that might branch on device type.
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        
        if 'CUDAExecutionProvider' in providers:
            print("ML_UTILS: CUDA (ONNX) available.")
            return 'cuda'
        
        if 'CoreMLExecutionProvider' in providers or 'AppleExecutionProvider' in providers:
            print("ML_UTILS: Apple Silicon (ONNX) available.")
            return 'mps'
            
        print("ML_UTILS: Defaulting to CPU.")
        return 'cpu'
    except Exception as e:
        # Catch all and fallback to CPU
        print(f"ML_UTILS: Device detection failed: {e}")
        return 'cpu'

def get_gpu_usage() -> float:
    """
    Return current GPU memory usage percentage (0.0 to 100.0).
    Returns 0.0 if no GPU is available or on error.
    """
    # Without torch, we rely on nvidia-smi directly
    import subprocess
    try:
        cmd = "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"
        result = subprocess.check_output(cmd.split(), encoding='utf-8')
        used, total = map(int, result.strip().split(','))
        return (used / total) * 100.0
    except Exception:
        return 0.0

def millisec_to_time(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS string."""
    seconds = ms // 1000
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"
