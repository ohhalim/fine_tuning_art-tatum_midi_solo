# For all things related to devices
#### ONLY USE PROVIDED FUNCTIONS, DO NOT USE GLOBAL CONSTANTS ####

import os
import torch

TORCH_CPU_DEVICE = torch.device("cpu")

if(torch.cuda.device_count() > 0):
    TORCH_CUDA_DEVICE = torch.device("cuda")
else:
    TORCH_CUDA_DEVICE = None

# Apple Silicon GPU (Metal / MPS). Detected once at import.
# Disable explicitly with env FORCE_CPU=1 (useful for debugging).
def _detect_mps():
    if os.environ.get("FORCE_CPU", "") == "1":
        return None
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
    except Exception:
        pass
    return None

TORCH_MPS_DEVICE = _detect_mps()

if(TORCH_CUDA_DEVICE is None and TORCH_MPS_DEVICE is None):
    print("----- WARNING: no CUDA or MPS device detected. Training will run on CPU (slow). -----")
    print("")

USE_CUDA = True

# use_cuda
def use_cuda(cuda_bool):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Sets whether to use the GPU (CUDA or MPS, if available), or force the CPU.
    Passing False forces CPU regardless of available accelerators.
    ----------
    """

    global USE_CUDA
    USE_CUDA = cuda_bool

# get_device
def get_device():
    """
    ----------
    Author: Damon Gwinn (MPS support added)
    ----------
    Grabs the default device. Preference order when use_cuda is not False:
    CUDA -> MPS (Apple Silicon) -> CPU. Passing use_cuda(False) forces CPU.
    Set env FORCE_CPU=1 to disable MPS detection at import time.
    ----------
    """

    if(not USE_CUDA):
        return TORCH_CPU_DEVICE
    if(TORCH_CUDA_DEVICE is not None):
        return TORCH_CUDA_DEVICE
    if(TORCH_MPS_DEVICE is not None):
        return TORCH_MPS_DEVICE
    return TORCH_CPU_DEVICE

# mps_device
def mps_device():
    """Grabs the MPS (Apple Silicon GPU) device, or None if unavailable."""

    return TORCH_MPS_DEVICE

# cuda_device
def cuda_device():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Grabs the cuda device (may be None if CUDA is not available)
    ----------
    """

    return TORCH_CUDA_DEVICE

# cpu_device
def cpu_device():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Grabs the cpu device
    ----------
    """

    return TORCH_CPU_DEVICE
