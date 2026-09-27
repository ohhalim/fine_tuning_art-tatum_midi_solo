"""Raise the calling thread's macOS QoS class so the OS wakes it on time.

S1 (docs/experiments/RUNTIME_STALL_CAUSE.md) attributed most >10 ms dispatch
lateness to system-level wake-up delays that also hit an unrelated process.
``pthread_set_qos_class_self_np`` applies to the calling thread only, so call
it from the thread that runs the scheduler.
"""
from __future__ import annotations

import ctypes
import sys

QOS_CLASSES = {
    "user-interactive": 0x21,
    "user-initiated": 0x19,
    "default": 0x15,
    "utility": 0x11,
}


def set_current_thread_qos(name: str) -> dict:
    """Returns {"requested", "applied", "error"}; never raises."""
    if name not in QOS_CLASSES:
        return {"requested": name, "applied": False, "error": "unknown QoS class"}
    if sys.platform != "darwin":
        return {"requested": name, "applied": False, "error": "not macOS"}
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        fn = libc.pthread_set_qos_class_self_np
        fn.argtypes = [ctypes.c_uint, ctypes.c_int]
        fn.restype = ctypes.c_int
        rc = fn(QOS_CLASSES[name], 0)
    except (OSError, AttributeError) as exc:
        return {"requested": name, "applied": False, "error": str(exc)}
    return {"requested": name, "applied": rc == 0, "error": None if rc == 0 else f"rc={rc}"}


def current_thread_qos() -> str | None:
    """Name of the calling thread's QoS class, or None when unavailable."""
    if sys.platform != "darwin":
        return None
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        fn = libc.pthread_get_qos_class_np
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)]
        fn.restype = ctypes.c_int
        libc.pthread_self.restype = ctypes.c_void_p
        qos, rel = ctypes.c_uint(), ctypes.c_int()
        if fn(libc.pthread_self(), ctypes.byref(qos), ctypes.byref(rel)) != 0:
            return None
    except (OSError, AttributeError):
        return None
    return next((k for k, v in QOS_CLASSES.items() if v == qos.value), hex(qos.value))
