# g2g-python

Hosts `gst-python-ml` element shells as first-class
[glass2glass](https://github.com/boxerab/glass2glass) elements through embedded
CPython (pyo3), with zero-copy frame access over the Python buffer protocol (CPU
frames) or `__cuda_array_interface__` / DLPack (CUDA frames, which stay on the
GPU), and detections routed into the frame analytics metadata.
