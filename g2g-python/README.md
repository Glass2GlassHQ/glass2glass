# g2g-python

Hosts `gst-python-ml` element shells as first-class
[glass2glass](https://github.com/boxerab/glass2glass) elements through embedded
CPython (pyo3), with zero-copy frame access over the Python buffer protocol and
detections routed into the frame analytics metadata.
