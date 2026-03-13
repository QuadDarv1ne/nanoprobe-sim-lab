import sys
print("Python:", sys.executable)
try:
    import numpy
    print("NumPy:", numpy.__version__)
except ImportError as e:
    print("NumPy error:", e)
