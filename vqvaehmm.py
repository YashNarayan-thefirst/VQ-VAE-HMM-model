import numpy as np
# vector in the form: (f1,f2,f3...)
# Encode vals into floats/float vectors
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
pack = np.array([x, y])

def encode_vectors(x):
    return x.astype(np.float32)

def quantize_data(data, precision=2):
    return np.round(data, precision)
