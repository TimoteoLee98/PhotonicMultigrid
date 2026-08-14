"""Fixed-point scaling helpers.

Quantizing to N bits means mapping the data onto integers in [-max_value, max_value].
`scale_factor` gives the multiplier that does that, and `clip_complex` keeps values
inside the range afterwards. Real and imaginary parts are treated as two independent
fixed-point channels throughout, which is what the hardware does.
"""

import numpy as np

def max_abs(matrix):
    return np.max(np.abs(matrix))

def max_complex(matrix):
    """Largest magnitude over both channels: max(max|Re|, max|Im|), not max|z|."""
    return np.max(np.array([max_abs(matrix.real.copy()), max_abs(matrix.imag.copy())]))

def scale_factor(vector, max_value=2**8-1):
    """Multiplier putting `vector`'s largest component exactly at `max_value`."""
    input_max = max_complex(vector)

    if input_max == 0:
        input_max = 1e-10  # Avoid division by zero

    input_factor = max_value/input_max


    return input_factor

def clip_complex(data, max_value):
    """Clip real and imaginary parts separately, which np.clip does not do."""
    return np.clip(data.real, -max_value, max_value) + np.clip(data.imag, -max_value, max_value)*1j

