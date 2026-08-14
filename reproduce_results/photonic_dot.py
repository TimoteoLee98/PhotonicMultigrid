"""Standalone emulation of the photonic-hardware matrix-vector product.

A thin wrapper over phoamg's low-precision machinery, kept separate so the hardware
model is one readable function rather than something spread through the solver
classes. The reproduce_*.py scripts swap it into a solver in place of the built-in
low-precision product, which is how the "photonic" runs are produced.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_ROOT, '..')))

from phoamg.low_precision.dot import dot_low_vector
from phoamg.low_precision.scale import scale_factor

# The "noisy 8-bit" settings of the device this emulates, and the only settings any
# photonic run in this project uses: 8-bit quantization of A and x, an 8-bit ADC on the
# output, and Gaussian noise of strength 2**(noise_bit-1)/sqrt(2) with noise_bit=4.
# Fixed rather than arguments -- the model describes one device, not a family of them.
BIT_PRECISION = 8
BIT_PRECISION_ADC = 8
NOISE_STRENGTH = 2**3 / (2**0.5)


def photonic_dot(A, x):
    """Emulate a photonic-hardware matrix-vector product A @ x.

    A must be a scipy.sparse (CSR) matrix. The precision and noise settings are the
    module constants above; there is nothing to configure per call.
    """
    max_value = 2**(BIT_PRECISION - 1) - 1
    A_factor = scale_factor(A, max_value)
    return dot_low_vector(A, x, max_value, A_factor,
                           bit_precision_adc=BIT_PRECISION_ADC,
                           noise_strength=NOISE_STRENGTH)
