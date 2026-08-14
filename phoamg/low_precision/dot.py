"""The matrix-vector product as the photonic hardware computes it.

`dot_low_vector` is the entry point, composing the three stages the device imposes on
A @ x: noise on the operator entries (`make_A_low`), fixed-point quantization
of the input vector plus its own noise (`make_x_low`), and requantization of the result
to the ADC's resolution (`output_quantization`). Moving between the fixed-point and
floating-point domains is left to `phoamg.low_precision.scale`; callers pass A already
quantized, together with the factor that quantized it.

The callers are `MultiGrid.dot_low_precision` and `TwoGridQCD.dot_low_precision`;
`reproduce_results/photonic_dot.py` wraps this same function as a standalone emulation
of the device.
"""

import numpy as np

from phoamg.low_precision.scale import clip_complex, scale_factor

def make_A_low(A, max_value, A_factor, noise_strength=None):
    """Quantized operator plus noise added to its entries."""
    A_low = A.copy()
    if noise_strength is not None:
        if np.iscomplex(A):
            A_low.data += np.random.normal(loc=0, scale=noise_strength/A_factor, size=A_low.data.shape) + 1j* np.random.normal(loc=0, scale=noise_strength/A_factor, size=A_low.data.shape)
            A_low.data = clip_complex(A_low.data, max_value/A_factor)
        else:
            A_low.data += np.random.normal(loc=0, scale=noise_strength/A_factor, size=A_low.data.shape)
            A_low.data = np.clip(A_low.data, -max_value/A_factor, max_value/A_factor)

    return A_low

def make_x_low(x, max_value, noise_strength=None):
    """Quantized input vector plus noise; also returns the scale factor used."""
    x_factor = scale_factor(x, max_value=max_value)
    x_low = np.round(x*x_factor)
    if noise_strength is not None:
        if np.all(np.iscomplex(x_low)):
            x_low += np.random.normal(loc=0, scale=noise_strength, size=x_low.shape) + 1j*np.random.normal(loc=0, scale=noise_strength, size=x_low.shape)
            x_low = clip_complex(x_low, max_value)
        else:
            x_low += np.random.normal(loc=0, scale=noise_strength, size=x_low.shape)
            x_low = np.clip(x_low, -max_value, max_value)
    return x_low, x_factor

def output_quantization(bit_precision_adc, max_value, result):
    """Requantize a product to the ADC's resolution, as the detector would."""
    max_value_adc = 2**(bit_precision_adc-1) - 1
    max_value_ratio = max_value_adc/max_value**2
    result = result*(max_value_ratio)
    if np.all(np.iscomplex(result)):
        result = clip_complex(result, max_value_adc)
    else:
        result = np.clip(result, -max_value_adc, max_value_adc)
    result = np.round(result)/(max_value_ratio)
    return result


def dot_low_vector(A,x,max_value, A_factor,bit_precision_adc=None, noise_strength=None):
    """A @ x as the hardware computes it: quantize both, multiply, quantize the output."""
    A_low = make_A_low(A, max_value, A_factor, noise_strength=noise_strength)
    x_low, x_factor = make_x_low(x, max_value, noise_strength=noise_strength)
    result = A_low@(x_low)

    if bit_precision_adc is None:
        result = result/(x_factor)
    else:
        result = output_quantization(bit_precision_adc, max_value, result*A_factor)/(A_factor*x_factor)

    return result


