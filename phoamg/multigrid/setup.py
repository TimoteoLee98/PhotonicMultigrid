"""Setup phase: building the interpolation basis the coarse grid is formed from.

The basis is found by relaxing random vectors, which leaves them dominated by the
error components the smoother cannot remove -- exactly what the coarse grid has to
represent. Two strategies are used here, selected through `run_setup`:
"standard" relaxes at using only one matrix-vector-multiplier, "mixed" alternates low- and
higher-precision passes. Setup is expensive but paid once and reused across solves.
"""

import numpy as np
from pyamg.util.linalg import norm


def start_random_guess(mg):
    x = np.random.randn(mg.A.shape[0], 1).astype(mg.A.dtype)
    if mg.A.dtype.name.startswith('complex'):
        x = x + 1.0j*np.random.randn(mg.A.shape[0], 1).astype(mg.A.dtype)
    return x

def generate_candidates(mg, numCandidates=8, numRelax=100):
    """Relax `numCandidates` random vectors into an interpolation basis."""

    x = start_random_guess(mg)

    for _ in range(numRelax):
        mg.relax(mg.A,x)

    x = x/norm(x, 'inf')
    B = x.reshape(-1,1)

    for _ in range(numCandidates - 1):
        x = np.random.randn(mg.A.shape[0]) + 1j*np.random.randn(mg.A.shape[0])
        for _ in range(numRelax):
            mg.relax(mg.A,x)

        x = x/norm(x, 'inf')
        B = np.hstack((B, x.reshape(-1, 1)))

    return B

def smooth_mixed(mg,x,b, numIter, numIter2):

    y = np.zeros_like(x)
    for _ in range(numIter):
        for _ in range(numIter2):
            mg.polynomial(mg.A,y,b)
        x = x + y
        b = -mg.A_mid.dot(x)
        y = np.zeros_like(x)


    return x,b

def setup_mixed_precision(mg, initial_relax=True, initial_precision=4,
                second_relax=False, second_precision=16,
                num5=1, num10=1, num20=30, numCandidates=8, noise_strength=None, bit_precision_adc=8):

    if second_relax:
        mg.change_relax_method(second_relax, second_precision)
        mg.A_mid = mg.A_low.copy()
    else:
        mg.A_mid = mg.A.copy()

    mg.change_relax_method(initial_relax, initial_precision, noise_strength, bit_precision_adc)

    x = start_random_guess(mg)
    b = -mg.A_mid.dot(x)

    x,b = smooth_mixed(mg,x,b, num5, 5)
    x,b = smooth_mixed(mg,x,b, num10, 10)
    x,b = smooth_mixed(mg,x,b, num20, 20)

    x = x/norm(x, 'inf')
    B = x.reshape(-1,1)

    for _ in range(numCandidates - 1):
        x = start_random_guess(mg)
        b = -mg.A_mid.dot(x)
        x,b = smooth_mixed(mg,x,b, num5, 5)
        x,b = smooth_mixed(mg,x,b, num10, 10)
        x,b = smooth_mixed(mg,x,b, num20, 20)

        x = x/norm(x, 'inf')
        B = np.hstack((B, x.reshape(-1, 1)))

    mg.create_operators(B)

def run_setup(mg, setup_name=None, initial_relax=None, initial_precision=None, numCandidates=8,
              numRelax=None, second_relax=None, second_precision=None,
              num5=1, num10=1, num20=30,
              noise_strength=None, bit_precision_adc=8):
    """Build `mg`'s interpolation basis with the named strategy and form its operators.

    'mixed' (also the default when `setup_name` is None) alternates low- and
    higher-precision relaxation passes; 'standard' relaxes at a single precision. The
    two strategies read different arguments, so each fills in its own defaults for the
    ones it uses and ignores the rest.
    """
    if setup_name == "mixed" or setup_name is None:
        if initial_precision is None:
            initial_precision = 4
        if second_precision is None:
            second_precision = 16

        if initial_relax is None:
            initial_relax = True
        if second_relax is None:
            second_relax = True

        setup_mixed_precision(mg, initial_relax=initial_relax, initial_precision=initial_precision,
            second_relax=second_relax, second_precision=second_precision,
            num5=num5, num10=num10, num20=num20, numCandidates=numCandidates,
            noise_strength=noise_strength, bit_precision_adc=bit_precision_adc)

    elif setup_name == "standard":
        if initial_precision is None:
            initial_precision = 8
        if initial_relax is None:
            initial_relax = True
        if numRelax is None:
            numRelax = 400

        mg.change_relax_method(initial_relax, initial_precision)
        B = generate_candidates(mg, numCandidates=numCandidates, numRelax=numRelax)
        mg.create_operators(B)

    else:
        raise ValueError(f"unknown setup_name {setup_name!r}; expected 'mixed' or 'standard'")

