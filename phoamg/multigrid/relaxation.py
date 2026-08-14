"""Relaxation methods for linear systems."""

from pyamg.util.linalg import norm, approximate_spectral_radius

def polynomial_dot(A, x, b, coefficient, dot_product, iterations=1):
    """Richardson smoothing of Ax=b, using `dot_product` for the matrix-vector product.

    Taking the product as an argument is what makes the smoother precision-agnostic:
    callers pass either an exact or an emulated-hardware product.

    `iterations` counts matrix-vector products, not updates to `x`. A step taken from
    x = 0 needs no product -- the residual is just b -- so it updates `x` without being
    counted, and a call with `iterations=n` from a zero start therefore performs n+1
    updates for n products. Both preconditioner paths start from zero (`MultiGrid.solve`
    and `TwoGridQCD._solve` allocate `x = np.zeros_like(b)`), so their pre-smoothing is
    one exact, free update followed by `numPre` counted ones. The operation counts the
    figures report are products, so they are unaffected by this; a caller starting from
    a nonzero `x` gets exactly `iterations` updates.
    """


    counter = 0
    while counter < iterations:

        if norm(x) == 0:
            residual = b
        else:
            result = dot_product(A, x)

            residual = b - result
            counter = counter + 1

        h = coefficient*residual

        x += h


def richardson_prolongation_smoother(S, T, omega=4.0/3.0, degree=1):
    """Smooth a tentative prolongator T against S, giving the final interpolation."""
    weight = omega/approximate_spectral_radius(S)

    P = T
    for _ in range(degree):
        P = P - weight*(S.dot(P))

    return P


