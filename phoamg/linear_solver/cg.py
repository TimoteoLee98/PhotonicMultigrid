"""Conjugate gradient reporting its iteration count, and the tolerance helper.

`scipy.sparse.linalg.cg` returns a convergence flag, but the figures in this project
plot iteration counts, so `cg` below wraps SciPy's and counts iterations through its
`callback` hook.
"""

from scipy.sparse.linalg import cg as _scipy_cg


def get_atol_rtol(name, b_norm, atol=0., rtol=1e-5):
    """Normalise a relative tolerance against the right-hand side norm.

    Follows SciPy's convention: the effective absolute tolerance is
    `max(atol, rtol * ||b||)`. Kept here because the solvers need the same number
    SciPy computes internally, to decide their own outer stopping criteria.
    """
    if atol == 'legacy' or atol is None or atol < 0:
        msg = (f"'scipy.sparse.linalg.{name}' called with invalid `atol`={atol}; "
               "if set, `atol` must be a real, non-negative number.")
        raise ValueError(msg)

    atol = max(float(atol), float(rtol) * float(b_norm))

    return atol, rtol


def cg(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None):
    """Solve `A x = b` by conjugate gradient; return `(solution, iterations)`.

    Signature and semantics match `scipy.sparse.linalg.cg` except for the second
    return value, which is the number of iterations performed rather than SciPy's
    convergence flag. A user `callback` is still honoured.
    """
    iterations = [0]

    def counting_callback(xk):
        iterations[0] += 1
        if callback is not None:
            callback(xk)

    x, _ = _scipy_cg(A, b, x0=x0, rtol=rtol, atol=atol, maxiter=maxiter, M=M,
                     callback=counting_callback)
    return x, iterations[0]
