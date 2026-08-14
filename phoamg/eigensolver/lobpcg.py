"""LOBPCG with operator-application counters.

This is a thin wrapper around `scipy.sparse.linalg.lobpcg`, not a reimplementation.
The only thing this project needs beyond SciPy's solver is a count of how many
individual vector applications of the operator `A` and the preconditioner `M` a solve
costs, which is the quantity the figures plot. SciPy does not report that, so `A` and
`M` are wrapped in counting callables -- both accept callables -- and the count is
appended to whatever SciPy returns.
"""

import numpy as np
from scipy.sparse.linalg import lobpcg as _scipy_lobpcg

__all__ = ["lobpcg"]


def _counting(operator, tally, index):
    """Wrap `operator` in a callable that tallies the columns it is applied to.

    Counting columns rather than calls matches the convention of the figures: a block
    application to k vectors costs k operator applications.
    """
    if operator is None:
        return None

    apply_to = operator if callable(operator) else operator.__matmul__

    def counted(x):
        tally[index] += x.shape[1] if x.ndim > 1 else 1
        return apply_to(x)

    return counted


def lobpcg(A, X, B=None, M=None, Y=None, tol=None, maxiter=None, largest=True,
           verbosityLevel=0, retLambdaHistory=False, retResidualNormsHistory=False,
           restartControl=20):
    """`scipy.sparse.linalg.lobpcg`, returning `[A applications, M applications]` last.

    Every argument is passed straight through. The return value is SciPy's, with the
    counter array appended: e.g. with both histories requested it is
    `(eigenvalues, eigenvectors, lambdaHistory, residualNormsHistory, counters)`.
    """
    tally = [0, 0]
    result = _scipy_lobpcg(
        _counting(A, tally, 0), X, B=B, M=_counting(M, tally, 1), Y=Y,
        tol=tol, maxiter=maxiter, largest=largest, verbosityLevel=verbosityLevel,
        retLambdaHistory=retLambdaHistory,
        retResidualNormsHistory=retResidualNormsHistory,
        restartControl=restartControl,
    )
    if not isinstance(result, tuple):
        result = (result,)
    return (*result, np.array(tally))
