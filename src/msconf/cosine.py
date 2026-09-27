"""Greedy cosine similarity (matchms CosineGreedy semantics) in numba, for the secondary engine."""
import numpy as np
from numba import njit


@njit(cache=True)
def cosine_greedy(a, b, tol, intensity_power):
    na = a.shape[0]
    nb = b.shape[0]
    if na == 0 or nb == 0:
        return 0.0
    ia = a[:, 1] ** intensity_power
    ib = b[:, 1] ** intensity_power
    # candidate pairs within tolerance
    ci = np.empty(na * 4 + nb * 4, np.int64)
    cj = np.empty(na * 4 + nb * 4, np.int64)
    cs = np.empty(na * 4 + nb * 4, np.float64)
    n = 0
    j0 = 0
    for i in range(na):
        while j0 < nb and b[j0, 0] < a[i, 0] - tol:
            j0 += 1
        j = j0
        while j < nb and b[j, 0] <= a[i, 0] + tol:
            if n >= ci.shape[0]:
                break
            ci[n] = i
            cj[n] = j
            cs[n] = ia[i] * ib[j]
            n += 1
            j += 1
    if n == 0:
        return 0.0
    order = np.argsort(-cs[:n])
    used_a = np.zeros(na, np.bool_)
    used_b = np.zeros(nb, np.bool_)
    tot = 0.0
    for k in range(n):
        p = order[k]
        if not used_a[ci[p]] and not used_b[cj[p]]:
            used_a[ci[p]] = True
            used_b[cj[p]] = True
            tot += cs[p]
    return tot / (np.sqrt((ia ** 2).sum()) * np.sqrt((ib ** 2).sum()))
