"""Single-scale dispersion features for short BLE windows, with explicit guards.

DE follows the NCDF dispersion-pattern construction. SPDE adapts the supplied
SPDE_2.m time/frequency pattern mixing; no multiscale claim or MATLAB parity claim.
Background: Rostaghi & Azami (2016), doi:10.1109/LSP.2016.2542881.
"""
import itertools
from collections import Counter
import numpy as np
from scipy.special import ndtr


def classes(x, c=3, rescale=False):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite 1D sequence, length>=2')
    if not isinstance(c, int) or c < 2:
        raise ValueError('c must be integer>=2')
    sigma = float(np.std(x, ddof=1))
    if sigma <= 1e-12:
        return np.ones(len(x), dtype=int)
    y = ndtr((x - x.mean()) / sigma)
    if rescale:
        span = float(np.ptp(y))
        if span <= 1e-12:
            return np.ones(len(x), dtype=int)
        y = (y - y.min()) / span
        y = np.clip(y, 1e-10, 1-1e-10)
    # Positive MATLAB round(c*y+.5) is floor(c*y)+1, followed by endpoint guard.
    return np.clip(np.floor(c*y).astype(int)+1, 1, c)


def patterns(z, m=2, lag=1):
    if not isinstance(m, int) or m < 2 or not isinstance(lag, int) or lag < 1:
        raise ValueError('m>=2 and lag>=1 must be integers')
    length = len(z)-(m-1)*lag
    if length <= 0:
        raise ValueError('Sequence too short for embedding')
    return np.column_stack([z[j*lag:j*lag+length] for j in range(m)])


def pattern_entropy(matrix, c):
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError('Empty pattern matrix')
    _, count = np.unique(matrix, axis=0, return_counts=True)
    p = count/count.sum()  # Only strictly positive occupied-pattern counts.
    value = float(-np.sum(p*np.log(p))/np.log(c**matrix.shape[1]))
    return value, int(len(count)), int(len(matrix))


def dispersion_entropy(x, c=3, m=2, lag=1):
    return pattern_entropy(patterns(classes(x, c, rescale=False), m, lag), c)


def spde(x, c=3, m=2, lag=1, demean_fft=False):
    x = np.asarray(x, dtype=float)
    time_patterns = patterns(classes(x, c, rescale=True), m, lag)
    # Define constant-window entropy as zero, including the spectrum branch.
    if np.std(x, ddof=1) <= 1e-12:
        return 0., 1, (len(x)-(m-1)*lag)*(int(__import__('math').comb(2*m,m))-2)
    spectrum = np.abs(np.fft.fft(x-x.mean() if demean_fft else x))
    frequency_patterns = patterns(classes(spectrum, c, rescale=True), m, lag)
    mixed = []
    for i in range(1, m):
        for a in itertools.combinations(range(m), i):
            for b in itertools.combinations(range(m), m-i):
                mixed.append(np.column_stack([time_patterns[:, a], frequency_patterns[:, b]]))
    return pattern_entropy(np.vstack(mixed), c)


def self_test():
    checks = []
    grid = np.array(list(itertools.product(range(1,4), repeat=2)))
    assert np.isclose(pattern_entropy(grid,3)[0],1.)
    checks.append('All nine equiprobable patterns have normalized entropy1')
    for func in [dispersion_entropy, spde]:
        assert np.isclose(func(np.full(16,-70.))[0],0.)
    checks.append('Constant negative RSSI gives entropy0, finite and guarded')
    fixtures = [np.arange(16,dtype=float)-80, np.array([-60.,-75.]*8),
                np.array([-70,-68,-72,-69,-71,-65,-74,-68,-70,-72,-66,-75,-69,-73,-67,-71.])]
    for fixture in fixtures:
        a=dispersion_entropy(fixture)[0]
        assert np.isclose(a,dispersion_entropy(2*fixture+10)[0])
        matrix=patterns(classes(fixture,rescale=False))
        counts=Counter(map(tuple,matrix.tolist())); p=np.array(list(counts.values()))/len(matrix)
        assert np.isclose(a,-np.sum(p*np.log(p))/np.log(9))
        for demean in [False,True]: assert 0 <= spde(fixture,demean_fft=demean)[0] <= 1+1e-12
        assert np.isclose(spde(fixture,demean_fft=True)[0],spde(fixture+20,demean_fft=True)[0])
    checks.extend(['DE affine invariance on deterministic fixtures',
                   'Independent Counter calculation agrees with DE counts',
                   'SPDE bounds and demeaned FFT shift invariance on fixtures'])
    for bad in [np.array([1.]), np.array([1.,np.nan])]:
        try: dispersion_entropy(bad)
        except ValueError: pass
        else: raise AssertionError('Invalid input not rejected')
    checks.append('Short/nonfinite sequences rejected')
    return dict(passed=True, checks=checks,
                limitation='Python internal consistency tests only; no MATLAB numerical equivalence run.')
