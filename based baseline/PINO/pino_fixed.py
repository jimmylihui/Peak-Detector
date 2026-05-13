"""Fixed J wave detection by Pino et al. (2015)

E. J. Pino, J. A. Chavez, and P. Aqueveque, 'Noninvasive ambulatory
measurement system of cardiac activity,' in Conf. Proc. IEEE Eng.
Med. Biol. Soc., 2015, pp. 7622-7625

Fixes vs original bcg-hr-dl implementation:
  1. Wavelet detail selection: keeps D4-D7 (coeffs indices 2-5)
     instead of D3-D6 (coeffs indices 3-6)
  2. D1 detail leak: loop now covers all coeffs (levels+1) so
     coeffs[8] (cD1) is properly zeroed
"""

import numpy as np
import scipy
import scipy.ndimage
import scipy.signal as sgnl
import pandas as pd
import pywt


def _get_padded_window(x, i, n, padding_value=0.):
    """Get padded window centered at index i with total size ~n."""
    x = np.asarray(x)
    nbefore = int(np.ceil(n / 2.))
    nafter = int(n // 2)

    left_pad = [padding_value] * max(0, nbefore - i)
    right_pad = [padding_value] * max(0, i + nafter - len(x))

    return np.concatenate([left_pad,
                           x[max(0, i - nbefore):min(i + nafter, len(x))],
                           right_pad])


def _filter_lowpass(x, f, cutoff, order=2):
    """Butterworth lowpass, zero-phase (filtfilt)."""
    coeffs = sgnl.butter(N=order, Wn=np.divide(cutoff, f / 2.),
                         btype="lowpass")
    return sgnl.filtfilt(coeffs[0], coeffs[1], x)


def wavelet_signal_separation(x, wavelet="db6", levels=8,
                              details=(2, 3, 4, 5)):
    """Extract BCG signal from raw data using wavelet decomposition.

    pywt.wavedec returns [cA_n, cD_n, cD_{n-1}, ..., cD_1].
    For levels=8 that is 9 elements (indices 0-8):
        coeffs[0]=cA8, coeffs[1]=cD8, coeffs[2]=cD7, ..., coeffs[8]=cD1

    Paper says keep D4-D7 → coeffs indices {2, 3, 4, 5}.
    """
    coeffs = pywt.wavedec(x, wavelet, level=levels)
    for i in range(levels + 1):
        if i not in details:
            coeffs[i][:] = 0.
    return pywt.waverec(coeffs, wavelet)


def length_transform(x, f, window_length=0.3, center=True):
    """Apply length transform to preprocessed signal."""
    winsize = int(f * window_length)
    xs = pd.Series(np.sqrt((x[1:] - x[:-1]) ** 2 + 1))
    return xs.rolling(winsize, min_periods=1, center=center).sum().values


def smoothing(x, f, window_length=0.3):
    """Apply smoothing with moving average window."""
    winsize = int(f * window_length)
    return scipy.ndimage.convolve1d(x, np.divide(np.ones(winsize), winsize),
                                    mode="nearest")


def first_elimination(lt, f, indices, window_length=0.3):
    """Eliminate peaks that are not true maxima within a window."""
    def is_maximum(i):
        winmax = _get_padded_window(lt, i, int(f * window_length)).max()
        return winmax <= lt[i]
    return list(filter(lambda i: is_maximum(i), indices))


def relocate_indices(x, f, indices, search_window=0.4):
    """Refine peak locations on the wavelet-reconstructed signal."""
    winsize = int(f * search_window)
    js = indices[:]
    for i, ind in enumerate(indices):
        js[i] = (ind - winsize // 2
                 + np.argmax(_get_padded_window(x, ind, winsize)))
    return js


def second_elimination(bcg, f, indices, dist=0.3):
    """Discard J wave locations that are too close to each other."""
    dist = int(f * dist)
    inds = indices[:]
    i = 1
    while i < len(inds):
        if inds[i] - inds[i - 1] <= dist:
            if bcg[inds[i]] > bcg[inds[i - 1]]:
                del inds[i - 1]
            else:
                del inds[i]
        else:
            i += 1
    return inds


def pino_fixed(x, f, low_cutoff=30., lt_window=0.3, smoothing_window=0.3,
               order=2, mother="db6", levels=8, details=(2, 3, 4, 5),
               elimination_window=0.6, search_window=0.4, min_dist=0.3):
    """J wave detection by Pino et al. (2015) — fixed version.

    Fixes:
      - Wavelet details D4-D7 (coeffs indices 2-5) instead of D3-D6
      - Loop covers all coeffs (levels+1) to prevent D1 leak
    """
    x = _filter_lowpass(x, f, low_cutoff, order=order)

    bcg = wavelet_signal_separation(x, wavelet=mother, levels=levels,
                                    details=details)

    lt = length_transform(bcg, f, window_length=lt_window, center=True)
    lt = smoothing(lt, f, window_length=smoothing_window)

    indices = sgnl.find_peaks(lt)[0]

    indices = first_elimination(lt, f, indices, elimination_window)
    j_indices = relocate_indices(bcg, f, indices, search_window)
    j_indices = second_elimination(bcg, f, j_indices, dist=min_dist)

    return j_indices
