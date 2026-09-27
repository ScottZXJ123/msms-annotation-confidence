"""Spectrum parsing and processing. Identical for queries and library (failure mode #10)."""
from __future__ import annotations

import numpy as np
from ms_entropy import clean_spectrum


def parse_peaks(mzs: str, intensities: str) -> np.ndarray:
    mz = np.array(mzs.split(","), dtype=np.float64)
    it = np.array(intensities.split(","), dtype=np.float64)
    return np.stack([mz, it], 1).astype(np.float32)


def process(peaks: np.ndarray, precursor_mz: float, sp: dict) -> np.ndarray:
    """Return processed peaks (sorted, merged, noise-filtered, precursor region removed, sum-normalized)."""
    return clean_spectrum(
        peaks,
        min_mz=0,
        max_mz=precursor_mz - sp["remove_precursor_region_da"],
        noise_threshold=sp["noise_threshold_rel"],
        min_ms2_difference_in_da=sp["merge_peaks_within_da"],
        normalize_intensity=True,
    )


def spectral_entropy(peaks: np.ndarray) -> float:
    p = peaks[:, 1].astype(np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())
