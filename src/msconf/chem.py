"""Single RDKit structure-standardization pipeline used everywhere (failure mode #11)."""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")


class Standardizer:
    def __init__(self, cfg_structure: dict):
        self.c = cfg_structure
        self._frag = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
        self._unch = rdMolStandardize.Uncharger()
        self._taut = rdMolStandardize.TautomerEnumerator() if cfg_structure.get("tautomer_canonicalize") else None
        fp = cfg_structure["fingerprint"]
        self._fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=fp["radius"], fpSize=fp["nbits"])

    def mol(self, smiles: str):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        if self.c.get("cleanup", True):
            m = rdMolStandardize.Cleanup(m)
        if self.c.get("largest_fragment", True):
            m = self._frag.choose(m)
        if self.c.get("uncharge", True):
            m = self._unch.uncharge(m)
        if self._taut is not None:
            m = self._taut.Canonicalize(m)
        return m

    def record(self, smiles: str) -> dict:
        m = self.mol(smiles)
        if m is None:
            return {"std_smiles": None, "inchikey": None, "ik14": None}
        ik = Chem.MolToInchiKey(m)
        return {"std_smiles": Chem.MolToSmiles(m), "inchikey": ik, "ik14": ik[:14] if ik else None}

    def fingerprint(self, smiles: str):
        m = self.mol(smiles)
        return None if m is None else self._fpgen.GetFingerprint(m)


def pack_fps(fps, nbits: int) -> np.ndarray:
    """Bit-packed uint8 matrix (n, nbits/8) — compact storage (failure mode #12)."""
    out = np.zeros((len(fps), nbits // 8), dtype=np.uint8)
    for i, fp in enumerate(fps):
        arr = np.zeros((nbits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = np.packbits(arr)
    return out


def unpack_to_bitvects(packed: np.ndarray):
    fps = []
    for row in packed:
        bits = np.unpackbits(row)
        bv = DataStructs.ExplicitBitVect(len(bits))
        bv.SetBitsFromList(np.nonzero(bits)[0].tolist())
        fps.append(bv)
    return fps


def tanimoto_neighbors(fps, threshold: float):
    """Sparse neighbor lists: for each i, indices j != i with Tanimoto >= threshold (and the values)."""
    nbr_idx, nbr_sim = [], []
    for i in range(len(fps)):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps), dtype=np.float32)
        sims[i] = -1
        j = np.nonzero(sims >= threshold)[0]
        nbr_idx.append(j.astype(np.int32))
        nbr_sim.append(sims[j])
    return nbr_idx, nbr_sim
