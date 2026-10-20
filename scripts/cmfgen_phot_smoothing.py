"""CMFGEN MODEL의 PHOT 전처리를 원본 SM_PHOT_V3 루틴으로 재현한다."""
from __future__ import annotations

import ctypes
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import re

import numpy as np

if __package__:
    from .cmfgen_parser import parse_phot
else:
    from cmfgen_parser import parse_phot


SOURCE_FILES = ("tools/set_kind_module.f", "tools/sm_phot_v3.f",
                "newsubs/cut_points_v3.f", "new_main/mod_subs/rd_control_variables.f")


def file_identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_smoothing_controls(model: Path) -> dict:
    values = {}
    for line in model.read_text().splitlines():
        match = re.search(r"\[(SIG_GAU_KMS|FRAC_SIG|CUT_ACC|ABV_EDGE)\]", line)
        if match:
            key = match.group(1)
            if key in values:
                raise ValueError(f"{model}: duplicate [{key}]")
            values[key] = line.split()[0]
    if "SIG_GAU_KMS" not in values:
        raise ValueError(f"{model}: missing [SIG_GAU_KMS]")
    # RDPHOT's optional defaults, RD_CONTROL_VARIABLES lines 437--444.
    flag = values.get("ABV_EDGE", "T").upper().strip(".")
    if flag not in {"T", "TRUE", "F", "FALSE"}:
        raise ValueError(f"{model}: invalid [ABV_EDGE]")
    controls = {
        "sigma_kms": float(values["SIG_GAU_KMS"].replace("D", "E")),
        "fraction_sigma": float(values.get("FRAC_SIG", ".25").replace("D", "E")),
        "cut_accuracy": float(values.get("CUT_ACC", ".02").replace("D", "E")),
        "above_edge": flag in {"T", "TRUE"},
    }
    if (not all(math.isfinite(controls[k]) for k in
                ("sigma_kms", "fraction_sigma", "cut_accuracy")) or
            controls["sigma_kms"] < 0.0 or controls["fraction_sigma"] <= 0.0 or
            controls["cut_accuracy"] <= 0.0):
        raise ValueError(f"{model}: invalid PHOT smoothing controls")
    return controls


class PhotSmoother:
    """덱 생성 시에만 원본 Fortran을 사용하며 생산 수송에는 의존성을 남기지 않는다."""

    def __init__(self, model: Path, library: Path | None = None,
                 source_root: Path | None = None):
        self.model = Path(model).resolve(strict=True)
        self.controls = read_smoothing_controls(self.model)
        self.library = Path(library).resolve(strict=True) if library else None
        self.sources = ([file_identity(Path(source_root) / rel) for rel in SOURCE_FILES]
                        if source_root else [])
        self._function = None
        self._cache = {}
        self.files = {}

    def _smooth(self, energy, sigma, width):
        if self._function is None:
            if self.library is None:
                raise RuntimeError(
                    "MODEL requires PHOT smoothing; set CMFGEN_PHOT_SMOOTH_LIB "
                    "to the library built from SM_PHOT_V3 and CUT_POINTS_V3")
            self._dll = ctypes.CDLL(str(self.library))
            self._function = self._dll.sm_phot_v3_
            real = ctypes.POINTER(ctypes.c_double)
            integer = ctypes.POINTER(ctypes.c_int)
            self._function.argtypes = [real, real, integer, integer,
                                       real, real, real, integer]
            self._function.restype = None
        energy = np.asarray(energy, dtype="f8")
        sigma = np.asarray(sigma, dtype="f8")
        if (energy.ndim != 1 or energy.size < 2 or sigma.shape != energy.shape or
                not np.isfinite(energy).all() or not np.isfinite(sigma).all() or
                np.any(np.diff(energy) < 0.0) or np.any(energy <= 0.0) or
                np.any(sigma < 0.0)):
            raise ValueError("invalid CMFGEN tabulated PHOT smoothing input")
        capacity = ctypes.c_int(max(200000, 4 * energy.size))
        count = ctypes.c_int(energy.size)
        x = np.zeros(capacity.value, dtype="f8")
        y = np.zeros_like(x)
        x[:count.value], y[:count.value] = energy, sigma
        p = ctypes.POINTER(ctypes.c_double)
        width = ctypes.c_double(width)
        fraction = ctypes.c_double(self.controls["fraction_sigma"])
        cut = ctypes.c_double(self.controls["cut_accuracy"])
        above = ctypes.c_int(self.controls["above_edge"])
        self._function(x.ctypes.data_as(p), y.ctypes.data_as(p),
                       ctypes.byref(count), ctypes.byref(capacity),
                       ctypes.byref(width), ctypes.byref(fraction),
                       ctypes.byref(cut), ctypes.byref(above))
        if not 2 <= count.value <= capacity.value:
            raise ValueError("SM_PHOT_V3 returned an invalid point count")
        x, y = x[:count.value].copy(), y[:count.value].copy()
        if (not np.isfinite(x).all() or not np.isfinite(y).all() or
                np.any(np.diff(x) <= 0.0) or np.any(y < 0.0)):
            raise ValueError("SM_PHOT_V3 returned invalid cross sections")
        return x, y

    def apply(self, path: Path, phot):
        key = str(Path(path).resolve(strict=True))
        if key in self._cache:
            return self._cache[key]
        old = phot.smoothing_sigma_kms
        if old is not None and (not math.isfinite(old) or old < 0.0):
            raise ValueError(f"{path}: invalid input PHOT smoothing sigma")
        target = self.controls["sigma_kms"]
        width = (math.sqrt(target**2 - old**2)
                 if old is not None and old < .9 * target else None)
        entries, n_smoothed = [], 0
        for entry in phot.entries:
            if width is not None and entry.cs_type in {20, 21, 22}:
                x, y = self._smooth(entry.energy, entry.sigma_Mb, width)
                entry = replace(entry, energy=x, sigma_Mb=y, n_points=len(x))
                n_smoothed += 1
            entries.append(entry)
        result = replace(phot, entries=entries,
                         n_data_pairs=sum(e.n_points for e in entries))
        self._cache[key] = result
        self.files[key] = {**file_identity(Path(path)),
                           "input_sigma_kms": old, "applied_sigma_kms": width,
                           "smoothed_entries": n_smoothed}
        return result

    def read(self, path: Path):
        return self.apply(path, parse_phot(path))

    def provenance(self) -> dict:
        return {"algorithm": "CMFGEN SM_PHOT_V3 + CUT_POINTS_V3",
                "model": file_identity(self.model), "controls": self.controls,
                "library": file_identity(self.library) if self.library else None,
                "sources": self.sources,
                "files": [self.files[k] for k in sorted(self.files)]}

    @classmethod
    def from_provenance(cls, record: dict):
        for identity in [record["model"], record["library"], *record["sources"]]:
            if identity and file_identity(Path(identity["path"])) != identity:
                raise ValueError("sealed PHOT preprocessing input changed")
        lib = record.get("library")
        result = cls(Path(record["model"]["path"]),
                     Path(lib["path"]) if lib else None)
        if result.controls != record["controls"]:
            raise ValueError("PHOT preprocessing controls disagree with MODEL")
        result.sources = record["sources"]
        return result
