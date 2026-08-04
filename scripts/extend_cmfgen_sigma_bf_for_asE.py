#!/usr/bin/env python3
"""Extend cmfgen_sigma_bf.bin from capraise (33829 lev) to asE (37513 lev).

For each row in the new asE levels.csv:
  - If row's (Z, ion, level_number) was in capraise → copy that level's
    has_cmfgen flag + sigma row from the old file.
  - Else (AS extension level, level_number >= N_cap_ion) → has_cmfgen=0,
    sigma row = zeros → Kramers fallback per-level (matches what new levels
    get for ALL their physics).

Layout (lumina_atomic.c:710-720):
  uint32 magic=0x434D4644, uint32 version=1, int32 n_levels, int32 n_freq
  double nu_min, double nu_max
  int8 has_cmfgen[n_levels]  +  pad to 8-byte alignment
  double sigma[n_levels * n_freq]
"""
import struct
from pathlib import Path
import numpy as np
import pandas as pd

import os
ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
_SUF = os.environ.get("CMFGEN_OUT_SUFFIX", "")
SRC_DIR = ROOT / f"data/tardis_reference_v3_femerge{_SUF}"
DST_DIR = ROOT / f"data/tardis_reference_v3_femerge{_SUF}_asE"

SRC_BIN = SRC_DIR / "cmfgen_sigma_bf.bin"
DST_BIN = DST_DIR / "cmfgen_sigma_bf.bin"

# remove the symlink first
if DST_BIN.exists() or DST_BIN.is_symlink():
    DST_BIN.unlink()

src_lev = pd.read_csv(SRC_DIR / "levels.csv")
dst_lev = pd.read_csv(DST_DIR / "levels.csv")
print(f"  src levels: {len(src_lev):,}   dst levels: {len(dst_lev):,}")

# Key: (Z, ion, level_number) → old row index in src
src_key = list(zip(src_lev.atomic_number, src_lev.ion_number, src_lev.level_number))
src_index_of = {k: i for i, k in enumerate(src_key)}

# For each dst row, find old index (-1 if extension)
dst_key = list(zip(dst_lev.atomic_number, dst_lev.ion_number, dst_lev.level_number))
mapping = np.array([src_index_of.get(k, -1) for k in dst_key], dtype=np.int64)
n_map = (mapping >= 0).sum()
n_ext = (mapping < 0).sum()
print(f"  mapped (carryover) = {n_map:,}    new (ext, zero/Kramers) = {n_ext:,}")
assert n_map == len(src_lev), "every src level must be carried over"

# Read source binary
with open(SRC_BIN, "rb") as f:
    magic, version, n_lev_src, n_freq = struct.unpack("<IIii", f.read(16))
    nu_min, nu_max = struct.unpack("<dd", f.read(16))
    assert magic == 0x434D4644 and version == 1
    assert n_lev_src == len(src_lev), f"src bin n_lev={n_lev_src} != csv {len(src_lev)}"
    print(f"  src bin: n_lev={n_lev_src} n_freq={n_freq} nu=[{nu_min:.3e},{nu_max:.3e}]")
    has_src = np.frombuffer(f.read(n_lev_src), dtype=np.int8).copy()
    pad = (8 - (n_lev_src % 8)) % 8
    if pad: f.read(pad)
    sigma_src = np.fromfile(f, dtype=np.float64, count=n_lev_src * n_freq)
sigma_src = sigma_src.reshape(n_lev_src, n_freq)
print(f"  src has_cmfgen ones: {int(has_src.sum())}/{n_lev_src} ({100*has_src.mean():.1f}%)")

# Build dst arrays
n_lev_dst = len(dst_lev)
has_dst = np.zeros(n_lev_dst, dtype=np.int8)
sigma_dst = np.zeros((n_lev_dst, n_freq), dtype=np.float64)

m = mapping >= 0
has_dst[m] = has_src[mapping[m]]
sigma_dst[m] = sigma_src[mapping[m]]
print(f"  dst has_cmfgen ones: {int(has_dst.sum())}/{n_lev_dst} ({100*has_dst.mean():.1f}%)")
print(f"  dst sigma carried: {(sigma_dst[m] != 0).any(axis=1).sum():,} nonzero rows from src")

# Write
with open(DST_BIN, "wb") as f:
    f.write(struct.pack("<IIii", 0x434D4644, 1, n_lev_dst, n_freq))
    f.write(struct.pack("<dd", nu_min, nu_max))
    f.write(has_dst.tobytes())
    pad = (8 - (n_lev_dst % 8)) % 8
    if pad: f.write(b"\x00" * pad)
    sigma_dst.tofile(f)

size = DST_BIN.stat().st_size
expected = 32 + n_lev_dst + ((8 - n_lev_dst % 8) % 8) + n_lev_dst * n_freq * 8
print(f"  wrote {DST_BIN}  size={size:,} (expected {expected:,})")
assert size == expected
