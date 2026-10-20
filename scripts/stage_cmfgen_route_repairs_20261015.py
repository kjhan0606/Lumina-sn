#!/usr/bin/env python3
"""확인된 Si 영경로18개·S 누락7개만 새 BF 입력 사본에 적용한다.

기존 덱과 그 밖의 바이트를 보존한다. 도착별 Si B 생산 경로를 구현한 것은 아니다.
Slurm에서만 실행하며 출력 디렉터리가 있으면 거부한다.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
import pandas as pd

import cmfgen_parser as P
import expand_atomic_data_cmfgen as B


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def header(path):
    with path.open("rb") as handle:
        magic, version, nl, nf = struct.unpack("<IIii", handle.read(16))
        low, high = struct.unpack("<dd", handle.read(16))
        flags = np.frombuffer(handle.read(nl), dtype="u1").copy()
    if (magic, version) != (0x434D4644, 1):
        raise ValueError("unsupported BF file")
    return nl, nf, low, high, flags, 32+nl+(-nl)%8


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("compute job required")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir()
    original = a.model/"cmfgen_sigma_bf.bin"
    original_hash = sha(original)
    nl, nf, low, high, flags, offset = header(original)
    levels = pd.read_csv(a.model/"levels.csv")
    vintage = pd.read_csv(a.model/"atomic_vintage_manifest.csv")
    if len(levels) != nl:
        raise ValueError("level count")
    targets = {14:[2,3,4,33,34,35,36,37,38,39,64,68,69,70,71,74,75,76],
               16:[117,118,123,148,152,173,179]}
    ions, lookup, maps, sources = {}, {}, [], []
    for z, n in ((14,125),(16,322)):
        rows = levels[(levels.atomic_number == z) & (levels.ion_number == 1)]
        row = vintage[(vintage.atomic_number == z) & (vintage.ion_number == 1)].iloc[0]
        osc_path, phot_path = Path(row.osc_path), Path(row.phot_path)
        osc, phot = P.parse_osc(osc_path), P.parse_phot(phot_path)
        if len(rows) != n:
            raise ValueError("unexpected ion model")
        np.testing.assert_array_equal(rows.level_number, np.arange(n))
        np.testing.assert_array_equal(rows.configuration, osc.levels["config"][:n])
        if list(rows.level_number[flags[rows.index] == 0]) != targets[z]:
            raise ValueError("missing-route inventory changed")
        ions[z,2] = dict(osc=osc, levels=osc.levels, n_kept=n, phot=phot)
        sources.extend((osc_path,phot_path))
        for r in rows.itertuples():
            lookup[z,2,r.level_number+1] = len(maps)
            maps.append((z,int(r.level_number),int(r.Index)))
    small = a.out/"rebaked_SiII_SII.bin"
    with redirect_stdout(sys.stderr):
        B.bake_sigma_bf_grid(ions, {}, [None]*len(maps), lookup, small)
    sn, sf, sl, sh, present, so = header(small)
    if (sn,sf,sl,sh) != (len(maps),nf,low,high) or not np.all(present == 1):
        raise ValueError("rebaked model/grid or route presence")
    old = np.memmap(original, dtype="<f8", mode="r", offset=offset, shape=(nl,nf))
    new = np.memmap(small, dtype="<f8", mode="r", offset=so, shape=(sn,nf))
    changes = []
    for local,(z,lev,gl) in enumerate(maps):
        if flags[gl]:
            scale = max(float(np.max(abs(old[gl]))),float(np.max(abs(new[local]))))
            error = float(np.max(abs(old[gl]-new[local])))/scale if scale else 0
            if error > 1e-8:
                raise ValueError(("existing positive route changed",z,lev,error))
            continue
        if not np.all(old[gl] == 0):
            raise ValueError("unexpected nonzero absent route")
        if z == 14 and not np.all(new[local] == 0):
            raise ValueError("Si forbidden A route became positive")
        if z == 16 and not np.any(new[local] > 0):
            raise ValueError("S route not restored")
        changes.append(dict(Z=z,level=lev,global_level=gl,local=local))
    target = a.out/"cmfgen_sigma_bf.bin"
    shutil.copyfile(original,target)
    allowed = []
    with target.open("r+b") as handle:
        for change in changes:
            gl, local = change["global_level"],change["local"]
            handle.seek(32+gl); handle.write(b"\x01")
            start = offset+gl*nf*8
            handle.seek(start); handle.write(new[local].tobytes())
            allowed.extend(((32+gl,33+gl),(start,start+nf*8)))
    changed_bytes, position = 0, 0
    with original.open("rb") as f0, target.open("rb") as f1:
        while True:
            b0,b1 = f0.read(1 << 20),f1.read(1 << 20)
            if len(b0) != len(b1):
                raise ValueError("size changed")
            if not b0:
                break
            diff = np.frombuffer(b0,dtype="u1") != np.frombuffer(b1,dtype="u1")
            changed_bytes += int(diff.sum())
            for start,end in allowed:
                left,right = max(0,start-position),min(len(b0),end-position)
                if left < right:
                    diff[left:right] = False
            if np.any(diff):
                raise ValueError("unrequested input bytes changed")
            position += len(b0)
    if sha(original) != original_hash:
        raise ValueError("original mutated")
    sources.extend((original,a.model/"levels.csv",a.model/"atomic_vintage_manifest.csv",
                    Path(__file__),Path(P.__file__),Path(B.__file__)))
    report = dict(job=os.environ["SLURM_JOB_ID"],scope="isolated BF input copy; Si forbidden A and S seven routes only; Si B not implemented",
                  output=str(target),output_sha256=sha(target),bytes=target.stat().st_size,
                  changes=changes,changed_bytes=changed_bytes,other_bytes_unchanged=True,
                  source_sha256={str(p):sha(p) for p in sources})
    with (a.out/"repair.json").open("x") as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
        handle.write("\n")
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
