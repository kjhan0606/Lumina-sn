#!/usr/bin/env python3
"""동일 O 물질에서 수송 수정이 Si/S의 표 기반 BF 율에 주는 영향.

표가 존재하는 경로만 비교하며 최종 이온비나 자기일관 스펙트럼으로 해석하지 않는다.
산란 폐합을 완료한 두 복사장이 입력이다. 생산 파일은 읽기만 한다.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pandas as pd

H, C, KB, EV = 6.62607015e-27, 2.99792458e10, 1.380649e-16, 1.602176634e-12


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--fields", required=True, type=Path)
    ap.add_argument("--base-fields", type=Path)
    a = ap.parse_args()
    base_fields = a.base_fields or a.fields
    cap = a.run / "work/det_obs_capture"
    model = a.run / "input/model"
    levels = pd.read_csv(model / "levels.csv")
    ions = pd.read_csv(model / "ionization_energies.csv")
    sigma_path = model / "cmfgen_sigma_bf.bin"
    with sigma_path.open("rb") as handle:
        magic, ver, nl, nf = struct.unpack("<IIii", handle.read(16))
        nmin, nmax = struct.unpack("<dd", handle.read(16))
        flags = np.frombuffer(handle.read(nl), dtype="u1")
    if (magic, ver, nl) != (0x434D4644, 1, len(levels)):
        raise ValueError("sigma identity")
    manifest = json.loads((cap / "manifest.json").read_text())
    if manifest["generations"]["material"] != 13 or manifest["generations"]["radiation"] != 13:
        raise ValueError("capture generation")
    T = np.fromfile(cap / "Te.f64", dtype="<f8")
    ns = len(T)
    nu = np.fromfile(cap / "nu.f64", dtype="<f8")
    capture_width = np.fromfile(cap / "dnu.f64", dtype="<f8")
    dlog = np.log(nmax/nmin)/nf
    edges = np.exp(np.log(nmin) + np.arange(nf+1)*dlog)
    np.testing.assert_allclose(nu, np.exp(np.log(nmin)+(np.arange(nf)+.5)*dlog), rtol=2e-12, atol=0)
    # Coarse transfer stores midpoint nu*dlog; BF rate integration uses the
    # actual bin edges. Verify each convention, rather than loosening a test.
    np.testing.assert_allclose(capture_width, nu*dlog, rtol=2e-12, atol=0)
    width = np.diff(edges)
    variants = ((base_fields,"base"), (a.fields,"corrected"))
    fields = [np.fromfile(folder / (v+".J.f64"), dtype="<f8").reshape(ns, nf)
              for folder,v in variants]
    for folder,variant in variants:
        result = json.loads((folder / (variant+".json")).read_text())
        if result["scattering_closure_residual"] > 1e-6:
            raise ValueError("unclosed scattering field")
    sigma = np.memmap(sigma_path, mode="r", dtype="<f8", offset=32+nl+(-nl)%8, shape=(nl,nf))
    records = []
    for z in (14,16):
        all_lower = levels[(levels.atomic_number == z) & (levels.ion_number == 1)]
        lo = all_lower[flags[all_lower.index] == 1]
        ion_e = ions[(ions.atomic_number == z) & (ions.ion_number == 1)].iloc[0,-1]
        edge = (float(ion_e)-lo.energy_eV.to_numpy())*EV/H
        lower = np.maximum(edges[:-1][None,:], edge[:,None])
        active = edges[1:][None,:] > lower
        ratio = np.divide(edges[1:][None,:], lower, out=np.ones_like(lower), where=active)
        scale = np.divide(width[None,:], edges[1:][None,:]-lower,
                          out=np.zeros_like(lower), where=active)
        sig = sigma[lo.index]
        kernel = 4*np.pi/H * sig*scale*np.log(ratio)
        gam0, gam1 = (field @ kernel.T for field in fields)
        mn = np.where((nu[None,:] < edge[:,None]) & active, (edge[:,None]+edges[1:][None,:])/2, nu[None,:])
        for s in range(ns):
            weights = lo.g.to_numpy()*np.exp(-lo.energy_eV.to_numpy()*EV/(KB*T[s]))
            denom = weights @ gam0[s]
            if denom <= 0:
                raise ValueError("nonpositive baseline rate")
            rows = []
            for k, lev in enumerate(lo.level_number):
                if z == 14 and int(lev) in (0,1,6,7,9,10,42,44,52,124):
                    inv = 4*np.pi/H*sig[k]/nu*width*np.exp(-H*mn[k]/(KB*T[s]))
                    spontaneous = 2*H*mn[k]**3/C**2
                    rec0, rec1 = (float(inv @ (spontaneous+field[s])) for field in fields)
                    rows.append(dict(level=int(lev), gamma0=float(gam0[s,k]), gamma1=float(gam1[s,k]),
                                     gamma_ratio=float(gam1[s,k]/gam0[s,k]), milne_ratio=rec1/rec0))
            records.append(dict(Z=z, shell=s, T=float(T[s]), included=len(lo), excluded=len(all_lower)-len(lo),
                                covered_LTE_weighted_gamma_ratio=float((weights @ gam1[s])/denom), selected=rows))
    inputs = [Path(__file__), model/"levels.csv", model/"ionization_energies.csv", sigma_path,
              cap/"manifest.json", cap/"Te.f64", cap/"nu.f64", cap/"dnu.f64"]
    inputs += [folder/(variant+suffix) for folder,variant in variants for suffix in (".json",".J.f64")]
    angular_refinement = None
    if base_fields != a.fields:
        coarser = np.fromfile(base_fields/"corrected.J.f64", dtype="<f8").reshape(ns,nf)
        rel = np.abs(coarser-fields[1])/(coarser+fields[1])
        angular_refinement = dict(max_symmetric_cell_difference=float(rel.max()),
                                  shell_integrated_fine_over_coarse=((fields[1] @ width)/(coarser @ width)).tolist())
        inputs.append(base_fields/"corrected.J.f64")
    print(json.dumps(dict(scope="fixed O material13; tabulated BF routes only; no NLTE population or emergent spectrum solve",
                          records=records, angular_refinement=angular_refinement,
                          sha256={str(p):sha(p) for p in inputs}), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
