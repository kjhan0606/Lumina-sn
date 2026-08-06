#!/usr/bin/env python3
"""범용 덱 생성기 — **대조할 CMFGEN 런의 `atomic_links.txt` 에 덱을 맞춘다.**

운영 방침 (user 2026-08-06): *"그때 그때 CMFGEN 에 맞춰서 돌리죠."*
⟹ 덱은 고정 자산이 아니라 **대조 상대(CMFGEN 런)에 종속된 산출물**이다.
비교하려는 런의 links 파일을 그대로 먹여 vintage 를 동일하게 만든다.
그래야 게이트가 vintage 차이가 아니라 **코드 차이**를 잰다.

이 스크립트가 `deck_regen_{r4_ftos,fullcov,r1_vintage,r4_offcontrol,r5_vacuum}`
_driver.py 를 대체한다 — 그 5개는 OLD/NEW 경로만 다른 복제본이었다.

기존 드라이버들이 갖고 있던 결함 하나를 여기서 닫는다:
**expand 만 하고 finalize 를 호출하지 않아** 미완성 덱(f_ul 부호 미적용,
B_lu/B_ul 미재계산)이 나올 수 있었다.  여기서는 두 단계를 항상 함께 돌린다.

사용:
  python3 scripts/deck_regen.py \
      --links /gpfs/kjhan/cmfgen_runs/<run>/atomic_links.txt \
      --out   data/tardis_reference_toy06_19p48d_<tag>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPAND = ROOT / "scripts/expand_atomic_data_cmfgen.py"
FINALIZE = ROOT / "scripts/finalize_cmfgen_ref_npy.py"
DEFAULT_BASE = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_links"

REBUILT = {
    "atom_masses.csv", "atomic_data_cmfgen.h5", "atomic_vintage_manifest.csv",
    "cmfgen_sigma_bf.bin", "coldata_cmfgen_manifest.csv", "ionization_energies.csv",
    "level_multiplicity.csv", "levels.csv", "line2macro_level_upper.npy",
    "line_list.csv", "kshape_contract.txt", "ma_radrecomb_target.bin",
    "ma_radrecomb_target_manifest.csv", "macro_atom_data.csv",
    "macro_atom_references.csv", "transition_probabilities.npy",
    "tau_sobolev.npy", "verification.log", "zeta_data.npy", "zeta_ions.csv",
    "zeta_temps.csv",
}


def validate_composition_shape(deck: Path) -> None:
    with (deck / "abundances.csv").open(newline="") as stream:
        header = next(csv.reader(stream), [])
    cols = max(len(header) - 1, 0)
    with (deck / "geometry.csv").open(newline="") as stream:
        rows = csv.reader(stream)
        next(rows, None)
        n = sum(1 for row in rows if row)
    if cols != n:
        raise SystemExit(f"abundances/geometry shape mismatch in {deck}: {cols} vs {n}")
    return n


def copy_companions(base: Path, new: Path) -> None:
    for source in sorted(base.iterdir()):
        if source.name in REBUILT or source.name.startswith("ige_col_"):
            continue
        target = new / source.name
        if target.exists():
            continue
        resolved = source.resolve(strict=True)
        if resolved.is_file():
            shutil.copy2(resolved, target)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--links", type=Path, required=True,
                   help="대조할 CMFGEN 런의 atomic_links.txt (덱 vintage 를 정한다)")
    p.add_argument("--out", type=Path, required=True, help="새 덱 디렉터리 (기존 덱 덮어쓰기 거부)")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE,
                   help="companion 파일을 가져올 provenance 덱")
    p.add_argument("--n-shells", type=int, default=0,
                   help="0 이면 geometry.csv 행 수에서 자동 결정")
    a = p.parse_args()

    links = a.links.expanduser().resolve()
    new = (a.out if a.out.is_absolute() else ROOT / a.out).resolve()
    base = (a.base if a.base.is_absolute() else ROOT / a.base).resolve()
    if not links.is_file():
        raise SystemExit(f"missing links file: {links}")
    if new.exists():
        raise SystemExit(f"refusing to overwrite existing deck: {new}")
    if not base.is_dir():
        raise SystemExit(f"missing provenance deck: {base}")

    env = {"CMFGEN_FULL_LEVELS": "1", "CMFGEN_SUPER_LEVELS": "1",
           "CMFGEN_LINK_FTOS": "1", "CMFGEN_LINKS": str(links)}
    os.environ.update(env)

    spec = importlib.util.spec_from_file_location("deck_expand", EXPAND)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not module.LINK_FTOS_ENABLED:
        raise SystemExit("expand module did not arm CMFGEN_LINK_FTOS")
    if not hasattr(module, "A_PREFACTOR"):
        raise SystemExit("expand module lacks A_PREFACTOR — I20 repair not present")
    module.ROOT = ROOT
    module.CMFGEN_ROOT = ROOT / "data/atomic/cmfgen"
    module.OUT_DIR = new
    module.OUT_H5 = new / "atomic_data_cmfgen.h5"
    module.OUT_SIGMA_BIN = new / "cmfgen_sigma_bf.bin"
    print(f"=== expand: links={links}")
    module.main()
    copy_companions(base, new)
    n_shells = a.n_shells or validate_composition_shape(new)

    # 기존 드라이버가 빠뜨렸던 단계 — 이것 없이는 f_ul 부호와 B_lu/B_ul 이 미완이다.
    print(f"=== finalize: n_shells={n_shells}")
    rc = subprocess.call([sys.executable, str(FINALIZE), str(new), str(n_shells)])
    if rc != 0:
        raise SystemExit(f"finalize failed rc={rc} — deck is INCOMPLETE: {new}")

    stamp = {
        "schema": "lumina-deck-provenance-v1",
        "deck": str(new),
        "cmfgen_links": str(links),
        "cmfgen_links_sha256": hashlib.sha256(links.read_bytes()).hexdigest(),
        "provenance_base": str(base),
        "env": env,
        "n_shells": n_shells,
        "finalized": True,
        # 런이 정본 원자 트리 대신 로컬 수리본을 쓴 지점 — 공시 대상이다.
        "atomic_local_overrides": [
            dict(r, sha256=hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest())
            for r in getattr(module, "ATOMIC_LOCAL_OVERRIDES", [])
        ],
        "note": "덱은 대조 CMFGEN 런에 종속된다 — links 가 vintage 를 정한다",
    }
    (new / "DECK_PROVENANCE.json").write_text(json.dumps(stamp, indent=1,
                                                         ensure_ascii=False))
    print(f"created deck: {new}  (finalized, provenance stamped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
