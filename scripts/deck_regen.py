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
    p.add_argument("--compare-to", type=Path,
                   default=Path("data/tardis_reference_toy06_19p48d_sivcaiv_ftos"),
                   help="완전성 게이트 기준 덱 — 여기 있는 파일이 하나라도 없으면 실패")
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

    # expand+finalize 만으로는 덱이 완결되지 않는다.  기존 드라이버 5개가 전부
    # 여기서 멈춰 있었고(수동 후속 단계에 의존), 그 결과 충돌자료 34개 파일
    # (ige_col_*_cmfgen.bin)·level_multiplicity·ma_radrecomb 이 통째로 빠진
    # 덱이 만들어질 수 있었다.  전 단계를 여기서 돌린다.
    for label, cmd in [
        # --source-manifest 를 반드시 준다.  없으면 sources=None 이 되어 빌더가
        # 원본 vintage 를 **스스로 고른다**(build_cmfgen_coldata_all.py:693).
        # 그러면 osc 쪽에서 닫은 vintage 혼입이 충돌자료 경로로 재진입한다 —
        # 덱의 선/준위는 링크가 정한 vintage, 충돌강도는 빌더가 고른 vintage 가 된다.
        ("coldata", [sys.executable, str(ROOT / "scripts/build_cmfgen_coldata_all.py"),
                     "--ref-dir", str(new), "--write",
                     "--source-manifest", str(new / "atomic_vintage_manifest.csv")]),
        ("level_multiplicity", [sys.executable,
                                str(ROOT / "scripts/bake_level_multiplicity.py"),
                                "--ref-dir", str(new)]),
        ("ma_radrecomb", [sys.executable,
                          str(ROOT / "scripts/build_ma_radrecomb_target.py"), str(new)]),
    ]:
        print(f"=== {label}")
        rc = subprocess.call(cmd)
        if rc != 0:
            raise SystemExit(f"{label} failed rc={rc} — deck is INCOMPLETE: {new}")

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
    # ★stale 게이트 — 빌더는 성공한 이온만 쓰고 실패한 이온의 기존 파일은 지우지
    # 않는다.  같은 덱에 두 번 구우면 "성공분 + 이전 실패분" 혼합이 남고, 그것은
    # 어느 한쪽보다 나쁘다(잘못된 vintage 의 Omega 가 조용히 실린다).
    import csv as _csv
    man = new / "coldata_cmfgen_manifest.csv"
    if man.is_file():
        ok = {Path(r["out_bin"]).name for r in _csv.DictReader(man.open())
              if r.get("status") == "OK" and r.get("out_bin")}
        stale = sorted(p.name for p in new.glob("ige_col_*") if p.name not in ok)
        if stale:
            raise SystemExit(
                f"deck has {len(stale)} stale collision file(s) not in the manifest "
                f"OK set — wrong-vintage Omega would ship silently: {stale[:8]}")

    # ★완전성 게이트 — 이 결함 부류가 재발하지 않게 하는 구조적 장치.
    # 기준 덱에 있는 파일이 새 덱에 하나라도 없으면 덱을 완성으로 선언하지 않는다.
    ref_deck = (a.compare_to if a.compare_to.is_absolute() else ROOT / a.compare_to)
    ref_names = {p.name for p in ref_deck.iterdir()} if ref_deck.is_dir() else set()
    if not ref_names:
        raise SystemExit(f"completeness reference deck unreadable: {ref_deck}")
    have = {p.name for p in new.iterdir()}
    missing = sorted(n for n in ref_names - have if not n.startswith("."))
    if missing:
        raise SystemExit(
            f"deck INCOMPLETE — {len(missing)} artifact(s) absent vs reference: "
            + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))

    print(f"created deck: {new}  (finalized, provenance stamped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
