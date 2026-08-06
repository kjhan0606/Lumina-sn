#!/usr/bin/env python3
"""C5 수리 — 덱 계보를 **기재**한다 (재생성이 아니라).

계측·배선 부채 census C5: 덱 107개 중 계보(atomic_vintage_manifest)를 가진 것이 6개,
provenance stamp 를 가진 것이 3개다. live 덱(스크립트 참조 또는 회귀 대장 등재) 33개로
좁혀도 계보 5 / stamp 2 다.

**107개를 재생성할 수는 없다.** 대신 캠페인 처분 원칙(발견의 처분 = 조용한 대장 기재)을
적용한다: **침묵을 기재로 바꾼다.**

  침묵  = DECK_PROVENANCE.json 이 없다 → 읽는 사람은 "문제 없다"로 읽는다
  기재  = stamp 는 있는데 필드가 UNKNOWN → 읽는 사람은 "계보 미상"으로 읽는다

두 상태는 정보량이 다르다. 후자만이 부채로 보인다.

각 덱에서 유도 가능한 것은 유도하고, 유도 불가한 것은 **UNKNOWN 으로 명시**한다.
추정하지 않는다 — 모르는 것은 모른다고 적는다.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# 계보가 유도 가능한 신호들
SIGNALS = [
    ("atomic_vintage_manifest.csv", "per-ion osc/phot/col 원본 경로"),
    ("DECK_PROVENANCE.json", "deck_regen.py stamp (links + sha256)"),
    ("coldata_cmfgen_manifest.csv", "충돌자료 원본"),
    ("kshape_contract.txt", "line epoch + npy 해시 결박"),
    ("config.json", "모델 설정"),
]


def sh(cmd: str) -> str:
    return subprocess.run(["bash", "-c", cmd], cwd=ROOT,
                          capture_output=True, text=True).stdout


def live_map() -> tuple[dict, set]:
    refs = sh("grep -rho 'data/tardis_reference[A-Za-z0-9_.]*' "
              "scripts/*.sh scripts/*.slurm scripts/*.py 2>/dev/null")
    cnt: dict[str, int] = {}
    for tok in refs.split():
        name = tok.split("/", 1)[1]
        cnt[name] = cnt.get(name, 0) + 1
    led = set(sh("grep -o 'tardis_reference[A-Za-z0-9_.]*' "
                 "validation/regression_ledger/ledger.jsonl 2>/dev/null").split())
    return cnt, led


def audit_deck(d: Path, refs: int, in_ledger: bool) -> dict:
    present = {f: (d / f).exists() for f, _ in SIGNALS}
    stamp = {
        "schema": "lumina-deck-provenance-audit-v1",
        "deck": d.name,
        "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "liveness": {"script_refs": refs, "in_regression_ledger": in_ledger,
                     "classification": "LIVE" if (refs or in_ledger) else "UNREFERENCED"},
        "signals_present": present,
        "shape": {},
        "atomic_provenance": "UNKNOWN",
        "builder": "UNKNOWN",
        "known_defects": [],
    }
    # 유도 가능한 것만 유도한다
    ll = d / "line_list.csv"
    lv = d / "levels.csv"
    if ll.exists():
        stamp["shape"]["n_lines"] = max(0, sum(1 for _ in ll.open()) - 1)
    if lv.exists():
        stamp["shape"]["n_levels"] = max(0, sum(1 for _ in lv.open()) - 1)
    geo = d / "geometry.csv"
    if geo.exists():
        stamp["shape"]["n_shells_geometry"] = max(0, sum(1 for _ in geo.open()) - 1)
    ks = d / "kshape_contract.txt"
    if ks.exists():
        for line in ks.read_text().splitlines():
            if line.startswith("n_shells="):
                stamp["shape"]["n_shells_contract"] = int(line.split("=")[1])
    if present["atomic_vintage_manifest.csv"]:
        stamp["atomic_provenance"] = "RECORDED (atomic_vintage_manifest.csv)"
    if present["DECK_PROVENANCE.json"]:
        try:
            j = json.loads((d / "DECK_PROVENANCE.json").read_text())
            stamp["builder"] = f"deck_regen.py links={j.get('cmfgen_links')}"
            stamp["atomic_provenance"] = "RECORDED (links + sha256)"
        except Exception:
            pass

    # 알려진 결함을 stamp 에 박는다 — 침묵을 기재로 바꾸는 핵심
    s = stamp["shape"]
    if "n_shells_geometry" in s and "n_shells_contract" in s \
            and s["n_shells_geometry"] != s["n_shells_contract"]:
        stamp["known_defects"].append(
            f"0-K 형상 불일치: geometry={s['n_shells_geometry']} 셸인데 "
            f"kshape 계약 n_shells={s['n_shells_contract']} — 현 로더는 이 덱을 거부한다")
    if stamp["atomic_provenance"] == "UNKNOWN":
        stamp["known_defects"].append(
            "원자자료 계보 미상 — 어느 vintage 에서 왔는지 산출물이 말하지 못한다. "
            "user 08-03 동일성 교리 아래에서 대조군 자격의 전제를 충족하지 못한다")
    return stamp


def main() -> int:
    cnt, led = live_map()
    decks = sorted(p for p in DATA.iterdir()
                   if p.is_dir() and p.name.startswith("tardis_reference"))
    summary = {"schema": "lumina-c5-deck-provenance-audit-summary-v1",
               "git_head": sh("git rev-parse --short HEAD").strip(),
               "decks_total": len(decks), "live": 0, "stamped": 0, "by_deck": {}}
    for d in decks:
        refs = cnt.get(d.name, 0)
        live = bool(refs or d.name in led)
        if not live:
            continue                      # UNREFERENCED 는 건드리지 않는다
        summary["live"] += 1
        st = audit_deck(d, refs, d.name in led)
        # 기존 DECK_PROVENANCE.json(= deck_regen.py 산출)은 덮어쓰지 않는다
        target = d / ("DECK_PROVENANCE.json" if not (d / "DECK_PROVENANCE.json").exists()
                      else "DECK_PROVENANCE_AUDIT.json")
        target.write_text(json.dumps(st, indent=1, ensure_ascii=False))
        summary["stamped"] += 1
        summary["by_deck"][d.name] = {
            "refs": refs, "atomic_provenance": st["atomic_provenance"],
            "known_defects": len(st["known_defects"]), "stamp": target.name}
    out = ROOT / "validation/instrumentation_debt/C5_DECK_PROVENANCE_AUDIT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    unknown = sum(1 for v in summary["by_deck"].values()
                  if v["atomic_provenance"] == "UNKNOWN")
    print(f"live={summary['live']} stamped={summary['stamped']} "
          f"atomic_provenance UNKNOWN={unknown}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
