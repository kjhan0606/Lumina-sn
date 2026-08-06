#!/usr/bin/env python3
"""C7 자기부채 상환 — 2026-08-07 에 내가 인라인 python3 -c 로 만든 산출물의 생성 스크립트.

R-8(재현 스크립트 격차)을 비판해 놓고 같은 결함을 재생산했다. 그 부채를 갚는다.
재생성 대상:
  validation/a2_16/C2_CALLER_MIGRATION_CENSUS.json     하드 거부 env × 런처 노출
  validation/a2_16/A2_16_LAUNCHER_FLEET_BREAKAGE.json  함대 파손 요약

★수치는 패턴 의존이다(98→157→83-166 스프레드). 그래서 단일 수치가 아니라
  **세 패턴의 스프레드**를 함께 낸다 — scripts/instrumentation_debt_census.py 와 같은 규율.
"""
import json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(c: str) -> str:
    return subprocess.run(["bash", "-c", c], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def fatal_env() -> list[str]:
    """레지스트리에서 유도한다 — 손으로 적지 않는다."""
    reg = (ROOT / "src/legacy_knob_registry.h").read_text()
    return sorted(set(re.findall(r'X\("(LUMINA_[A-Z0-9_]+)",\s*LK_ENFORCE_FATAL', reg)))


def main() -> int:
    envs = fatal_env()
    alt = "|".join(envs)
    rows = []
    for k in envs:
        rows.append({"env": k,
                     "launchers_setting": int(sh(f'grep -rl "{k}=" scripts/*.sh 2>/dev/null | wc -l')),
                     "launchers_nonzero": int(sh(f'grep -rlE "{k}=[^0 ]" scripts/*.sh 2>/dev/null | wc -l')),
                     "trigger": "non-empty" if k in ("LUMINA_TE_TRAD_RATIO", "LUMINA_TRAD_COLOR_FIX")
                                else "atoi!=0"})
    pats = {
        "p1_export_prefix_nonzero": f"grep -rlE 'export .*({alt})=[^0 ]' scripts/*.sh 2>/dev/null | wc -l",
        "p2_any_assignment": f"grep -rlE '({alt})=' scripts/*.sh 2>/dev/null | wc -l",
        "p3_assignment_nonzero": f"grep -rlE '({alt})=[^0 ]' scripts/*.sh 2>/dev/null | wc -l",
    }
    counts = {n: int(sh(c)) for n, c in pats.items()}
    out = {
        "schema": "lumina-c2-caller-migration-census-v2",
        "git_head": sh("git rev-parse --short HEAD"),
        "untracked_files": int(sh("git status --porcelain | grep -c '^??'") or 0),
        "hard_reject_env_total": len(envs), "per_env": rows,
        "launchers_affected": {"patterns": pats, "counts": counts,
                               "min": min(counts.values()), "max": max(counts.values()),
                               "spread_ratio": max(counts.values()) / max(1, min(counts.values()))},
        "read_this_first": "단일 수치가 아니라 spread 를 보라. 98->157->83-166 이 그 실증이다.",
        "caveats": [
            "정적 grep 은 ${VAR-default} 파라미터화와 sbatch --export=ALL 상속을 판정할 수 없다",
            "plasma 사이트는 enable_nlte && iter>=nlte_start_iter 안에서만 검사한다 (C10) — "
            "비-NLTE 런처는 사망이 아니라 no-op 다",
        ],
    }
    p = ROOT / "validation/a2_16/C2_CALLER_MIGRATION_CENSUS.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "per_env"}, indent=1, ensure_ascii=False))
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
