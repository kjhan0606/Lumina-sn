#!/usr/bin/env python3
"""C7 수리 — 봉인에 검증기를 붙인다.

계측·배선 부채 census C7 분류에서 최대 미결이 `seal_no_verifier` 11건이었다.
**검증기 없는 봉인은 봉인이 아니라 그냥 파일이다** — 아무도 그것이 봉인 이후
바뀌었는지 알 수 없다.

봉인의 목적은 "구현 전에 무엇이 바뀔지 미리 못박는 것"이다. 따라서 검증해야 할 것은
**봉인된 시점 이후 내용이 바뀌지 않았는가** 하나다.

두 갈래로 검증한다(a2_08 의 3중 대조 패턴을 일반화):
  A. 사이드카 있음 : 현재 sha256 == `<파일>.sha256` == **최초 커밋의 blob** (3중)
  B. 사이드카 없음 : 현재 sha256 == **최초 커밋의 blob**                    (2중)

B 도 봉인의 핵심 성질(사후 변조 탐지)을 보장한다 — 사이드카는 편의이지 근거가 아니다.
최초 커밋은 `git log --diff-filter=A` 로 찾는다(봉인 커밋이 따로 기록돼 있지 않아도 된다).

rc=0 전부 무결 · rc=2 하나라도 변조/불일치.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ROOT / "validation/instrumentation_debt/C7_ORPHAN_CLASSES.json"


def sh(cmd: list[str]) -> bytes:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True).stdout


def seal_paths() -> list[str]:
    """분류 산출물에서 봉인 부류를 읽는다(손으로 목록을 적지 않는다)."""
    if not CLASSES.is_file():
        raise SystemExit(f"missing {CLASSES} — scripts 로 먼저 분류를 생성하라")
    c = json.loads(CLASSES.read_text())["classes"]
    return sorted(set(c.get("seal_no_verifier", []) + c.get("seal_with_verifier", [])))


def main() -> int:
    rows, bad = [], 0
    for rel in seal_paths():
        p = ROOT / rel
        if not p.is_file():
            rows.append((rel, "MISSING", "-", "-")); bad += 1; continue
        cur = hashlib.sha256(p.read_bytes()).hexdigest()
        side_p = Path(str(p) + ".sha256")
        side = side_p.read_text().split()[0] if side_p.is_file() else None
        add = sh(["git", "log", "--diff-filter=A", "--format=%H", "-1", "--", rel]).decode().strip()
        sealed = None
        if add:
            blob = sh(["git", "show", f"{add}:{rel}"])
            if blob:
                sealed = hashlib.sha256(blob).hexdigest()
        mode = "3-way" if side else "2-way"
        # ★불변 봉인 vs 국면 기록을 가른다.
        #   allowlist  = 구현 전에 못박은 약속 → **절대 불변**이어야 한다
        #   manifest   = 국면 시작 기록 → 새 국면이 새 버전을 쓰는 것은 정당하다
        # 단 후자도 **교체 자체는 보고**한다. 2026-08-07 에 이 구분 없이 돌렸더니
        # implementation_start_manifest 가 걸렸는데, 실제로는 V1→V2 교체였고
        # 그 교체가 "A2-13 이 봉인 없이(BLOCKED_GIT_READ_ONLY) src 편집에 들어갔다"는
        # 기록을 트리에서 지운 것이었다 — 변조는 아니나 **소실**이므로 보고 대상이다.
        immutable = "allowlist" in Path(rel).name.lower()
        changed = sealed is not None and cur != sealed
        if immutable:
            ok = (sealed is not None and cur == sealed and (side is None or side == cur))
            status = "OK" if ok else "TAMPERED_SEAL"
        else:
            ok = sealed is not None and (side is None or side == cur)
            status = "OK" if (ok and not changed) else (
                "SUPERSEDED (교체 기록 확인 필요)" if changed and ok else "UNVERIFIABLE")
        if status.startswith("TAMPERED") or status == "UNVERIFIABLE":
            bad += 1
        rows.append((rel, status, mode,
                     f"cur={cur[:12]} sealed={(sealed or '-')[:12]}"
                     + (f" side={side[:12]}" if side else "")))

    for rel, st, mode, detail in rows:
        print(f"  {st:<26} {mode:<6} {Path(rel).name:<52} {detail}")
    print(f"\nSEAL_VERIFY total={len(rows)} ok={len(rows)-bad} bad={bad} "
          f"verdict={'PASS' if bad == 0 else 'FAIL'}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
