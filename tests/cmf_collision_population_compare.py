"""고정장 충돌 수정의 실제 인구 비교. 최종 CMFGEN 동질성 판정은 하지 않는다."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
result = {"root": str(root), "fixed_field_T_ne_NT": True,
          "SF_feedback": False, "charge_solved": False, "RTE_solved": False,
          "CMFGEN_equivalence": "NOT_JUDGED", "cases": [], "inputs": {}}
for iteration in (1, 2):
    case = {"iteration": iteration, "ions": [], "levels": {}}
    for name in ("ions", "levels"):
        path_old = root / f"old-{iteration}" / f"{name}.csv"
        path_new = root / f"new-{iteration}" / f"{name}.csv"
        for path in (path_old, path_new):
            result["inputs"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path_old.open() as f:
            old = list(csv.DictReader(f))
        with path_new.open() as f:
            new = list(csv.DictReader(f))
        assert old and len(old) == len(new)
        for a, b in zip(old, new):
            assert {k: v for k, v in a.items() if k != "after_cm3"} == {
                k: v for k, v in b.items() if k != "after_cm3"}
            before, x, y, total = map(float, (a["before_cm3"], a["after_cm3"], b["after_cm3"], a["nuclei_cm3"]))
            assert all(math.isfinite(v) and v >= 0 for v in (before, x, y)) and total > 0
            z, q = int(a["Z"]), int(a["charge"])
            if name == "ions":
                row = dict(Z=z, charge=q, before_cm3=before, old_cm3=x, new_cm3=y,
                           before_fraction=before/total, old_fraction=x/total, new_fraction=y/total)
                case["ions"].append(row)
                print(f'COLLISION_POP_ION iteration={iteration} Z={z} charge={q} '
                      f'old_fraction={x/total:.12g} new_fraction={y/total:.12g} old_cm3={x:.12g} new_cm3={y:.12g}')
            else:
                item = case["levels"].setdefault(str(z), dict(old_L1_per_nucleus=0.0,
                    change_L1_per_nucleus=0.0, max_change_fraction=0.0, max_change=None))
                item["old_L1_per_nucleus"] += abs(x-before)/total
                item["change_L1_per_nucleus"] += abs(x-y)/total
                change = abs(x-y)/total
                row = dict(Z=z, charge=q, global_level=int(a["global_level"]), local_level=int(a["local_level"]),
                           energy_eV=float(a["energy_eV"]), old_cm3=x, new_cm3=y, change_per_nucleus=change)
                if change > item["max_change_fraction"]:
                    item["max_change_fraction"], item["max_change"] = change, row
                if z == 27 and q == 4 and int(a["local_level"]) in (25, 28):
                    case.setdefault("CoV_target_levels", []).append(row)
    for z, item in case["levels"].items():
        print(f'COLLISION_POP_BASELINE iteration={iteration} Z={z} '
              f'old_L1_per_nucleus={item["old_L1_per_nucleus"]:.12g} '
              f'change_L1_per_nucleus={item["change_L1_per_nucleus"]:.12g}')
        # 기존 저장 인구의 재현 오차와 물리 수정 효과를 구별한다.
        assert item["old_L1_per_nucleus"] < 1e-6, (iteration, z, item)
    result["cases"].append(case)
out = Path("comparison.json")
with out.open("x") as f:
    json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    f.write("\n")
print("COLLISION_POP_COMPARISON_PASS old_snapshot_reproduced=1 final_equivalence=NOT_JUDGED")
