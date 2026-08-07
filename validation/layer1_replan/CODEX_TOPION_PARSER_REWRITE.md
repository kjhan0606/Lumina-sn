```python
#!/usr/bin/env python3
"""최상단 이온 15종의 전 준위와 바닥준위를 추출한다.

CMFGEN osc 파일은 이 파일에서 다시 해석하지 않는다. 덱 빌드의 정본 경로인
``expand_atomic_data_cmfgen.parse_all_ions()``를 그대로 호출하고, 그 함수가 반환한
``osc.n_levels`` 및 ``osc.levels``를 검증한다.

대상 최상단 이온은 atomic_links.txt에 존재하지 않는다. parse_all_ions()는 링크 전용
함수가 아니며, ``CMFGEN_LINK_MAP``에 이온이 없으면 ``_pick_latest()``와
``parse_osc()``를 사용하는 자동 선택 경로로 내려간다. 여기서는 링크 맵을 명시적으로
빈 값으로 두고 대상 이온만 ``ION_LEVEL_CAPS``에 cap=None으로 주입한다. 따라서
atomic_links.txt를 위조하거나 최상단 링크를 새로 만들지 않는다.

Cloudy/Stout .nrg는 CMFGEN osc가 아니므로 Stout의 공개 파일 규약대로 직접 읽는다.
.nrg 첫 줄은 준위 수가 아니라 형식 버전이고, 별표 행이 자료 블록의 끝이다. 각 자료
행의 첫 열은 1부터 시작하는 준위 인덱스이므로 이를 연속으로 검사하고, 마지막 인덱스와
실제로 읽은 레코드 수가 같은지 확인한다.

모든 검사가 끝나기 전에는 산출 파일을 만들거나 덮어쓰지 않는다.
"""

from __future__ import annotations

import csv
import importlib
import io
import math
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

CMFGEN_ROOT = Path("/gpfs/kjhan/cmfgen_21jun23/atomic")
STOUT_ROOT = Path("/gpfs/kjhan/cloudy-master/data/stout")

GROUND_OUTPUT = ROOT / "data/atomic/topion_ground_levels.csv"
LEVEL_OUTPUT = ROOT / "data/atomic/topion_levels.csv"

GROUND_COLUMNS = [
    "Z",
    "ion_stage_0based",
    "label",
    "E0_cm-1",
    "g0",
    "provenance",
    "source_file",
]
LEVEL_COLUMNS = [
    "Z",
    "ion_stage_0based",
    "label",
    "level_index",
    "E_cm-1",
    "g",
    "provenance",
]

# E(cm^-1)를 Boltzmann 지수로 바꾸는 hc/k 값 [cm K].
HC_OVER_K_CM_K = 1.4387768775039338

# 전 온도에 대한 하한은 아래의 대수적 검사로 증명한다. 이 온도들은 계산 구현도
# 실제로 같은 하한을 내는지 확인하기 위한 수치 대조점이다.
PARTITION_CHECK_TEMPERATURES_K = (
    1.0e-3,
    1.0,
    10.0,
    100.0,
    1.0e3,
    3.0e3,
    1.0e4,
    3.0e4,
    1.0e5,
    1.0e6,
    1.0e9,
    1.0e12,
)

STOUT_MAGIC = (17, 9, 5)


class ValidationError(RuntimeError):
    """자료 계약을 어겼을 때 산출을 중단시키는 오류."""


@dataclass(frozen=True)
class Target:
    z: int
    stage0: int
    label: str
    cmf_dir: Path | None = None
    stout_file: Path | None = None

    @property
    def cmf_stage1(self) -> int:
        """expand_atomic_data_cmfgen.py가 사용하는 1-based 이온 단계."""
        return self.stage0 + 1

    @property
    def electron_count(self) -> int:
        return self.z - self.stage0


@dataclass(frozen=True)
class Level:
    index: int
    energy_cm: float
    g: float


@dataclass(frozen=True)
class ParsedIon:
    target: Target
    levels: tuple[Level, ...]
    provenance: str
    source_file: Path


TARGETS = (
    Target(6, 3, "C IV", cmf_dir=CMFGEN_ROOT / "CARB/IV"),
    Target(8, 3, "O IV", cmf_dir=CMFGEN_ROOT / "OXY/IV"),
    Target(12, 3, "Mg IV", cmf_dir=CMFGEN_ROOT / "MG/IV"),
    Target(13, 4, "Al V", cmf_dir=CMFGEN_ROOT / "AL/V"),
    Target(14, 5, "Si VI", cmf_dir=CMFGEN_ROOT / "SIL/VI"),
    Target(16, 5, "S VI", cmf_dir=CMFGEN_ROOT / "SUL/VI"),
    Target(20, 5, "Ca VI", cmf_dir=CMFGEN_ROOT / "CA/VI"),
    Target(
        21,
        3,
        "Sc IV",
        stout_file=STOUT_ROOT / "sc/sc_4/sc_4.nrg",
    ),
    Target(
        22,
        4,
        "Ti V",
        stout_file=STOUT_ROOT / "ti/ti_5/ti_5.nrg",
    ),
    Target(
        23,
        1,
        "V II",
        stout_file=STOUT_ROOT / "v/v_2/v_2.nrg",
    ),
    Target(24, 4, "Cr V", cmf_dir=CMFGEN_ROOT / "CHRO/V"),
    Target(25, 3, "Mn IV", cmf_dir=CMFGEN_ROOT / "MAN/IV"),
    Target(26, 6, "Fe VII", cmf_dir=CMFGEN_ROOT / "FE/VII"),
    Target(27, 6, "Co VII", cmf_dir=CMFGEN_ROOT / "COB/VII"),
    Target(28, 6, "Ni VII", cmf_dir=CMFGEN_ROOT / "NICK/VII"),
)


def load_deck_parser() -> Any:
    """덱 빌드가 사용하는 CMFGEN 판독 모듈을 가져온다."""
    script_dir = str(SCRIPT_DIR)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        return importlib.import_module("expand_atomic_data_cmfgen")
    except Exception as exc:
        raise ValidationError(
            "expand_atomic_data_cmfgen.py를 불러오지 못했다. "
            "정본 CMFGEN 판독기를 사용할 수 없으므로 중단한다."
        ) from exc


def path_is_inside(path: Path, parent: Path) -> bool:
    """심볼릭 링크를 해소한 실제 경로가 지정한 이온 디렉터리 안인지 확인한다."""
    try:
        path.resolve(strict=True).relative_to(parent.resolve(strict=True))
    except (FileNotFoundError, ValueError):
        return False
    return True


def call_canonical_cmfgen_parser(
    parser: Any,
    targets: tuple[Target, ...],
) -> dict[tuple[int, int], dict[str, Any]]:
    """parse_all_ions()의 자동 선택 경로로 CMFGEN 최상단 이온만 읽는다."""
    requested = {
        (target.z, target.cmf_stage1): None
        for target in targets
    }

    # 이 값들은 환경변수 노브가 아니다. 정본 함수의 입력 전역을 호출 동안만 한정해
    # atomic_links.txt에 없는 최상단 이온을 자동 선택 경로로 보내는 호출 어댑터다.
    overrides = {
        "CMFGEN_ROOT": CMFGEN_ROOT,
        "ION_LEVEL_CAPS": requested,
        "CMFGEN_LINKS_PATH": None,
        "CMFGEN_LINK_MAP": {},
        "VINTAGE_MATCH": False,
        "VINTAGE_PHOT_ONLY": False,
        "SUPER_LEVEL_ENABLED": False,
        "LINK_FTOS_ENABLED": False,
    }

    missing_globals = [
        name for name in overrides
        if not hasattr(parser, name)
    ]
    if missing_globals:
        raise ValidationError(
            "정본 판독기의 호출 계약이 바뀌었다: "
            + ", ".join(sorted(missing_globals))
        )

    saved = {
        name: getattr(parser, name)
        for name in overrides
    }
    try:
        for name, value in overrides.items():
            setattr(parser, name, value)
        parsed = parser.parse_all_ions()
    except Exception as exc:
        raise ValidationError(
            "expand_atomic_data_cmfgen.parse_all_ions()가 실패했다."
        ) from exc
    finally:
        for name, value in saved.items():
            setattr(parser, name, value)

    if not isinstance(parsed, dict):
        raise ValidationError(
            "parse_all_ions()가 dict를 반환하지 않았다."
        )

    expected_keys = set(requested)
    actual_keys = set(parsed)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValidationError(
            "CMFGEN 대상 집합 불일치: "
            f"누락={missing}, 예상 밖={extra}"
        )

    return parsed


def parse_cmfgen_targets(targets: tuple[Target, ...]) -> dict[str, ParsedIon]:
    """정본 parse_all_ions() 결과에서 E(cm^-1)와 g 전 준위를 꺼낸다."""
    parser = load_deck_parser()
    parsed = call_canonical_cmfgen_parser(parser, targets)
    result: dict[str, ParsedIon] = {}

    for target in targets:
        key = (target.z, target.cmf_stage1)
        data = parsed[key]

        try:
            osc = data["osc"]
            selected_levels = data["levels"]
            provenance = data["provenance"]
            source_file = Path(provenance["osc_path"])
            declared_count = int(osc.n_levels)
            parser_level_count = len(osc.levels)
            selected_count = len(selected_levels)
            kept_count = int(data["n_kept"])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValidationError(
                f"{target.label}: parse_all_ions() 반환 형식이 예상과 다르다."
            ) from exc

        if target.cmf_dir is None:
            raise ValidationError(f"{target.label}: CMFGEN 경로가 지정되지 않았다.")
        if not source_file.is_file():
            raise ValidationError(
                f"{target.label}: 선택된 osc 파일이 없다: {source_file}"
            )
        if not path_is_inside(source_file, target.cmf_dir):
            raise ValidationError(
                f"{target.label}: 다른 이온 경로의 osc 파일을 선택했다: "
                f"{source_file} (기대 경로 {target.cmf_dir})"
            )

        # osc.n_levels는 정본 parse_osc가 '!Number of energy levels'에서 읽은
        # 선언 수다. cap=None이므로 parser 배열, parse_all_ions 선택 배열 및
        # n_kept가 모두 이 선언 수와 정확히 같아야 한다.
        if declared_count <= 0:
            raise ValidationError(
                f"{target.label}: 선언 준위 수가 양수가 아니다: {declared_count}"
            )
        if parser_level_count != declared_count:
            raise ValidationError(
                f"{target.label}: 정본 osc 판독 수 {parser_level_count} != "
                f"파일 선언 수 {declared_count}: {source_file}"
            )
        if selected_count != declared_count or kept_count != declared_count:
            raise ValidationError(
                f"{target.label}: parse_all_ions()가 준위를 잘랐다: "
                f"선택={selected_count}, n_kept={kept_count}, "
                f"선언={declared_count}"
            )

        levels: list[Level] = []
        try:
            for index, row in enumerate(selected_levels, 1):
                levels.append(
                    Level(
                        index=index,
                        energy_cm=float(row["E_cm"]),
                        g=float(row["g"]),
                    )
                )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValidationError(
                f"{target.label}: 정본 판독 결과에서 E_cm/g를 읽지 못했다."
            ) from exc

        if len(levels) != declared_count:
            raise ValidationError(
                f"{target.label}: 변환 준위 수 {len(levels)} != "
                f"파일 선언 수 {declared_count}"
            )

        result[target.label] = ParsedIon(
            target=target,
            levels=tuple(levels),
            provenance="CMFGEN_21jun23",
            source_file=source_file,
        )

    return result


def parse_stout_target(target: Target) -> ParsedIon:
    """Cloudy가 읽는 것과 같은 .nrg 자료 블록을 엄격하게 읽는다."""
    path = target.stout_file
    if path is None:
        raise ValidationError(f"{target.label}: Stout 경로가 지정되지 않았다.")
    if not path.is_file():
        raise ValidationError(f"{target.label}: Stout 파일이 없다: {path}")

    try:
        lines = path.read_text(encoding="latin-1").splitlines()
    except OSError as exc:
        raise ValidationError(f"{target.label}: Stout 파일 읽기 실패: {path}") from exc

    if not lines:
        raise ValidationError(f"{target.label}: 빈 Stout 파일: {path}")

    # StoutFormat 및 Cloudy atmdat_STOUT_readin()에 따르면 첫 줄은 준위 수가
    # 아니라 형식 버전이다. 대상 세 파일은 Cloudy가 검사하는 17 09 05 형식이다.
    magic_fields = lines[0].split()
    try:
        magic = tuple(int(field) for field in magic_fields)
    except ValueError as exc:
        raise ValidationError(
            f"{target.label}: 잘못된 Stout 형식 버전 행: {lines[0]!r}"
        ) from exc
    if magic != STOUT_MAGIC:
        raise ValidationError(
            f"{target.label}: Stout 형식 버전 {magic}, 기대값 {STOUT_MAGIC}"
        )

    levels: list[Level] = []
    sentinel_reached = False

    for lineno, raw_line in enumerate(lines[1:], 2):
        if raw_line.startswith("*"):
            if not raw_line or set(raw_line) != {"*"}:
                raise ValidationError(
                    f"{path}:{lineno}: 잘못된 자료 종료 표식"
                )
            sentinel_reached = True
            break

        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        fields = stripped.split(maxsplit=3)
        if len(fields) < 3:
            raise ValidationError(
                f"{path}:{lineno}: .nrg 준위 열이 3개보다 적다."
            )

        try:
            source_index = int(fields[0])
            energy_cm = float(fields[1])
            g = float(fields[2])
        except ValueError as exc:
            raise ValidationError(
                f"{path}:{lineno}: .nrg 준위 레코드를 해석할 수 없다."
            ) from exc

        expected_index = len(levels) + 1
        if source_index != expected_index:
            raise ValidationError(
                f"{path}:{lineno}: 준위 ID={source_index}, "
                f"기대 ID={expected_index}"
            )

        levels.append(
            Level(index=source_index, energy_cm=energy_cm, g=g)
        )

    if not sentinel_reached:
        raise ValidationError(
            f"{target.label}: Stout 자료 종료 별표 행이 없다: {path}"
        )
    if not levels:
        raise ValidationError(f"{target.label}: Stout 준위가 없다: {path}")

    # .nrg에는 별도의 N 헤더가 없다. 대신 각 행이 준위 ID를 선언하며 Cloudy도
    # 자료 블록 크기를 HighestIndexInFile로 사용한다. ID를 1..N으로 강제했으므로
    # 마지막 선언 ID와 실제 파싱 레코드 수를 비교하는 것이 이 형식의 수량 검사다.
    declared_count = levels[-1].index
    if len(levels) != declared_count:
        raise ValidationError(
            f"{target.label}: Stout 파싱 수 {len(levels)} != "
            f"마지막 선언 ID {declared_count}"
        )

    return ParsedIon(
        target=target,
        levels=tuple(levels),
        provenance="CLOUDY_STOUT_NIST",
        source_file=path,
    )


def validate_partition_lower_bound(ion: ParsedIon) -> None:
    """모든 물리 온도 T>0에서 Z(T)>=g0임을 검사한다."""
    levels = ion.levels
    ground = levels[0]
    e0 = ground.energy_cm
    g0 = ground.g

    # 모든 E-E0가 유한한 비음수이고 모든 g가 양수이면 T>0에서 각 여기항은
    # 비음수다. 바닥항은 g0*exp(0)=g0이므로 이 조건은 모든 T에 대해
    # Z(T)=sum g*exp(-(E-E0)hc/kT) >= g0임을 직접 증명한다.
    for level in levels:
        delta_e = level.energy_cm - e0
        if not math.isfinite(delta_e) or delta_e < 0.0:
            raise ValidationError(
                f"{ion.target.label}: 바닥보다 낮거나 유한하지 않은 준위 "
                f"{level.index}: E-E0={delta_e}"
            )

    # 대수적 조건과 별도로 실제 계산 경로도 저온부터 고온까지 대조한다.
    # 지수 언더플로는 0이라는 올바른 극한이며, 어떤 floor/clamp도 적용하지 않는다.
    for temperature in PARTITION_CHECK_TEMPERATURES_K:
        terms = [
            level.g
            * math.exp(
                -(level.energy_cm - e0)
                * HC_OVER_K_CM_K
                / temperature
            )
            for level in levels
        ]
        partition = math.fsum(terms)
        if not math.isfinite(partition) or partition < g0:
            raise ValidationError(
                f"{ion.target.label}: T={temperature:g} K에서 "
                f"Z={partition!r} < g0={g0}"
            )


def validate_ion(ion: ParsedIon) -> None:
    """한 이온의 전 준위, 바닥준위 및 분배함수 계약을 검사한다."""
    if not ion.levels:
        raise ValidationError(f"{ion.target.label}: 준위가 없다.")

    for expected_index, level in enumerate(ion.levels, 1):
        if level.index != expected_index:
            raise ValidationError(
                f"{ion.target.label}: level_index 불연속: "
                f"{level.index} != {expected_index}"
            )
        if not math.isfinite(level.energy_cm):
            raise ValidationError(
                f"{ion.target.label}: 준위 {level.index}의 E가 유한하지 않다: "
                f"{level.energy_cm}"
            )
        if (
            not math.isfinite(level.g)
            or level.g <= 0.0
            or not level.g.is_integer()
        ):
            raise ValidationError(
                f"{ion.target.label}: 준위 {level.index}의 g가 "
                f"양의 정수가 아니다: {level.g}"
            )

    ground = ion.levels[0]
    if ground.energy_cm != 0.0:
        raise ValidationError(
            f"{ion.target.label}: 바닥준위 E0={ground.energy_cm}, 기대값 0"
        )
    if ground.g < 1.0:
        raise ValidationError(
            f"{ion.target.label}: 바닥준위 g0={ground.g}, 기대값 >=1"
        )

    validate_partition_lower_bound(ion)


def validate_isoelectronic_ground_weights(
    ions: tuple[ParsedIon, ...],
) -> None:
    """같은 전자 수를 가진 대상 이온들의 g0가 같은지 검사한다."""
    groups: dict[int, list[ParsedIon]] = defaultdict(list)
    for ion in ions:
        groups[ion.target.electron_count].append(ion)

    for electron_count, group in sorted(groups.items()):
        if len(group) < 2:
            continue

        weights = {ion.levels[0].g for ion in group}
        if len(weights) != 1:
            details = ", ".join(
                f"{ion.target.label}:g0={ion.levels[0].g:g}"
                for ion in group
            )
            raise ValidationError(
                f"{electron_count}전자 등전자 g0 불일치: {details}"
            )


def collect_and_validate() -> tuple[ParsedIon, ...]:
    """15개 이온을 모두 읽고, 어떤 파일도 쓰기 전에 전 검사를 끝낸다."""
    if len(TARGETS) != 15:
        raise ValidationError(f"대상 이온 수가 15가 아니다: {len(TARGETS)}")

    labels = [target.label for target in TARGETS]
    if len(set(labels)) != len(labels):
        raise ValidationError("대상 이온 표기에 중복이 있다.")

    for target in TARGETS:
        source_count = int(target.cmf_dir is not None) + int(
            target.stout_file is not None
        )
        if source_count != 1:
            raise ValidationError(
                f"{target.label}: CMFGEN/Stout 출처 중 정확히 하나가 필요하다."
            )

    cmf_targets = tuple(
        target for target in TARGETS
        if target.cmf_dir is not None
    )
    cmf_ions = parse_cmfgen_targets(cmf_targets)

    ions: list[ParsedIon] = []
    for target in TARGETS:
        if target.cmf_dir is not None:
            ion = cmf_ions[target.label]
        else:
            ion = parse_stout_target(target)
        validate_ion(ion)
        ions.append(ion)

    result = tuple(ions)
    if len(result) != 15:
        raise ValidationError(f"파싱된 이온 수가 15가 아니다: {len(result)}")

    validate_isoelectronic_ground_weights(result)
    return result


def make_csv_text(
    columns: list[str],
    rows: list[dict[str, object]],
) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=columns,
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def build_outputs(ions: tuple[ParsedIon, ...]) -> dict[Path, str]:
    """기존 CSV 경로와 컬럼 순서를 그대로 유지한다."""
    ground_rows: list[dict[str, object]] = []
    level_rows: list[dict[str, object]] = []

    for ion in ions:
        target = ion.target
        ground = ion.levels[0]

        ground_rows.append(
            {
                "Z": target.z,
                "ion_stage_0based": target.stage0,
                "label": target.label,
                "E0_cm-1": f"{ground.energy_cm:.6f}",
                "g0": f"{ground.g:.1f}",
                "provenance": ion.provenance,
                "source_file": str(ion.source_file),
            }
        )

        for level in ion.levels:
            level_rows.append(
                {
                    "Z": target.z,
                    "ion_stage_0based": target.stage0,
                    "label": target.label,
                    "level_index": level.index,
                    "E_cm-1": f"{level.energy_cm:.6f}",
                    "g": f"{level.g:.1f}",
                    "provenance": ion.provenance,
                }
            )

    if len(ground_rows) != 15:
        raise ValidationError(
            f"바닥준위 산출 행 수가 15가 아니다: {len(ground_rows)}"
        )
    if len(level_rows) != sum(len(ion.levels) for ion in ions):
        raise ValidationError("전 준위 산출 행 수가 메모리 자료와 다르다.")

    return {
        GROUND_OUTPUT: make_csv_text(GROUND_COLUMNS, ground_rows),
        LEVEL_OUTPUT: make_csv_text(LEVEL_COLUMNS, level_rows),
    }


def write_outputs_atomically(outputs: dict[Path, str]) -> None:
    """검증 완료 자료를 같은 디렉터리의 임시 파일에서 원자적으로 교체한다."""
    temporary_paths: dict[Path, Path] = {}

    try:
        for destination, contents in outputs.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                text=True,
            )
            temporary = Path(temporary_name)
            temporary_paths[destination] = temporary

            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())

        for destination, temporary in temporary_paths.items():
            os.replace(temporary, destination)
    finally:
        for temporary in temporary_paths.values():
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    try:
        ions = collect_and_validate()
        outputs = build_outputs(ions)
        write_outputs_atomically(outputs)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    for ion in ions:
        ground = ion.levels[0]
        print(
            f"{ion.target.label:<8} "
            f"levels={len(ion.levels):>4} "
            f"E0={ground.energy_cm:.6f} "
            f"g0={ground.g:.1f} "
            f"{ion.provenance}"
        )

    print(f"{len(ions)}개 이온 바닥준위 -> {GROUND_OUTPUT}")
    print(
        f"{sum(len(ion.levels) for ion in ions)}개 전 준위 -> "
        f"{LEVEL_OUTPUT}"
    )
    print("모든 fail-closed 검사 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```