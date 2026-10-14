"""CMFGEN의 개별 광이온화 표와 상위 목적 상태를 보존한다.

GET_NUM_PHOT_ROUTES_V1의 active-parent 경로 선택을 따른다. 서로 다른
A/B 단면적을 하나로 합친 뒤 확률로 재분배하지 않는다. 반환값은 새 채널별
단면적/율 입력을 만드는 용도이며 기존 MART v2의 단일 σ 행을 대체하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import math
import re
import numpy as np
from cmfgen_parser import OscIon, PhotIon, PhotEntry, parse_phot

CM_TO_EV = 1.239841984e-4


@dataclass(frozen=True)
class PhotoTable:
    path: Path
    data: PhotIon


@dataclass(frozen=True)
class PhotoRoute:
    table: int                    # 0부터 시작하는 독립 단면적 파일 번호
    target_level: int | None      # 상위 full-level anchor(0-based)
    target_super: int | None
    members: tuple[int, ...]
    excitation_cm1: float
    statistical_weight: float
    scale: float                  # 같은 파일 내부 g / sum(g)


@dataclass(frozen=True)
class LevelPhotoChannel:
    lower_level: int
    route: PhotoRoute
    entry: PhotEntry
    threshold_eV: float


def term_name(name: str) -> str:
    return re.sub(r'\[[^\[\]]*\]$', '', str(name).strip())


def read_photo_tables(first: Path) -> tuple[PhotoTable, ...]:
    """첫 파일이 선언한 A.. 경로만 읽는다. 최신 날짜/다른 파일을 추측하지 않는다."""
    first = Path(first)
    phot = parse_phot(first)
    if not 1 <= phot.n_routes <= 26:
        raise ValueError(f'{first}: invalid photo table count {phot.n_routes}')
    if phot.n_routes > 1 and not first.name.endswith('_A'):
        raise ValueError(f'{first}: multiple tables require explicit _A path')
    paths = [first] + [first.with_name(first.name[:-1] + chr(ord('A') + i))
                       for i in range(1, phot.n_routes)]
    tables = []
    for i, path in enumerate(paths):
        data = phot if i == 0 else parse_phot(path)
        if not data.final_state or data.final_state_g <= 0 or not math.isfinite(data.final_state_g):
            raise ValueError(f'{path}: missing final state/weight')
        if not math.isfinite(data.final_state_excit_eV) or data.final_state_excit_eV < 0:
            raise ValueError(f'{path}: invalid header excitation')
        if data.cs_unit != 'Megabarns' or len(data.entries) != data.n_levels_listed:
            raise ValueError(f'{path}: unsupported unit or incomplete cross sections')
        tables.append(PhotoTable(path, data))
    return tuple(tables)


def resolve_photo_routes(tables: tuple[PhotoTable, ...], upper: OscIon | None,
                         super_of_level: np.ndarray | None = None) -> tuple[PhotoRoute, ...]:
    """상위 이온이 있으면 이름/SL/osc 에너지가 정본이며 헤더의 에너지는 쓰지 않는다.

    upper=None은 원본의 terminal-parent 메타데이터 표현일 뿐이다. 이를 active
    full-level처럼 쓰거나 과거 원자수를 삭제하지 않는다. 실제 terminal 율식은
    별도의 물리 소비부가 구현해야 한다.
    """
    if not tables:
        raise ValueError('empty photo tables')
    if upper is not None:
        if len(upper.levels) == 0 or super_of_level is None or len(super_of_level) != len(upper.levels):
            raise ValueError('upper levels and super-level mapping must agree')
        if np.any(np.asarray(super_of_level) < 0):
            raise ValueError('negative super-level identity')
    routes = []
    for table, item in enumerate(tables):
        phot = item.data
        if upper is None:
            routes.append(PhotoRoute(table, None, None, (),
                phot.final_state_excit_eV / CM_TO_EV, phot.final_state_g, 1.0))
            continue
        selected = []
        for alternative in phot.final_state.split('/'):
            selected = [i for i, name in enumerate(upper.levels['config'])
                        if term_name(name) == alternative.strip()]
            if selected:
                break
        if not selected:
            raise ValueError(f'{item.path}: upper final state not represented: {phot.final_state}')
        first_super = int(super_of_level[selected[0]])
        local = []
        for level in selected:
            sl = int(super_of_level[level])
            g = float(upper.levels['g'][level])
            excitation = float(upper.levels['E_cm'][level] - upper.levels['E_cm'][0])
            if not math.isfinite(g) or g <= 0 or not math.isfinite(excitation) or excitation < 0:
                raise ValueError('invalid upper level energy/weight')
            # 원본 함수의 F_TO_S_SAVE는 첫 일치 SL이다. 임의로 전체 term이나
            # 마지막 SL 기준으로 재분류하지 않는다(직접 Fortran 대조 대상).
            if local and sl == first_super:
                previous = local[-1]
                local[-1] = replace(previous, members=previous.members + (level,),
                                    statistical_weight=previous.statistical_weight + g)
            else:
                local.append(PhotoRoute(table, level, sl, (level,), excitation, g, 0.0))
        gsum = sum(route.statistical_weight for route in local)
        routes.extend(replace(route, scale=route.statistical_weight / gsum) for route in local)
    if len(routes) > 40:
        raise ValueError('CMFGEN MAX_NUM_ROUTES exceeded')
    return tuple(routes)


def level_photo_channels(lower: OscIon, tables: tuple[PhotoTable, ...],
                         routes: tuple[PhotoRoute, ...], n_kept: int | None = None
                         ) -> tuple[LevelPhotoChannel, ...]:
    """각 하위 준위 × 각 독립 경로의 표·실제 문턱을 연결한다. 명시적 σ=0도 남긴다."""
    n = len(lower.levels) if n_kept is None else n_kept
    if not 0 < n <= len(lower.levels):
        raise ValueError('invalid active lower level count')
    entries = []
    for item in tables:
        mapping = {}
        for entry in item.data.entries:
            name = entry.config.strip()
            if name in mapping:
                raise ValueError(f'{item.path}: duplicate photo configuration {name}')
            mapping[name] = entry
        entries.append(mapping)
    channels = []
    for level in range(n):
        for route in routes:
            phot = tables[route.table].data
            name = str(lower.levels['config'][level]).strip()
            if not phot.split_J:
                name = term_name(name)
            if name not in entries[route.table]:
                raise ValueError(f'{tables[route.table].path}: no photo cross section for {name}')
            threshold = lower.ionization_eV + CM_TO_EV * (
                route.excitation_cm1 - float(lower.levels['E_cm'][level]))
            if not math.isfinite(threshold) or threshold <= 0:
                raise ValueError(f'invalid channel threshold: level={level} table={route.table}')
            channels.append(LevelPhotoChannel(level, route, entries[route.table][name], threshold))
    return tuple(channels)
