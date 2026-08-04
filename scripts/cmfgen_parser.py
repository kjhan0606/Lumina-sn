#!/usr/bin/env python3
"""CMFGEN atomic data parser.

Reads CMFGEN-format files for a single (element, ion):
  - osc_data         : energy levels + radiative transitions (f, A, lambda)
  - phot_data_A      : photoionization cross-sections sigma_bf(E) per level/term
  - col_data         : effective collision strengths Omega(T) per transition

Returns numpy structured arrays + dict tables that downstream code converts
to LUMINA's HDF5 schema. All file IO uses latin-1 to tolerate stray bytes.

Reference: Hillier & Miller 1998; current CMFGEN data compilation 19apr23.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import numpy as np


# CMFGEN header tags we look for in osc_data / phot_data / col_data
_TAG_RE = re.compile(r'^\s*([0-9.eE+\-DdRr]+)\s*!(.*)$')


def _read_text(path: Path) -> list[str]:
    with open(path, 'rb') as f:
        return f.read().decode('latin-1', errors='replace').splitlines()


def _parse_header_value(line: str) -> tuple[str | None, str]:
    """Match '<value>   !<key>' lines. Return (value_str, key)."""
    m = _TAG_RE.match(line)
    if not m:
        return None, ''
    return m.group(1).strip(), m.group(2).strip()


def _to_float(s: str) -> float:
    """CMFGEN sometimes uses Fortran 'D' exponents (1.234D-5)."""
    return float(s.replace('D', 'E').replace('d', 'e'))


# ---------------------------------------------------------------------------
# osc_data
# ---------------------------------------------------------------------------

@dataclass
class OscIon:
    n_levels: int = 0
    n_transitions: int = 0
    ionization_eV: float = 0.0   # converted from cm^-1
    z_screen: float = 0.0
    # Levels: structured array
    #   config(U64), g(f8), E_cm(f8), nu_1e15Hz(f8), E_eV(f8), lam_A(f8),
    #   ID(i4), A_rad(f8)
    levels: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Transitions: structured array
    #   i(i4)  -- 1-based lower level ID
    #   j(i4)  -- 1-based upper level ID
    #   f(f8)  -- absorption oscillator strength f_lu
    #   A(f8)  -- Einstein A_ul (s^-1)
    #   lam_A(f8)
    #   trans_id(i4)
    transitions: np.ndarray = field(default_factory=lambda: np.empty(0))


_LEVEL_DTYPE = np.dtype([
    ('config', 'U64'), ('g', 'f8'), ('E_cm', 'f8'),
    ('nu15', 'f8'), ('E_eV', 'f8'), ('lam_A', 'f8'),
    ('ID', 'i4'), ('A_rad', 'f8'),
])

_TRANS_DTYPE = np.dtype([
    ('i', 'i4'), ('j', 'i4'),
    ('f', 'f8'), ('A', 'f8'), ('lam_A', 'f8'),
    ('trans_id', 'i4'),
])

# CMFGEN units: 1 cm^-1 = 1.239841984e-4 eV
_CM2EV = 1.239841984e-4


def parse_osc(path: Path) -> OscIon:
    """Parse a CMFGEN osc_data file."""
    lines = _read_text(path)
    out = OscIon()

    # ---- Pass 1: header tags + locate level/transition blocks ----
    n_lev = 0
    n_trn = 0
    ion_cm = 0.0
    z_scr = 0.0
    for ln in lines:
        v, k = _parse_header_value(ln)
        if v is None:
            continue
        if 'Number of energy levels' in k and n_lev == 0:
            n_lev = int(v)
        elif 'Number of transitions' in k:
            n_trn = int(v)
        elif 'Ionization energy' in k:
            ion_cm = _to_float(v)
        elif 'Screened nuclear charge' in k:
            z_scr = _to_float(v)
        if n_lev and n_trn and ion_cm:
            break
    out.n_levels = n_lev
    out.n_transitions = n_trn
    out.ionization_eV = ion_cm * _CM2EV
    out.z_screen = z_scr

    # ---- Pass 2: parse level rows ----
    # Two known column layouts (CMFGEN files vary by date):
    #   modern (e.g. Fe II 19apr23, Si II 19apr23):
    #     config g E_cm nu15 E_eV lam_A ID ARAD [C4 C6]   -> 7..10 numeric cols
    #   old    (e.g. Mg I 14-Aug-1997):
    #     config g E_cm nu15 lam_A ID                     -> 5  numeric cols
    # Strategy: find the trailing integer ID (last token that is an int);
    # always treat tokens[1..2] as g, E_cm. Layout-specific fields (nu, eV,
    # lam_A, A_rad) are filled best-effort, otherwise zero.
    levels = np.empty(n_lev, dtype=_LEVEL_DTYPE)
    n_loaded = 0
    in_levels = False
    transition_block_start = None
    for idx, ln in enumerate(lines):
        if not in_levels:
            if 'E(cm' in ln or '!Number of energy levels' in ln:
                in_levels = True
            continue
        if n_loaded >= n_lev:
            transition_block_start = idx
            break
        s = ln.strip()
        if not s or s.startswith('!') or s.startswith('*'):
            continue
        toks = s.split()
        if len(toks) < 4:
            continue
        try:
            cfg = toks[0]
            # tokens[1..] must all be numeric for a valid level row
            nums = [_to_float(t) for t in toks[1:]]
        except ValueError:
            continue
        # ID is the only pure-integer token (no decimal point, no exponent).
        # CMFGEN sometimes emits negative IDs (-N) for super-level grouping;
        # we take abs() since LUMINA treats every level as distinct.
        ID = -1
        ID_pos = -1
        for k, t in enumerate(toks[1:]):
            stripped = t.strip()
            if stripped.startswith(('+', '-')):
                stripped = stripped[1:]
            if stripped.isdigit():
                cand = abs(int(t))
                if 0 < cand <= n_lev and cand == n_loaded + 1:
                    ID = cand
                    ID_pos = k
                    break
        if ID < 0:
            continue
        # Mandatory: g (nums[0]), E_cm (nums[1])
        g = nums[0]
        E_cm = nums[1]
        # Optional fields by layout (modern has 4 numeric fields before ID,
        # old has 3). Use ID_pos to disambiguate.
        if ID_pos >= 5:
            # modern: nu15=2, E_eV=3, lam_A=4
            nu15 = nums[2]
            E_eV = nums[3]
            lam = nums[4]
            A_rad = nums[ID_pos + 1] if len(nums) > ID_pos + 1 else 0.0
        elif ID_pos == 4:
            # 5-col plus ARAD?  e.g. config g E_cm nu lam_A ID ARAD
            nu15 = nums[2]
            E_eV = E_cm * _CM2EV
            lam = nums[3]
            A_rad = nums[ID_pos + 1] if len(nums) > ID_pos + 1 else 0.0
        else:
            # very compact: config g E_cm ID
            nu15 = 0.0
            E_eV = E_cm * _CM2EV
            lam = 0.0
            A_rad = 0.0
        levels[n_loaded] = (cfg, g, E_cm, nu15, E_eV, lam, ID, A_rad)
        n_loaded += 1

    if n_loaded != n_lev:
        # Truncate: some files have fewer rows than the tag claims.
        levels = levels[:n_loaded]
        out.n_levels = n_loaded
    out.levels = levels

    # ---- Pass 3: parse transition rows ----
    # Each transition row looks like:
    #   <cfg_l> -<cfg_u>   f  A  lam   i-j  trans#  ...
    # We anchor on the regex for the 3-float block followed by "i- j" pair.
    # Use a robust regex on the joined remainder:
    if transition_block_start is None:
        transition_block_start = idx + 1

    # Pattern: 3 floats, then "<int>-<int>", optional trans_id, optional "|...".
    # Some ions (Si II, S II, ...) omit the explicit Trans. # column;
    # we fall back to sequential numbering in that case.
    # Floats can use D/E exponents.
    fnum = r'[-+]?\d+\.?\d*(?:[DdEe][-+]?\d+)?'
    trn_re = re.compile(
        rf'\s({fnum})\s+({fnum})\s+({fnum})\s+(\d+)\s*-\s*(\d+)'
        rf'(?:\s+(\d+))?(?=\s|\||$)'
    )

    trans = np.empty(n_trn, dtype=_TRANS_DTYPE)
    n_t = 0
    for ln in lines[transition_block_start:]:
        m = trn_re.search(ln)
        if not m:
            continue
        if n_t >= n_trn:
            break
        try:
            f = _to_float(m.group(1))
            A = _to_float(m.group(2))
            lam = _to_float(m.group(3))
            i = int(m.group(4))
            j = int(m.group(5))
            trans_id = int(m.group(6)) if m.group(6) is not None else (n_t + 1)
        except ValueError:
            continue
        trans[n_t] = (i, j, f, A, lam, trans_id)
        n_t += 1

    if n_t != n_trn:
        trans = trans[:n_t]
        out.n_transitions = n_t
    out.transitions = trans
    return out


# ---------------------------------------------------------------------------
# phot_data_A
# ---------------------------------------------------------------------------

@dataclass
class PhotEntry:
    config: str
    cs_type: int                # CMFGEN type code (20 = tabulated, 1 = Seaton fit, ...)
    n_points: int
    energy: np.ndarray          # shape (n,) — units: ionization-energy ratios for type 20
    sigma_Mb: np.ndarray        # shape (n,) — megabarns


@dataclass
class PhotIon:
    n_levels_listed: int = 0
    n_routes: int = 0
    z_screen: float = 0.0
    final_state: str = ''
    final_state_excit_eV: float = 0.0
    final_state_g: float = 0.0
    cs_unit: str = 'Megabarns'
    split_J: bool = False
    n_data_pairs: int = 0
    entries: list[PhotEntry] = field(default_factory=list)


def parse_phot(path: Path) -> PhotIon:
    """Parse a CMFGEN phot_data_A file. Currently focuses on type 20 (tabulated)."""
    lines = _read_text(path)
    out = PhotIon()

    # ---- header ----
    body_start = 0
    for idx, ln in enumerate(lines):
        v, k = _parse_header_value(ln)
        if v is None:
            continue
        if 'Number of energy levels' in k:
            out.n_levels_listed = int(v)
        elif 'Number of photoionization routes' in k:
            out.n_routes = int(v)
        elif 'Screened nuclear charge' in k:
            out.z_screen = _to_float(v)
        elif 'Final state in ion' in k:
            out.final_state = v
        elif 'Excitation energy of final state' in k:
            out.final_state_excit_eV = _to_float(v) * _CM2EV
        elif 'Statistical weight of ion' in k:
            out.final_state_g = _to_float(v)
        elif 'Cross-section unit' in k:
            out.cs_unit = v
        elif 'Split J levels' in k:
            out.split_J = (v.strip().lower() == 'true')
        elif 'Total number of data pairs' in k:
            out.n_data_pairs = int(v)
            body_start = idx + 1
            break

    # ---- per-entry blocks ----
    # CMFGEN cs_type semantics (Hillier & Miller 1998 + later additions):
    #   1, 6, 7      : Seaton fits — npts = 1 or 3 *scalar* parameters per row
    #   2, 8, 9      : hydrogenic / Seaton variants — npts = 1..6 scalars
    #   20, 21, 22   : tabulated (E_ratio, sigma_Mb) pairs
    # We collect *all* numeric tokens in the next npts data lines and reshape.
    # For tabulated types we get (E, sigma) pairs; for fit types we keep the
    # raw parameter array in the `sigma_Mb` field with `energy` empty so
    # downstream code can dispatch on `cs_type`.

    TABULATED = {20, 21, 22}

    i = body_start
    n = len(lines)
    entries: list[PhotEntry] = []
    while i < n:
        # locate next "Configuration name" line
        while i < n and 'Configuration name' not in lines[i]:
            i += 1
        if i >= n:
            break
        config = lines[i].split('!')[0].strip()
        i += 1

        # Type — must be next non-blank, non-banner header line
        cs_type = 0
        while i < n and 'Type of cross-section' not in lines[i]:
            i += 1
        if i >= n:
            break
        try:
            cs_type = int(lines[i].split('!')[0].strip())
        except ValueError:
            i += 1
            continue
        i += 1

        # Number of points / parameters
        while i < n and 'Number of cross-section points' not in lines[i]:
            i += 1
        if i >= n:
            break
        try:
            npts = int(lines[i].split('!')[0].strip())
        except ValueError:
            i += 1
            continue
        i += 1

        # Read up to `npts` data lines (each line has 1+ numeric tokens).
        # Stop early if we hit the next entry header.
        nums: list[float] = []
        rows_consumed = 0
        while rows_consumed < npts and i < n:
            ln = lines[i].strip()
            # Reaching the next "Configuration name" / banner ends this entry
            if 'Configuration name' in ln or '!' in ln:
                # Header line — stop without consuming
                break
            if not ln or ln.startswith('*'):
                i += 1
                continue
            toks = ln.split()
            line_nums: list[float] = []
            ok = True
            for t in toks:
                try:
                    line_nums.append(_to_float(t))
                except ValueError:
                    ok = False
                    break
            if not ok or not line_nums:
                # Stop on a non-numeric row to avoid runaway parsing
                break
            nums.extend(line_nums)
            rows_consumed += 1
            i += 1

        if not nums:
            continue

        if cs_type in TABULATED and len(nums) >= 2:
            # Pairs (E, sigma)
            arr = np.array(nums, dtype='f8')
            n_pairs = arr.size // 2
            arr = arr[:2 * n_pairs].reshape(n_pairs, 2)
            entries.append(PhotEntry(
                config=config, cs_type=cs_type, n_points=n_pairs,
                energy=arr[:, 0].copy(), sigma_Mb=arr[:, 1].copy(),
            ))
        else:
            # Fit / hydrogenic — keep raw parameters
            arr = np.array(nums, dtype='f8')
            entries.append(PhotEntry(
                config=config, cs_type=cs_type, n_points=arr.size,
                energy=np.empty(0, dtype='f8'), sigma_Mb=arr,
            ))
    out.entries = entries
    return out


# ---------------------------------------------------------------------------
# col_data
# ---------------------------------------------------------------------------

@dataclass
class ColIon:
    n_transitions: int = 0
    n_T: int = 0
    scale_factor: float = 1.0
    default_omega: float = 0.0
    T_grid_kK: np.ndarray = field(default_factory=lambda: np.empty(0))   # in 10^4 K
    # Each entry: (config_lower, config_upper, omega[n_T])
    entries: list[tuple[str, str, np.ndarray]] = field(default_factory=list)


def parse_col(path: Path) -> ColIon:
    """Parse CMFGEN col_data file (Omega tabulated at fixed T grid)."""
    lines = _read_text(path)
    out = ColIon()

    body_start = 0
    n_t = 0
    for idx, ln in enumerate(lines):
        v, k = _parse_header_value(ln)
        if v is not None:
            if 'Number of transitions' in k:
                out.n_transitions = int(v)
            elif 'Number of T values' in k:
                out.n_T = int(v)
                n_t = out.n_T
            elif 'Scaling factor' in k:
                out.scale_factor = _to_float(v)
            elif 'Value for OMEGA if f=0' in k or "Value for OMEGA" in k:
                try:
                    out.default_omega = _to_float(v)
                except ValueError:
                    pass
        # The T-grid line begins with "Transition\T" (or contains it).
        if 'Transition' in ln and '\\T' in ln and n_t > 0:
            toks = ln.split()
            # Strip the "Transition\T" token; remaining are T values.
            t_vals = []
            for tk in toks[1:]:
                try:
                    t_vals.append(_to_float(tk))
                except ValueError:
                    continue
            if len(t_vals) >= n_t:
                out.T_grid_kK = np.array(t_vals[:n_t], dtype='f8')
                body_start = idx + 1
                break

    # transition rows
    if out.n_T == 0 or out.T_grid_kK.size == 0:
        return out

    fnum = r'[-+]?\d+\.?\d*(?:[DdEe][-+]?\d+)?'
    omega_re = re.compile(
        r'^\s*(\S+)\s*-\s*(\S+)\s+((?:' + fnum + r'\s+){' +
        str(out.n_T - 1) + r'}' + fnum + r')\s*$'
    )

    entries: list[tuple[str, str, np.ndarray]] = []
    for ln in lines[body_start:]:
        m = omega_re.match(ln)
        if not m:
            continue
        cfg_l = m.group(1)
        cfg_u = m.group(2)
        try:
            arr = np.array([_to_float(x) for x in m.group(3).split()],
                           dtype='f8')
        except ValueError:
            continue
        if arr.size == out.n_T:
            entries.append((cfg_l, cfg_u, arr))
    out.entries = entries
    return out


# ---------------------------------------------------------------------------
# f_to_s  (super-level assignments)
# ---------------------------------------------------------------------------

@dataclass
class FtoS:
    """CMFGEN full-level -> super-level map (XzV_F_TO_S files).

    n_levels : number of full levels (header) -- matches osc_data ID order.
    n_super  : number of super levels (max SL index).
    sl_of_fl : int array length n_levels, 0-based super-level index per
               full level, indexed by (osc cmfgen_id - 1).
    """
    n_levels: int = 0
    n_super: int = 0
    sl_of_fl: np.ndarray = field(default_factory=lambda: np.empty(0, dtype='i4'))
    format_name: str = ''
    format_basis: str = ''


def parse_f_to_s(path: Path, *, _test_explicit_fl_column_offset: int = 0,
                 _test_force_format: str | None = None) -> FtoS:
    """Parse a CMFGEN f_to_s (super-level assignment) file.

    Supported canonical layouts are detected from the complete body, never
    guessed a row at a time:

    ``explicit_fl_id``
        ``config g E nu Lam SL aux FL_id``.  ``FL_id`` is the final token and
        must be an exact permutation of ``1..Number of energy levels``.

    ``implicit_fl_row_order``
        Old layout ``config g E nu Lam SL 0``.  There is no FL-ID column; the
        full-level ID is the 1-based ordinal of the data row.  To prevent a
        damaged explicit file silently falling back to this layout, every row
        must have exactly one token after the declared SL column and that token
        must be integer zero.

    The header tag ``Entry number of link to super level`` gives the 1-based
    token position of the SL column (configuration counted as token 1).  Both
    layouts are self-validated for complete, one-to-one FL coverage and a
    contiguous ``1..N`` set of super-level IDs.

    The two ``_test_*`` arguments are intentionally private fault-injection
    hooks used only by the R4 negative fixture.  Production callers must leave
    them at their defaults.
    """
    lines = _read_text(path)
    n_levels = 0
    sl_token = 6  # 1-based; overridden by header if present
    body_start = 0
    for lineno, ln in enumerate(lines):
        v, key = _parse_header_value(ln)
        if v is None:
            continue
        kl = key.lower()
        if 'number of energy levels' in kl:
            n_levels = int(float(v))
        elif 'link to super level' in kl:
            sl_token = int(float(v))
            body_start = lineno + 1
    if n_levels <= 0:
        raise ValueError(f"f_to_s {path}: no level count in header")
    if sl_token <= 1:
        raise ValueError(f"f_to_s {path}: invalid SL token position {sl_token}")

    sl_col = sl_token - 1  # 0-based index after split
    rows: list[tuple[int, list[str], int]] = []
    for lineno, ln in enumerate(lines[body_start:], body_start + 1):
        toks = ln.split()
        if len(toks) <= sl_col:
            continue
        try:
            # Requiring the physical numeric columns rejects prose/header rows
            # that happen to carry an integer in the declared SL position.
            float(toks[1].replace('D', 'E').replace('d', 'e'))
            float(toks[2].replace('D', 'E').replace('d', 'e'))
            sl = int(toks[sl_col])
        except (ValueError, IndexError):
            continue
        if sl < 1:
            raise ValueError(f"f_to_s {path}:{lineno}: nonpositive SL id {sl}")
        rows.append((lineno, toks, sl))

    if len(rows) != n_levels:
        raise ValueError(f"f_to_s {path}: mapped-row count {len(rows)} != "
                         f"declared FL count {n_levels}")

    # New layout: the selected final column must itself prove that it is an FL
    # ID, by being a complete permutation.  Merely being integer is not enough.
    explicit_ids: list[int] = []
    explicit_column_present = True
    for _lineno, toks, _sl in rows:
        col = len(toks) - 1 + _test_explicit_fl_column_offset
        if col <= sl_col:
            explicit_column_present = False
            break
        try:
            explicit_ids.append(int(toks[col]))
        except (ValueError, IndexError):
            explicit_column_present = False
            break
    explicit_valid = (explicit_column_present and
                      sorted(explicit_ids) == list(range(1, n_levels + 1)))

    # Old layout signature is deliberately narrow.  In particular, an
    # 8-column explicit row whose FL column was corrupted to zero cannot be
    # reinterpreted as row-order implicit.
    implicit_valid = all(
        len(toks) == sl_col + 2 and toks[sl_col + 1] == '0'
        for _lineno, toks, _sl in rows
    )
    if _test_force_format is not None:
        if _test_force_format == 'explicit_fl_id':
            implicit_valid = False
        elif _test_force_format == 'implicit_fl_row_order':
            explicit_valid = False
        else:
            raise ValueError(f"invalid test-only forced format {_test_force_format!r}")

    if explicit_valid and implicit_valid:
        raise ValueError(f"f_to_s {path}: ambiguous explicit/implicit format")
    if explicit_valid:
        format_name = 'explicit_fl_id'
        format_basis = (
            f"header SL token={sl_token}; final token is exact FL-ID "
            f"permutation 1..{n_levels}"
        )
        fl_ids = explicit_ids
    elif implicit_valid:
        format_name = 'implicit_fl_row_order'
        format_basis = (
            f"header SL token={sl_token}; every data row has exactly one "
            "post-SL integer-zero auxiliary token and no FL-ID token; "
            f"FL ID is data-row ordinal 1..{n_levels}"
        )
        fl_ids = list(range(1, n_levels + 1))
    else:
        reason = (
            f"selected explicit FL column is not permutation 1..{n_levels}; "
            f"old row-order signature (exactly {sl_col + 2} tokens with "
            "post-SL zero) is false"
        )
        raise ValueError(f"f_to_s {path}: unrecognised/invalid format: {reason}")

    counts = np.bincount(np.asarray(fl_ids, dtype='i4'), minlength=n_levels + 1)
    bad_fl = np.flatnonzero(counts[1:] != 1) + 1
    if bad_fl.size:
        preview = ','.join(str(int(value)) for value in bad_fl[:8])
        raise ValueError(f"f_to_s {path}: FL IDs not mapped exactly once: {preview}")

    sl_of_fl = np.full(n_levels, -1, dtype='i4')
    for fl_id, (_lineno, _toks, sl) in zip(fl_ids, rows, strict=True):
        if sl_of_fl[fl_id - 1] != -1:
            raise ValueError(f"f_to_s {path}: duplicate FL id {fl_id}")
        sl_of_fl[fl_id - 1] = sl - 1
    mapped = int(np.count_nonzero(sl_of_fl >= 0))
    if mapped != n_levels:
        raise ValueError(f"f_to_s {path}: mapped FL count {mapped} != "
                         f"declared FL count {n_levels}")

    observed_sl = sorted(set((sl_of_fl + 1).tolist()))
    max_sl = observed_sl[-1] if observed_sl else 0
    expected_sl = list(range(1, max_sl + 1))
    if observed_sl != expected_sl:
        missing_sl = sorted(set(expected_sl) - set(observed_sl))
        raise ValueError(f"f_to_s {path}: non-contiguous SL ids 1..{max_sl}; "
                         f"holes={missing_sl}")

    return FtoS(n_levels=n_levels, n_super=max_sl, sl_of_fl=sl_of_fl,
                format_name=format_name, format_basis=format_basis)


# ---------------------------------------------------------------------------
# Main: smoke test on Fe II 19apr23
# ---------------------------------------------------------------------------

def _smoke_test():
    root = Path('/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/cmfgen')
    fe2 = root / 'FE' / 'II' / '19apr23'

    print(f"=== Fe II 19apr23 ===")
    print(f"Path: {fe2}")

    print("\n[osc_data]")
    osc = parse_osc(fe2 / 'osc_data')
    print(f"  header: n_levels={osc.n_levels}, n_trans={osc.n_transitions}, "
          f"E_ion={osc.ionization_eV:.3f} eV, Z_scr={osc.z_screen}")
    print(f"  levels loaded: {len(osc.levels)}, "
          f"trans loaded: {len(osc.transitions)}")
    if len(osc.levels):
        l0 = osc.levels[0]
        print(f"  level[0]: cfg={l0['config']!r}, g={l0['g']}, "
              f"E={l0['E_cm']} cm-1, ID={l0['ID']}")
    if len(osc.transitions):
        t0 = osc.transitions[0]
        print(f"  trans[0]: {t0['i']}->{t0['j']}, f={t0['f']:.3e}, "
              f"A={t0['A']:.3e}, lam={t0['lam_A']:.2f} A, id={t0['trans_id']}")
        # Sanity: lam(A) -> nu and check matches (i,j) energy difference
        cm_lo = osc.levels['E_cm'][t0['i'] - 1]
        cm_up = osc.levels['E_cm'][t0['j'] - 1]
        lam_check = 1.0e8 / (cm_up - cm_lo) if cm_up > cm_lo else 0.0
        print(f"    energy-diff lambda = {lam_check:.2f} A "
              f"(file: {t0['lam_A']:.2f})")

    print("\n[phot_data_A]")
    phot = parse_phot(fe2 / 'phot_data_A')
    print(f"  header: n_levels_listed={phot.n_levels_listed}, "
          f"routes={phot.n_routes}, n_data_pairs={phot.n_data_pairs}, "
          f"split_J={phot.split_J}")
    print(f"  entries parsed: {len(phot.entries)}")
    total_pts = sum(e.n_points for e in phot.entries)
    print(f"  total data points: {total_pts} (header: {phot.n_data_pairs})")
    if phot.entries:
        e = phot.entries[0]
        print(f"  entry[0]: cfg={e.config!r}, type={e.cs_type}, "
              f"npts={e.n_points}, E[0:3]={e.energy[:3]}, "
              f"sigma[0:3]={e.sigma_Mb[:3]}")

    print("\n[col_data]")
    col = parse_col(fe2 / 'col_data')
    print(f"  header: n_trans={col.n_transitions}, n_T={col.n_T}, "
          f"scale={col.scale_factor}")
    print(f"  T grid (10^4 K): {col.T_grid_kK}")
    print(f"  entries parsed: {len(col.entries)}")
    if col.entries:
        cfg_l, cfg_u, om = col.entries[0]
        print(f"  entry[0]: {cfg_l} -> {cfg_u}")
        print(f"           Omega = {om}")

    return osc, phot, col


if __name__ == '__main__':
    _smoke_test()
