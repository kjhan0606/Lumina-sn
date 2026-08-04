#!/usr/bin/env python3
"""Direct ARTIS ts27, CMFGEN OBSFLUX, and LUMINA formal-spectrum comparison.

The ARTIS construction uses the ts27 escape-time window because this packet
snapshot is truncated at the end of ts27 and cannot contain a complete
observer-arrival-time window.  It retains nu_rf as the observed-frequency
coordinate, e_rf weighting, division by the eight rank ensembles, and
division by the ts27 duration and wavelength-bin width.

CMFGEN OBSFLUX is parsed from its named blocks.  Its frequency grid is in
10^15 Hz and "Observed intensity" is F_nu in Jy at 1 kpc.  LUMINA's formal
CSV stores L_lambda per cm, so conversion to per Angstrom is multiplication
by 1e-8.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"lumina-mplconfig-{os.getuid()}"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "Noto Sans CJK KR"

C_CGS = 2.99792458e10
DAY_S = 86400.0
PARSEC_CM = 3.0857e18
LSUN_CGS = 3.826e33
JY_CGS = 1.0e-23
ANGSTROM_PER_CM = 1.0e8
CM_PER_ANGSTROM = 1.0e-8
N_RANKS = 8
TS_INDEX = 27
ARTIS_NU_MIN_HZ = 1.0e13
ARTIS_NU_MAX_HZ = 5.0e15
ARTIS_NBINS = 1000

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
DEFAULT_ARTIS_DIR = ROOT.parent / "artis-ref/tests/toy06_ts1948"
DEFAULT_CMFGEN_OBSFLUX = Path(
    "/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/OBSFLUX"
)
DEFAULT_LUMINA_CSV = ROOT / "logs/coevolve_consume_parity54/lumina_spectrum_formal.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "validation/cmfgen_toy06_19p48d/analysis/"
    "artis_ts1948_vs_cmfgen_spectrum.png"
)

BANDS = (
    ("912-2000", 912.0, 2000.0),
    ("2000-2500", 2000.0, 2500.0),
    ("2500-5000", 2500.0, 5000.0),
    ("5000-10000", 5000.0, 10000.0),
)

PACKET_COLUMNS = (
    "number",
    "where",
    "type_id",
    "posx",
    "posy",
    "posz",
    "dirx",
    "diry",
    "dirz",
    "tdecay",
    "e_cmf",
    "e_rf",
    "nu_cmf",
    "nu_rf",
    "escape_type_id",
    "escape_time",
    "emissiontype",
    "trueemissiontype",
    "em_posx",
    "em_posy",
    "em_posz",
    "absorption_type",
    "absorption_freq",
    "nscatterings",
    "em_time",
    "originated_from_particlenotgamma",
    "trueem_posx",
    "trueem_posy",
    "trueem_posz",
    "trueem_time",
    "pellet_nucindex",
    "pellet_decaytype",
)


@dataclass
class Spectrum:
    name: str
    wavelength_A: np.ndarray
    luminosity_lambda: np.ndarray
    total_luminosity: float
    band_fractions: dict[str, float]


@dataclass
class ArtisSpectrum(Spectrum):
    wavelength_edges_A: np.ndarray
    selected_count: int
    raw_energy_erg: float
    t_start_days: float
    t_stop_days: float
    t_width_days: float
    max_arrival_days: float
    max_escape_days: float
    light_curve_luminosity: float


_FORTRAN_MISSING_E = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))([+-]\d+)$"
)


def fortran_float(token: str) -> float:
    """Parse E/D notation and CMFGEN fields such as ``5.6644-265``."""
    cleaned = token.replace("D", "E").replace("d", "e")
    try:
        return float(cleaned)
    except ValueError:
        match = _FORTRAN_MISSING_E.match(cleaned)
        if match is None:
            raise
        return float(f"{match.group(1)}E{match.group(2)}")


def read_numeric_block(lines: list[str], heading_index: int, count: int) -> np.ndarray:
    values: list[float] = []
    for line in lines[heading_index + 1 :]:
        for token in line.split():
            try:
                value = fortran_float(token)
            except ValueError:
                continue
            values.append(value)
            if len(values) == count:
                return np.asarray(values, dtype=float)
    raise ValueError(
        f"Block after line {heading_index + 1} ended at {len(values)} values; "
        f"expected {count}"
    )


def heading_index(lines: list[str], heading: str) -> int:
    matches = [i for i, line in enumerate(lines) if line.strip().startswith(heading)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {heading!r} block, found {len(matches)}")
    return matches[0]


def integrate_interval(
    wavelength_A: np.ndarray,
    luminosity_lambda: np.ndarray,
    lower_A: float,
    upper_A: float,
) -> float:
    """Piecewise-linear integral with interpolated values at band boundaries."""
    order = np.argsort(wavelength_A)
    x = wavelength_A[order]
    y = luminosity_lambda[order]
    lower = max(lower_A, float(x[0]))
    upper = min(upper_A, float(x[-1]))
    if lower >= upper:
        return 0.0
    inside = (x > lower) & (x < upper)
    xb = np.concatenate(([lower], x[inside], [upper]))
    yb = np.interp(xb, x, y)
    return float(np.trapezoid(yb, xb))


def packet_header(path: Path) -> tuple[str, ...]:
    with path.open() as handle:
        columns = tuple(handle.readline().lstrip("#").split())
    if columns != PACKET_COLUMNS:
        raise ValueError(
            f"{path}: packet header is not the documented 32-column schema"
        )
    return columns


def load_artis(artis_dir: Path) -> ArtisSpectrum:
    timestep_path = artis_dir / "timesteps.out"
    timesteps = np.loadtxt(timestep_path, comments="#")
    row = timesteps[timesteps[:, 0].astype(int) == TS_INDEX]
    next_row = timesteps[timesteps[:, 0].astype(int) == TS_INDEX + 1]
    if len(row) != 1 or len(next_row) != 1:
        raise ValueError(f"Cannot resolve ts{TS_INDEX} boundaries in {timestep_path}")
    t_start_days = float(row[0, 1])
    t_stop_days = float(next_row[0, 1])
    t_width_days = float(row[0, 3])
    t_start_s = t_start_days * DAY_S
    t_stop_s = t_stop_days * DAY_S

    files = sorted(artis_dir.glob("packets00_000[0-7].out"))
    if len(files) != N_RANKS:
        raise ValueError(f"Expected {N_RANKS} ARTIS rank files, found {len(files)}")

    use_names = (
        "type_id",
        "posx",
        "posy",
        "posz",
        "dirx",
        "diry",
        "dirz",
        "e_rf",
        "nu_rf",
        "escape_type_id",
        "escape_time",
    )
    usecols = tuple(PACKET_COLUMNS.index(name) for name in use_names)
    all_energy: list[np.ndarray] = []
    all_frequency: list[np.ndarray] = []
    all_arrival: list[np.ndarray] = []
    all_escape: list[np.ndarray] = []

    for path in files:
        packet_header(path)
        data = np.loadtxt(path, comments="#", usecols=usecols)
        type_id = data[:, 0].astype(np.int64)
        pos = data[:, 1:4]
        direction = data[:, 4:7]
        e_rf = data[:, 7]
        nu_rf = data[:, 8]
        escape_type_id = data[:, 9].astype(np.int64)
        escape_time = data[:, 10]
        arrival_time = escape_time - np.einsum("ij,ij->i", pos, direction) / C_CGS

        selected = (
            (type_id == 32)
            & (escape_type_id == 11)
            & (escape_time >= t_start_s)
            & (escape_time < t_stop_s)
        )
        all_energy.append(e_rf[selected])
        all_frequency.append(nu_rf[selected])
        all_arrival.append(arrival_time[selected])
        all_escape.append(escape_time[selected])

    energy_erg = np.concatenate(all_energy)
    frequency_hz = np.concatenate(all_frequency)
    arrival_s = np.concatenate(all_arrival)
    escape_s = np.concatenate(all_escape)
    if energy_erg.size == 0:
        raise ValueError(f"No escaped r-packets in the ts{TS_INDEX} escape window")

    # nu_rf is already the observer/rest-frame packet frequency at escape.
    wavelength_A = C_CGS / frequency_hz * ANGSTROM_PER_CM
    # Preserve the nominal ARTIS logarithmic resolution while extending the
    # edge only when selected nu_rf packets lie outside the standard grid.
    nominal_dlognu = np.log(ARTIS_NU_MAX_HZ / ARTIS_NU_MIN_HZ) / ARTIS_NBINS
    nu_min_hz = min(
        ARTIS_NU_MIN_HZ, float(np.nextafter(np.min(frequency_hz), 0.0))
    )
    nu_max_hz = max(
        ARTIS_NU_MAX_HZ, float(np.nextafter(np.max(frequency_hz), np.inf))
    )
    nbins = int(np.ceil(np.log(nu_max_hz / nu_min_hz) / nominal_dlognu))
    nu_edges = np.geomspace(nu_min_hz, nu_max_hz, nbins + 1)
    wavelength_edges_A = C_CGS / nu_edges[::-1] * ANGSTROM_PER_CM
    luminosity_weights = energy_erg / N_RANKS / (t_width_days * DAY_S)
    binned_luminosity, _ = np.histogram(
        wavelength_A, bins=wavelength_edges_A, weights=luminosity_weights
    )
    bin_width_A = np.diff(wavelength_edges_A)
    luminosity_lambda = binned_luminosity / bin_width_A
    wavelength_centers_A = np.sqrt(
        wavelength_edges_A[:-1] * wavelength_edges_A[1:]
    )
    total_luminosity = float(np.sum(binned_luminosity))

    direct_total = float(np.sum(energy_erg) / N_RANKS / (t_width_days * DAY_S))
    if not np.isclose(total_luminosity, direct_total, rtol=1e-12):
        raise ValueError("Selected ARTIS packets fell outside the standard spectrum grid")

    light_curve = np.loadtxt(artis_dir / "light_curve.out")
    lc_row = light_curve[np.isclose(light_curve[:, 0], float(row[0, 2]), atol=1.0e-5)]
    if len(lc_row) != 1:
        raise ValueError("Cannot resolve the ts27 row in ARTIS light_curve.out")
    light_curve_luminosity = float(lc_row[0, 1] * LSUN_CGS)

    band_fractions = {
        label: float(np.sum(energy_erg[(wavelength_A >= lo) & (wavelength_A < hi)]))
        / float(np.sum(energy_erg))
        for label, lo, hi in BANDS
    }
    return ArtisSpectrum(
        name="ARTIS",
        wavelength_A=wavelength_centers_A,
        luminosity_lambda=luminosity_lambda,
        total_luminosity=total_luminosity,
        band_fractions=band_fractions,
        wavelength_edges_A=wavelength_edges_A,
        selected_count=int(energy_erg.size),
        raw_energy_erg=float(np.sum(energy_erg)),
        t_start_days=t_start_days,
        t_stop_days=t_stop_days,
        t_width_days=t_width_days,
        max_arrival_days=float(np.max(arrival_s) / DAY_S),
        max_escape_days=float(np.max(escape_s) / DAY_S),
        light_curve_luminosity=light_curve_luminosity,
    )


def load_cmfgen(obsflux_path: Path) -> Spectrum:
    lines = obsflux_path.read_text().splitlines()
    frequency_heading = heading_index(lines, "Continuum Frequencies")
    count_match = re.search(r"\(\s*(\d+)\s*\)", lines[frequency_heading])
    if count_match is None:
        raise ValueError("OBSFLUX frequency count is missing")
    count = int(count_match.group(1))
    observed_heading = heading_index(lines, "Observed intensity (Janskys)")
    frequency_1e15_hz = read_numeric_block(lines, frequency_heading, count)
    flux_nu_jy = read_numeric_block(lines, observed_heading, count)

    frequency_hz = frequency_1e15_hz * 1.0e15
    wavelength_A = C_CGS / frequency_hz * ANGSTROM_PER_CM
    distance_cm = 1000.0 * PARSEC_CM
    luminosity_nu = 4.0 * np.pi * distance_cm**2 * flux_nu_jy * JY_CGS
    luminosity_lambda = (
        luminosity_nu
        * (C_CGS * ANGSTROM_PER_CM)
        / wavelength_A**2
    )
    order = np.argsort(wavelength_A)
    wavelength_A = wavelength_A[order]
    luminosity_lambda = luminosity_lambda[order]
    total_luminosity = float(np.trapezoid(luminosity_lambda, wavelength_A))
    band_fractions = {
        label: integrate_interval(wavelength_A, luminosity_lambda, lo, hi)
        / total_luminosity
        for label, lo, hi in BANDS
    }
    return Spectrum(
        name="CMFGEN OBSFLUX",
        wavelength_A=wavelength_A,
        luminosity_lambda=luminosity_lambda,
        total_luminosity=total_luminosity,
        band_fractions=band_fractions,
    )


def load_lumina(csv_path: Path) -> Spectrum:
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"{csv_path}: expected two CSV columns")
    wavelength_A = data[:, 0]
    # The CSV is L_lambda per cm.  d lambda_cm = 1e-8 d lambda_A.
    luminosity_lambda = data[:, 1] * CM_PER_ANGSTROM
    order = np.argsort(wavelength_A)
    wavelength_A = wavelength_A[order]
    luminosity_lambda = luminosity_lambda[order]
    total_luminosity = float(np.trapezoid(luminosity_lambda, wavelength_A))
    band_fractions = {
        label: integrate_interval(wavelength_A, luminosity_lambda, lo, hi)
        / total_luminosity
        for label, lo, hi in BANDS
    }
    return Spectrum(
        name="LUMINA formal",
        wavelength_A=wavelength_A,
        luminosity_lambda=luminosity_lambda,
        total_luminosity=total_luminosity,
        band_fractions=band_fractions,
    )


def write_band_table(path: Path, spectra: tuple[Spectrum, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["spectrum", "total_luminosity_erg_s"]
            + [f"fraction_{label}_A" for label, _, _ in BANDS]
        )
        for spectrum in spectra:
            writer.writerow(
                [spectrum.name, f"{spectrum.total_luminosity:.12e}"]
                + [f"{spectrum.band_fractions[label]:.12e}" for label, _, _ in BANDS]
            )


def print_report(
    artis: ArtisSpectrum,
    cmfgen: Spectrum,
    lumina: Spectrum,
    output_path: Path,
    band_table_path: Path,
) -> None:
    print(f"ARTIS ts{TS_INDEX} escape-time window: "
          f"[{artis.t_start_days:.5f}, {artis.t_stop_days:.5f}) d")
    print(f"ARTIS selected escaped r-packets: {artis.selected_count}")
    print(f"ARTIS raw sum(e_rf): {artis.raw_energy_erg:.12e} erg")
    print(f"ARTIS light_curve.out arrival-frame reference (truncated): "
          f"{artis.light_curve_luminosity:.12e} erg/s; not a cross-check for "
          "the escape-time spectrum")
    print(f"ARTIS max selected t_arrive/t_escape: "
          f"{artis.max_arrival_days:.6f}/{artis.max_escape_days:.6f} d")
    print()
    print("| spectrum | L_total [erg/s] | "
          + " | ".join(f"{label} A" for label, _, _ in BANDS) + " |")
    print("|---|---:|" + "---:|" * len(BANDS))
    for spectrum in (artis, cmfgen, lumina):
        fractions = " | ".join(
            f"{100.0 * spectrum.band_fractions[label]:.4f}%"
            for label, _, _ in BANDS
        )
        print(f"| {spectrum.name} | {spectrum.total_luminosity:.6e} | "
              f"{fractions} |")
    print()
    print("ARTIS frame note: escape-time binned — arrival-frame과 상이, "
          "truncated run의 표준 대안; wavelength = c/nu_rf.")
    print("CMFGEN: fixed-T·미수렴(MAXCH 3.46e3%)·자체 L=9.8e44"
          "(총입력 대비 ~90×) — 형상 참고용")
    print(f"Saved figure: {output_path}")
    print(f"Saved band table: {band_table_path}")


def make_figure(
    artis: ArtisSpectrum,
    cmfgen: Spectrum,
    lumina: Spectrum,
    output_path: Path,
) -> None:
    colors = {
        "artis": "#2673B8",
        "cmfgen": "#171717",
        "lumina": "#D45B3E",
    }
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.0, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.08, 1.0], "hspace": 0.08},
    )
    ax_abs, ax_shape = axes

    cmf_mask = (
        (cmfgen.wavelength_A >= 1000.0)
        & (cmfgen.wavelength_A <= 10000.0)
        & (cmfgen.luminosity_lambda > 0.0)
    )
    lum_mask = (
        (lumina.wavelength_A >= 1000.0)
        & (lumina.wavelength_A <= 10000.0)
        & (lumina.luminosity_lambda > 0.0)
    )
    artis_visible = (
        (artis.wavelength_A >= 1000.0)
        & (artis.wavelength_A <= 10000.0)
    )

    artis_label = (
        f"ARTIS ts27: L={artis.total_luminosity:.3e} erg s$^{{-1}}$; "
        f"{artis.selected_count} pkt, /8, escape-time binned"
    )
    cmfgen_label = (
        f"CMFGEN OBSFLUX: L={cmfgen.total_luminosity:.3e} erg s$^{{-1}}$; "
        "fixed-T·미수렴(MAXCH 3.46e3%)·자체 L=9.8e44\n"
        "(총입력 대비 ~90×) — 형상 참고용"
    )
    lumina_label = (
        f"LUMINA formal: L={lumina.total_luminosity:.3e} erg s$^{{-1}}$; "
        r"$L_\lambda$ cm$^{-1}\times10^{-8}$"
    )

    ax_abs.stairs(
        artis.luminosity_lambda[artis_visible],
        artis.wavelength_edges_A[
            np.flatnonzero(artis_visible)[0] : np.flatnonzero(artis_visible)[-1] + 2
        ],
        color=colors["artis"],
        linewidth=1.6,
        label=artis_label,
        zorder=4,
    )
    ax_abs.plot(
        cmfgen.wavelength_A[cmf_mask],
        cmfgen.luminosity_lambda[cmf_mask],
        color=colors["cmfgen"],
        linewidth=1.0,
        alpha=0.92,
        label=cmfgen_label,
        zorder=3,
    )
    ax_abs.plot(
        lumina.wavelength_A[lum_mask],
        lumina.luminosity_lambda[lum_mask],
        color=colors["lumina"],
        linewidth=1.35,
        alpha=0.92,
        label=lumina_label,
        zorder=2,
    )
    ax_abs.set_yscale("log")
    ax_abs.set_ylabel(r"$L_\lambda$ [erg s$^{-1}$ $\mathrm{\AA}^{-1}$]")
    ax_abs.set_title("Absolute observer-frame spectral luminosity")
    ax_abs.legend(loc="upper right", fontsize=8.2, framealpha=0.94)

    ax_shape.stairs(
        artis.luminosity_lambda[artis_visible] / artis.total_luminosity,
        artis.wavelength_edges_A[
            np.flatnonzero(artis_visible)[0] : np.flatnonzero(artis_visible)[-1] + 2
        ],
        color=colors["artis"],
        linewidth=1.6,
        label="ARTIS ts27 (escape-time binned)",
        zorder=4,
    )
    ax_shape.plot(
        cmfgen.wavelength_A[cmf_mask],
        cmfgen.luminosity_lambda[cmf_mask] / cmfgen.total_luminosity,
        color=colors["cmfgen"],
        linewidth=1.0,
        alpha=0.92,
        label=("CMFGEN: fixed-T·미수렴(MAXCH 3.46e3%)·자체 L=9.8e44\n"
               "(총입력 대비 ~90×) — 형상 참고용"),
        zorder=3,
    )
    ax_shape.plot(
        lumina.wavelength_A[lum_mask],
        lumina.luminosity_lambda[lum_mask] / lumina.total_luminosity,
        color=colors["lumina"],
        linewidth=1.35,
        alpha=0.92,
        label="LUMINA formal",
        zorder=2,
    )
    ax_shape.set_ylabel(r"shape $L_\lambda/\int L_\lambda\,d\lambda$ "
                        r"[$\mathrm{\AA}^{-1}$]")
    ax_shape.set_xlabel(r"observer-frame wavelength [$\mathrm{\AA}$]")
    ax_shape.set_title("Shape-normalized spectra (native-domain integral = 1)")
    ax_shape.legend(loc="upper right", fontsize=8.5, framealpha=0.94)

    for axis in axes:
        axis.set_xlim(1000.0, 10000.0)
        axis.grid(True, which="both", alpha=0.20, linewidth=0.6)
        for boundary in (2000.0, 2500.0, 5000.0):
            axis.axvline(
                boundary, color="#777777", linestyle=":", linewidth=0.7, alpha=0.45
            )

    fig.suptitle(
        "ARTIS ts1948 vs CMFGEN toy06 19.48 d"
        " (with LUMINA parity54 formal bonus)",
        fontsize=14,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "ARTIS: escape-time binned — arrival-frame과 상이, truncated run의 표준 대안; "
        f"ts27 [{artis.t_start_days:.4f},{artis.t_stop_days:.4f}) d; "
        r"$\lambda_{\rm obs}=c/\nu_{\rm rf}$.",
        ha="center",
        va="bottom",
        fontsize=8.2,
    )
    fig.subplots_adjust(top=0.93, bottom=0.105, left=0.09, right=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artis-dir", type=Path, default=DEFAULT_ARTIS_DIR)
    parser.add_argument(
        "--cmfgen-obsflux", type=Path, default=DEFAULT_CMFGEN_OBSFLUX
    )
    parser.add_argument("--lumina-csv", type=Path, default=DEFAULT_LUMINA_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--band-table",
        type=Path,
        default=None,
        help="CSV output (default: alongside PNG with _band_occupancy.csv suffix)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    band_table = args.band_table
    if band_table is None:
        band_table = args.output.with_name(
            args.output.stem + "_band_occupancy.csv"
        )
    artis = load_artis(args.artis_dir)
    cmfgen = load_cmfgen(args.cmfgen_obsflux)
    lumina = load_lumina(args.lumina_csv)
    spectra: tuple[Spectrum, ...] = (artis, cmfgen, lumina)
    make_figure(artis, cmfgen, lumina, args.output)
    write_band_table(band_table, spectra)
    print_report(artis, cmfgen, lumina, args.output, band_table)


if __name__ == "__main__":
    main()
