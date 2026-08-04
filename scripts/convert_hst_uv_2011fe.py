#!/usr/bin/env python3
"""Convert downloaded HST/STIS x1d/sx1 FITS to per-epoch CSV.

Reads EXPSTART from extension 1 (primary header has it missing on these files).
Selects observations within ±15 days of SN 2011fe B-max (MJD 55814.0).
"""

from pathlib import Path
import numpy as np
from astropy.io import fits

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sn2011fe" / "hst_uv"
RAW_DIR = OUT_DIR / "raw"
BMAX_MJD = 55814.0


def read_x1d(f):
    with fits.open(f) as hdul:
        hdr0 = hdul[0].header
        hdr1 = hdul[1].header
        data = hdul[1].data
        info = {
            'detector': hdr0.get('DETECTOR', ''),
            'opt_elem': hdr0.get('OPT_ELEM', ''),
            'cenwave':  hdr0.get('CENWAVE', 0),
            'targname': hdr0.get('TARGNAME', ''),
            'mjd_start': hdr1.get('EXPSTART', None),
            'mjd_end':   hdr1.get('EXPEND', None),
            'texptime':  hdr0.get('TEXPTIME', hdr1.get('TEXPTIME', 0)),
            'date_obs':  hdr0.get('TDATEOBS', ''),
        }
        wave = np.asarray(data['WAVELENGTH']).flatten()
        flux = np.asarray(data['FLUX']).flatten()
        err  = np.asarray(data['ERROR']).flatten() if 'ERROR' in data.names \
               else np.zeros_like(flux)
        return info, wave, flux, err


def main():
    fits_files = sorted(set(list(RAW_DIR.rglob("*x1d.fits")) +
                            list(RAW_DIR.rglob("*sx1.fits"))))
    print(f"Found {len(fits_files)} FITS files")

    rows = []
    for f in fits_files:
        try:
            info, wave, flux, err = read_x1d(f)
        except Exception as e:
            print(f"  FAILED {f.name}: {e}")
            continue

        mjd = info['mjd_start']
        if mjd is None:
            print(f"  SKIP {f.name}: no EXPSTART")
            continue
        phase = mjd - BMAX_MJD

        tag = (f"{info['detector']}_{info['opt_elem']}"
               f"_mjd{mjd:.2f}_phase{phase:+05.1f}").replace('/', '_')
        out_csv = OUT_DIR / f"{tag}_{f.stem}.csv"
        arr = np.column_stack([wave, flux, err])
        np.savetxt(out_csv, arr, delimiter=',',
                   header="wavelength_angstrom,flux_erg_s_cm2_angstrom,error",
                   comments='', fmt='%.6e')
        rows.append({
            'file': f.name, 'csv': out_csv.name,
            'mjd_start': mjd, 'phase_d': phase,
            'detector': info['detector'], 'opt_elem': info['opt_elem'],
            'cenwave': info['cenwave'],
            'wave_min': float(wave.min()), 'wave_max': float(wave.max()),
            'n_pix': int(len(wave)), 'texp': info['texptime'],
        })
        print(f"  {f.name}: {info['detector']}/{info['opt_elem']} "
              f"MJD={mjd:.2f} ({phase:+5.1f}d) "
              f"[{wave.min():.0f}-{wave.max():.0f} A] -> {out_csv.name}")

    # Manifest
    if rows:
        import csv
        keys = list(rows[0].keys())
        manifest = OUT_DIR / "manifest.csv"
        with open(manifest, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nManifest: {manifest}")

        # Highlight epochs near B-max
        rows.sort(key=lambda r: abs(r['phase_d']))
        print(f"\n=== Closest to B-max (sorted by |phase|) ===")
        for r in rows[:10]:
            print(f"  phase={r['phase_d']:+6.1f}d  {r['detector']}/{r['opt_elem']}"
                  f"  cenwave={r['cenwave']}  range={r['wave_min']:.0f}-"
                  f"{r['wave_max']:.0f}A  -> {r['csv']}")


if __name__ == '__main__':
    main()
