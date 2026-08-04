#!/usr/bin/env python3
"""Download HST/STIS UV spectra of SN 2011fe near B-max from MAST.

Target observations (proposal GO-12298, Ellis et al.):
  - NUV-MAMA at MJD ~55801, ~55804 (B-max -13d, -10d)
  - FUV-MAMA at MJD ~55817 (B-max +3d, closest to B-max)
  - NUV-MAMA at MJD ~55823 (B-max +9d)

B-max for SN 2011fe: MJD 55814.0 = 2011-09-10 (B-band max).

Saves x1d.fits files to data/sn2011fe/hst_uv/raw/ and a combined CSV
spectrum per epoch to data/sn2011fe/hst_uv/.
"""

import sys
from pathlib import Path

from astroquery.mast import Observations
from astropy.io import fits
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sn2011fe" / "hst_uv"
RAW_DIR = OUT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BMAX_MJD = 55814.0


def main():
    print("Querying MAST for SN 2011fe HST/STIS observations...")
    obs = Observations.query_criteria(
        target_name="SN-2011FE",
        obs_collection="HST",
        instrument_name=["STIS/CCD", "STIS/NUV-MAMA", "STIS/FUV-MAMA"],
    )
    if len(obs) == 0:
        # Fallback by coordinates (M101 host)
        print("No matches by target_name; trying coordinates...")
        obs = Observations.query_criteria(
            coordinates="14h03m05.81s +54d16m25.4s",
            radius="0.001 deg",
            obs_collection="HST",
            instrument_name=["STIS/CCD", "STIS/NUV-MAMA", "STIS/FUV-MAMA"],
        )
    print(f"  Found {len(obs)} matches")
    if len(obs) == 0:
        print("ERROR: no HST/STIS data for SN 2011fe found.")
        sys.exit(1)

    # Filter to ±15 days from B-max in MJD
    cols = obs.colnames
    print(f"  Columns: {cols[:20]}")
    if 't_min' in cols:
        mask = np.abs(np.asarray(obs['t_min']) - BMAX_MJD) < 15.0
        obs = obs[mask]
        print(f"  After ±15d B-max filter: {len(obs)}")

    obs.write(OUT_DIR / "obs_table.csv", format="csv", overwrite=True)
    print(f"  Saved obs table -> {OUT_DIR / 'obs_table.csv'}")

    # Fetch product list (x1d science extracted spectra)
    print("\nFetching product list...")
    products = Observations.get_product_list(obs)
    print(f"  {len(products)} total products")
    sci = Observations.filter_products(
        products,
        productSubGroupDescription=["X1D", "SX1"],
        mrp_only=False,
    )
    print(f"  {len(sci)} science x1d/sx1 products")
    sci.write(OUT_DIR / "products.csv", format="csv", overwrite=True)

    if len(sci) == 0:
        print("WARNING: no x1d/sx1 products. Trying all extensions.")
        sci = products

    print("\nDownloading...")
    res = Observations.download_products(
        sci, download_dir=str(RAW_DIR), cache=True,
    )
    print(f"  Downloaded {len(res)} files")
    res.write(OUT_DIR / "downloads.csv", format="csv", overwrite=True)

    # Convert each x1d FITS to CSV (wave, flux, err)
    print("\nConverting FITS -> CSV...")
    fits_files = list(RAW_DIR.rglob("*x1d.fits")) + list(RAW_DIR.rglob("*sx1.fits"))
    print(f"  {len(fits_files)} FITS files")
    for f in fits_files:
        try:
            with fits.open(f) as hdul:
                hdr = hdul[0].header
                data = hdul[1].data
                mjd = hdr.get('EXPSTART', None)
                detector = hdr.get('DETECTOR', '')
                opt_elem = hdr.get('OPT_ELEM', '')
                wave = data['WAVELENGTH'].flatten()
                flux = data['FLUX'].flatten()
                err  = data['ERROR'].flatten() if 'ERROR' in data.names else \
                       np.zeros_like(flux)
            tag = f"{detector}_{opt_elem}_mjd{mjd:.2f}".replace('/', '_')
            out_csv = OUT_DIR / f"{tag}.csv"
            arr = np.column_stack([wave, flux, err])
            header = "wavelength_angstrom,flux_erg_s_cm2_angstrom,error"
            np.savetxt(out_csv, arr, delimiter=',', header=header,
                       comments='', fmt='%.6e')
            phase = mjd - BMAX_MJD if mjd else None
            print(f"  {f.name}: {detector}/{opt_elem} MJD={mjd:.2f} "
                  f"({phase:+.1f}d) -> {out_csv.name} "
                  f"(wave {wave.min():.0f}-{wave.max():.0f}A, n={len(wave)})")
        except Exception as e:
            print(f"  FAILED {f}: {e}")

    print(f"\nDone. Per-epoch CSVs in {OUT_DIR}")


if __name__ == '__main__':
    main()
