#!/usr/bin/env python3
"""
Convert flux-calibrated SN 2011fe multi-epoch spectra to consistent CSV format.

Sources: Snifs (Pereira+2013) for 18 epochs, WHT-ISIS for phase -10d.
All spectra have u_fluxes = erg/s/cm^2/Angstrom from the Open Supernova Catalog.

Explosion date: MJD 55796.7 (Nugent+2011)
B-band maximum: MJD 55814.0 (Pereira+2013)
=> t_exp at B-max ~ 17.3 days
"""

import os
import numpy as np

# Paths
SPECTRA_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..',
    'Lumina', 'data', 'sn2011fe', 'spectra'
)
OUT_DIR = os.path.dirname(__file__)
BMAX_CSV = os.path.join(os.path.dirname(__file__), '..', 'sn2011fe_observed_Bmax.csv')

# Constants
B_MAX_MJD = 55814.0
EXPLOSION_MJD = 55796.7  # Nugent+2011
WAVE_MIN = 3300.0   # Angstrom
WAVE_MAX = 10000.0  # Angstrom

# Selected calibrated spectra: (phase, filename, instrument)
SELECTED = [
    (-10.0, 'sn2011fe_m10d0d_11fe_20110831_WHT_v3.dat', 'WHT-ISIS'),
    ( -9.8, 'sn2011fe_m9d8d_SN2011fe-20110831-b10p3.dat', 'Snifs'),
    ( -8.8, 'sn2011fe_m8d8d_SN2011fe-20110901-b09p3.dat', 'Snifs'),
    ( -7.8, 'sn2011fe_m7d8d_SN2011fe-20110902-b08p3.dat', 'Snifs'),
    ( -6.7, 'sn2011fe_m6d7d_SN2011fe-20110903-b07p2.dat', 'Snifs'),
    ( -5.8, 'sn2011fe_m5d8d_SN2011fe-20110904-b06p3.dat', 'Snifs'),
    ( -4.8, 'sn2011fe_m4d8d_SN2011fe-20110905-b05p3.dat', 'Snifs'),
    ( -0.8, 'sn2011fe_m0d8d_SN2011fe-20110909-b01p3.dat', 'Snifs'),
    ( +0.2, 'sn2011fe_p0d2d_SN2011fe-20110910-b00p3.dat', 'Snifs'),
    ( +1.2, 'sn2011fe_p1d2d_SN2011fe-20110911-a00p7.dat', 'Snifs'),
    ( +2.2, 'sn2011fe_p2d2d_SN2011fe-20110912-a01p7.dat', 'Snifs'),
    ( +3.2, 'sn2011fe_p3d2d_SN2011fe-20110913-a02p7.dat', 'Snifs'),
    ( +4.2, 'sn2011fe_p4d2d_SN2011fe-20110914-a03p7.dat', 'Snifs'),
    ( +7.2, 'sn2011fe_p7d2d_SN2011fe-20110917-a06p7.dat', 'Snifs'),
    ( +9.2, 'sn2011fe_p9d2d_SN2011fe-20110919-a08p7.dat', 'Snifs'),
    (+12.2, 'sn2011fe_p12d2d_SN2011fe-20110922-a11p7.dat', 'Snifs'),
    (+14.2, 'sn2011fe_p14d2d_SN2011fe-20110924-a13p7.dat', 'Snifs'),
    (+17.2, 'sn2011fe_p17d2d_SN2011fe-20110927-a16p7.dat', 'Snifs'),
    (+19.2, 'sn2011fe_p19d2d_SN2011fe-20110929-a18p7.dat', 'Snifs'),
]


def read_spectrum(filepath):
    """Read wavelength/flux from archive file (# comment header, 2 columns)."""
    wave, flux = [], []
    with open(filepath) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) >= 2:
                w, fl = float(parts[0]), float(parts[1])
                if WAVE_MIN <= w <= WAVE_MAX and fl > 0:
                    wave.append(w)
                    flux.append(fl)
    return np.array(wave), np.array(flux)


def write_csv(filepath, wave, flux, phase, t_exp, instrument):
    """Write spectrum as CSV with metadata header."""
    with open(filepath, 'w') as f:
        f.write(f'# SN 2011fe observed spectrum\n')
        f.write(f'# Phase: {phase:+.1f} days from B-max (MJD {B_MAX_MJD})\n')
        f.write(f'# t_exp: {t_exp:.1f} days since explosion (MJD {EXPLOSION_MJD})\n')
        f.write(f'# Instrument: {instrument}\n')
        f.write(f'# Wavelength range: {wave[0]:.1f} - {wave[-1]:.1f} Angstrom\n')
        f.write('wavelength_angstrom,flux_erg_s_cm2_angstrom\n')
        for w, fl in zip(wave, flux):
            f.write(f'{w:.4f},{fl:.6e}\n')


def main():
    print(f'SN 2011fe Multi-Epoch Spectra Conversion')
    print(f'Explosion MJD: {EXPLOSION_MJD}, B-max MJD: {B_MAX_MJD}')
    print(f'Wavelength window: {WAVE_MIN}-{WAVE_MAX} Angstrom')
    print('=' * 80)

    catalog = []

    for phase, srcfile, instrument in SELECTED:
        t_exp = phase + (B_MAX_MJD - EXPLOSION_MJD)
        srcpath = os.path.join(SPECTRA_DIR, srcfile)

        if not os.path.exists(srcpath):
            print(f'  SKIP (not found): {srcfile}')
            continue

        wave, flux = read_spectrum(srcpath)
        if len(wave) < 10:
            print(f'  SKIP (too few points): {srcfile}')
            continue

        # Output filename: phase-based
        phase_tag = f'{phase:+.1f}'.replace('+', 'p').replace('-', 'm').replace('.', 'd')
        outname = f'sn2011fe_{phase_tag}d.csv'
        outpath = os.path.join(OUT_DIR, outname)

        write_csv(outpath, wave, flux, phase, t_exp, instrument)
        catalog.append((phase, t_exp, instrument, outname, len(wave),
                         wave[0], wave[-1]))
        print(f'  {phase:+6.1f}d  t_exp={t_exp:5.1f}d  {len(wave):5d} pts  '
              f'{wave[0]:.0f}-{wave[-1]:.0f}A  {instrument:10s} -> {outname}')

    # Also link the existing B-max CSV
    print(f'\nExisting B-max spectrum: sn2011fe_observed_Bmax.csv (phase ~ -0.3d)')

    # Write epoch index
    idx_path = os.path.join(OUT_DIR, 'epoch_index.csv')
    with open(idx_path, 'w') as f:
        f.write('phase_days,t_exp_days,instrument,filename,n_points,wave_min,wave_max\n')
        for phase, t_exp, instr, fname, npts, wmin, wmax in catalog:
            f.write(f'{phase:+.1f},{t_exp:.1f},{instr},{fname},{npts},{wmin:.1f},{wmax:.1f}\n')

    print(f'\nSaved {len(catalog)} epoch spectra + epoch_index.csv')
    print(f'Output directory: {OUT_DIR}')


if __name__ == '__main__':
    main()
