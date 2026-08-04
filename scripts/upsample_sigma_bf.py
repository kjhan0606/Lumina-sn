#!/usr/bin/env python3
"""Upsample a baked cmfgen_sigma_bf.bin from N_old log-freq bins to N_new bins by
log-grid interpolation of each level's bf cross-section. bf sigma is a smooth
continuum (edges + hydrogenic decline), so resampling is faithful enough for the
frequency-resolution falsifier — the LINE (bb) forest, which is what we are
testing, is assembled at runtime at the new bin count from the line list.

Layout (little-endian): magic u32=0x434D4644, ver i32=1, n_levels i32, n_bins i32,
nu_min f64, nu_max f64, has_cmfgen[n_levels] u8, pad to 8B, sigma[n_levels*n_bins] f64 (C order).
"""
import struct, sys, numpy as np

src = sys.argv[1]
dst = sys.argv[2]
N_new = int(sys.argv[3])

with open(src, "rb") as f:
    magic, ver = struct.unpack("<Ii", f.read(8))
    n_levels, N_old = struct.unpack("<ii", f.read(8))
    nu_min, nu_max = struct.unpack("<dd", f.read(16))
    has = f.read(n_levels)
    pad = (8 - (n_levels % 8)) % 8
    f.read(pad)
    assert magic == 0x434D4644 and ver == 1
    print(f"src: n_levels={n_levels} N_old={N_old} nu=[{nu_min:.3e},{nu_max:.3e}]")
    # old/new log-grid bin centers (must match lumina.h cs->nu[b]=nu_min*exp((b+0.5)*dlog))
    dlo_old = np.log(nu_max / nu_min) / N_old
    dlo_new = np.log(nu_max / nu_min) / N_new
    nu_old = nu_min * np.exp((np.arange(N_old) + 0.5) * dlo_old)
    nu_new = nu_min * np.exp((np.arange(N_new) + 0.5) * dlo_new)
    lnu_old = np.log(nu_old); lnu_new = np.log(nu_new)

    with open(dst, "wb") as g:
        g.write(struct.pack("<Ii", magic, ver))
        g.write(struct.pack("<ii", n_levels, N_new))
        g.write(struct.pack("<dd", nu_min, nu_max))
        g.write(has)
        g.write(b"\x00" * pad)
        CH = 512  # levels per chunk (mem: CH*N_new*8)
        for lo in range(0, n_levels, CH):
            hi = min(lo + CH, n_levels)
            buf = np.frombuffer(f.read((hi - lo) * N_old * 8), dtype="<f8").reshape(hi - lo, N_old)
            out = np.empty((hi - lo, N_new), dtype="<f8")
            for i in range(hi - lo):
                # linear interp in log-nu; flat-extrapolate (np.interp default holds endpoints)
                out[i] = np.interp(lnu_new, lnu_old, buf[i])
            g.write(out.tobytes(order="C"))
            if lo % 4096 == 0:
                print(f"  {hi}/{n_levels} levels", flush=True)
print(f"wrote {dst}: n_levels={n_levels} n_bins={N_new}")
