#!/usr/bin/env python3
"""True carbon ionization state from the per-stage ion-pop dump (lumina_ion_pops.csv).

The NLTE level dump is blind to non-NLTE ion stages (C I is Saha/φ_neb-treated,
not NLTE-tracked), so it cannot measure neutral carbon. This reads the full
ion-population dump instead and reports C I/C II/C III fractions + <Z_eff>.
"""
import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "lumina_ion_pops.csv"
Z = int(sys.argv[2]) if len(sys.argv) > 2 else 6  # carbon

df = pd.read_csv(path)
el = df[df.Z == Z]
print(f"# element Z={Z}  stages present: {sorted(el.stage.unique())}")
print(f"{'sh':>3} {'n_tot':>11}  " +
      "  ".join(f"f({Z},{s})" for s in sorted(el.stage.unique())) + "   <Zeff>")
for sh in sorted(el.shell_id.unique()):
    r = el[el.shell_id == sh]
    tot = r.n_ion.sum()
    if tot <= 0:
        continue
    fr = {int(s): float(v) / tot for s, v in zip(r.stage, r.n_ion)}
    zeff = sum(s * f for s, f in fr.items())
    fstr = "  ".join(f"{fr.get(s,0.0):6.4f}" for s in sorted(el.stage.unique()))
    print(f"{sh:3d} {tot:11.3e}  {fstr}   {zeff:5.3f}")
