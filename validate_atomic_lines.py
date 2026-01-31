#!/usr/bin/env python3
"""
Validate atomic line data for important spectral features.
"""

import pandas as pd
import numpy as np
import sys

def format_element(Z):
    """Get element symbol from atomic number."""
    elements = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
        9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
        16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
        23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn'
    }
    return elements.get(Z, f'Z={Z}')

def roman_numeral(ion):
    """Convert ion number to Roman numeral."""
    numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']
    return numerals[ion] if ion < len(numerals) else f'{ion+1}'

# Important spectral lines for Type Ia SNe
# Note: The atomic data uses slightly different wavelengths from NIST in some cases
IMPORTANT_LINES = {
    # Calcium H&K and IR triplet
    'Ca II K': {'Z': 20, 'ion': 1, 'lambda': 3934.77, 'ref_f': 0.688},
    'Ca II H': {'Z': 20, 'ion': 1, 'lambda': 3969.59, 'ref_f': 0.341},
    'Ca II IR 8498': {'Z': 20, 'ion': 1, 'lambda': 8500.0, 'ref_f': 0.0103},
    'Ca II IR 8542': {'Z': 20, 'ion': 1, 'lambda': 8544.0, 'ref_f': 0.0205},
    'Ca II IR 8662': {'Z': 20, 'ion': 1, 'lambda': 8664.0, 'ref_f': 0.0305},

    # Silicon - velocity diagnostic (injected lines at 6347/6371)
    'Si II 6347': {'Z': 14, 'ion': 1, 'lambda': 6347.10, 'ref_f': 0.708},
    'Si II 6371': {'Z': 14, 'ion': 1, 'lambda': 6371.37, 'ref_f': 0.419},
    # Natural lines in data
    'Si II 6349': {'Z': 14, 'ion': 1, 'lambda': 6349.0, 'ref_f': 0.99},
    'Si II 6373': {'Z': 14, 'ion': 1, 'lambda': 6373.0, 'ref_f': 0.50},
    'Si II 4128': {'Z': 14, 'ion': 1, 'lambda': 4128.0, 'ref_f': 0.3},
    'Si II 4131': {'Z': 14, 'ion': 1, 'lambda': 4131.0, 'ref_f': 0.5},
    'Si III 4553': {'Z': 14, 'ion': 2, 'lambda': 4553.0, 'ref_f': 0.5},
    'Si III 4568': {'Z': 14, 'ion': 2, 'lambda': 4568.0, 'ref_f': 0.4},

    # Sulfur "W" feature
    'S II 5432': {'Z': 16, 'ion': 1, 'lambda': 5432.0, 'ref_f': 0.1},
    'S II 5454': {'Z': 16, 'ion': 1, 'lambda': 5454.0, 'ref_f': 0.15},
    'S II 5606': {'Z': 16, 'ion': 1, 'lambda': 5606.0, 'ref_f': 0.1},
    'S II 5640': {'Z': 16, 'ion': 1, 'lambda': 5640.0, 'ref_f': 0.12},

    # Carbon
    'C II 6578': {'Z': 6, 'ion': 1, 'lambda': 6578.0, 'ref_f': 0.2},
    'C II 6583': {'Z': 6, 'ion': 1, 'lambda': 6583.0, 'ref_f': 0.1},

    # Oxygen
    'O I 7772': {'Z': 8, 'ion': 0, 'lambda': 7772.0, 'ref_f': 0.3},
    'O I 7774': {'Z': 8, 'ion': 0, 'lambda': 7774.0, 'ref_f': 0.5},
    'O I 7775': {'Z': 8, 'ion': 0, 'lambda': 7775.0, 'ref_f': 0.2},

    # Magnesium
    'Mg II 4481': {'Z': 12, 'ion': 1, 'lambda': 4481.0, 'ref_f': 0.7},
    'Mg II 2796': {'Z': 12, 'ion': 1, 'lambda': 2796.35, 'ref_f': 0.6},
    'Mg II 2803': {'Z': 12, 'ion': 1, 'lambda': 2803.53, 'ref_f': 0.3},

    # Iron-group elements
    'Fe II 4924': {'Z': 26, 'ion': 1, 'lambda': 4924.0, 'ref_f': 0.04},
    'Fe II 5018': {'Z': 26, 'ion': 1, 'lambda': 5018.0, 'ref_f': 0.04},
    'Fe II 5169': {'Z': 26, 'ion': 1, 'lambda': 5169.0, 'ref_f': 0.04},
    'Fe II 5276': {'Z': 26, 'ion': 1, 'lambda': 5276.0, 'ref_f': 0.03},
    'Fe II 5317': {'Z': 26, 'ion': 1, 'lambda': 5317.0, 'ref_f': 0.03},
    'Fe III 4420': {'Z': 26, 'ion': 2, 'lambda': 4420.0, 'ref_f': 0.05},
    'Fe III 5129': {'Z': 26, 'ion': 2, 'lambda': 5129.0, 'ref_f': 0.03},
    'Fe III 5156': {'Z': 26, 'ion': 2, 'lambda': 5156.0, 'ref_f': 0.02},

    # Cobalt
    'Co II 4161': {'Z': 27, 'ion': 1, 'lambda': 4161.0, 'ref_f': 0.02},
    'Co III 5888': {'Z': 27, 'ion': 2, 'lambda': 5888.0, 'ref_f': 0.02},

    # Nickel
    'Ni II 4067': {'Z': 28, 'ion': 1, 'lambda': 4067.0, 'ref_f': 0.02},
    'Ni II 7378': {'Z': 28, 'ion': 1, 'lambda': 7378.0, 'ref_f': 0.01},

    # Hydrogen (reference)
    'H alpha': {'Z': 1, 'ion': 0, 'lambda': 6564.6, 'ref_f': 0.6407},
    'H beta': {'Z': 1, 'ion': 0, 'lambda': 4862.7, 'ref_f': 0.1193},
    'H gamma': {'Z': 1, 'ion': 0, 'lambda': 4341.7, 'ref_f': 0.0447},

    # Helium (for core-collapse comparison)
    'He I 5876': {'Z': 2, 'ion': 0, 'lambda': 5876.0, 'ref_f': 0.7},
    'He I 6678': {'Z': 2, 'ion': 0, 'lambda': 6678.0, 'ref_f': 0.3},

    # Sodium
    'Na I D': {'Z': 11, 'ion': 0, 'lambda': 5893.0, 'ref_f': 0.64},
}

def main():
    filename = "atomic/kurucz_cd23_chianti_H_He.h5"

    print(f"Loading atomic data from: {filename}")

    try:
        lines = pd.read_hdf(filename, 'lines_data')
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Reset index to make columns accessible
    lines_flat = lines.reset_index()

    # Wavelength is already in Angstroms in this dataset
    lines_flat['wavelength_A'] = lines_flat['wavelength']

    print(f"\nTotal lines loaded: {len(lines_flat)}")
    print(f"Wavelength range: {lines_flat['wavelength_A'].min():.1f} - {lines_flat['wavelength_A'].max():.1f} Å")

    # Count lines per element
    print("\nLines per element (top 15):")
    element_counts = lines_flat.groupby('atomic_number').size().sort_values(ascending=False)
    for Z, count in element_counts.head(15).items():
        print(f"  {format_element(Z):>3}: {count:>6} lines")

    print("\n" + "="*110)
    print("  ATOMIC LINE VALIDATION FOR TYPE Ia SN SPECTROSCOPY")
    print("="*110)

    results = []

    # Group by category
    categories = {
        'Calcium (H&K, IR triplet)': ['Ca II K', 'Ca II H', 'Ca II IR 8498', 'Ca II IR 8542', 'Ca II IR 8662'],
        'Silicon (velocity diagnostic)': ['Si II 6347', 'Si II 6371', 'Si II 6349', 'Si II 6373',
                                          'Si II 4128', 'Si II 4131', 'Si III 4553', 'Si III 4568'],
        'Sulfur ("W" feature)': ['S II 5432', 'S II 5454', 'S II 5606', 'S II 5640'],
        'Carbon': ['C II 6578', 'C II 6583'],
        'Oxygen': ['O I 7772', 'O I 7774', 'O I 7775'],
        'Magnesium': ['Mg II 4481', 'Mg II 2796', 'Mg II 2803'],
        'Iron (Fe II/III blend)': ['Fe II 4924', 'Fe II 5018', 'Fe II 5169', 'Fe II 5276', 'Fe II 5317',
                                   'Fe III 4420', 'Fe III 5129', 'Fe III 5156'],
        'Cobalt': ['Co II 4161', 'Co III 5888'],
        'Nickel': ['Ni II 4067', 'Ni II 7378'],
        'Hydrogen (reference)': ['H alpha', 'H beta', 'H gamma'],
        'Helium (reference)': ['He I 5876', 'He I 6678'],
        'Sodium': ['Na I D'],
    }

    for category, line_names in categories.items():
        print(f"\n### {category}")
        print("-"*110)
        print(f"{'Line':<18} {'λ_ref [Å]':>12} {'λ_found [Å]':>14} {'Δλ [Å]':>10} {'f_lu':>10} {'A_ul [s⁻¹]':>12} {'Status':<10}")
        print("-"*110)

        for name in line_names:
            if name not in IMPORTANT_LINES:
                continue

            info = IMPORTANT_LINES[name]
            Z, ion, target_lambda = info['Z'], info['ion'], info['lambda']

            # Search with tolerance
            tolerance = 20.0 if info['lambda'] > 7000 else 15.0

            mask = (
                (lines_flat['atomic_number'] == Z) &
                (lines_flat['ion_number'] == ion) &
                (np.abs(lines_flat['wavelength_A'] - target_lambda) < tolerance)
            )

            matches = lines_flat[mask]

            if len(matches) == 0:
                print(f"{name:<18} {target_lambda:>12.2f} {'NOT FOUND':>14} {'-':>10} {'-':>10} {'-':>12} {'MISSING':<10}")
                results.append({
                    'name': name, 'status': 'MISSING', 'lambda_ref': target_lambda,
                    'Z': Z, 'ion': ion
                })
            else:
                # Find closest match with highest f_lu
                matches = matches.copy()
                matches['delta'] = np.abs(matches['wavelength_A'] - target_lambda)

                # If multiple matches, prefer the one with highest f_lu within 3 Angstroms
                close_matches = matches[matches['delta'] < 5.0]
                if len(close_matches) > 0:
                    best_idx = close_matches['f_lu'].idxmax()
                else:
                    best_idx = matches['delta'].idxmin()

                best = matches.loc[best_idx]

                found_lambda = best['wavelength_A']
                found_f = best['f_lu']
                found_A = best['A_ul']
                delta_lambda = found_lambda - target_lambda

                status = 'PASS' if abs(delta_lambda) < 5.0 else 'CHECK'

                print(f"{name:<18} {target_lambda:>12.2f} {found_lambda:>14.4f} {delta_lambda:>+10.4f} {found_f:>10.4f} {found_A:>12.4e} {status:<10}")

                results.append({
                    'name': name, 'status': status, 'lambda_ref': target_lambda,
                    'lambda_found': found_lambda, 'delta': delta_lambda,
                    'f_lu': found_f, 'A_ul': found_A, 'Z': Z, 'ion': ion
                })

    # Summary statistics
    print("\n" + "="*110)
    print("  SUMMARY")
    print("="*110)

    n_pass = sum(1 for r in results if r['status'] == 'PASS')
    n_check = sum(1 for r in results if r['status'] == 'CHECK')
    n_missing = sum(1 for r in results if r['status'] == 'MISSING')

    print(f"\n  Lines validated: {len(results)}")
    print(f"  PASS:    {n_pass}")
    print(f"  CHECK:   {n_check}")
    print(f"  MISSING: {n_missing}")

    # Print for markdown table
    print("\n\n### MARKDOWN TABLE OUTPUT ###\n")
    print("| Line | Element | λ_ref [Å] | λ_found [Å] | Δλ [Å] | f_lu | A_ul [s⁻¹] | Status |")
    print("|------|---------|-----------|-------------|--------|------|------------|--------|")

    for r in results:
        elem = f"{format_element(r['Z'])} {roman_numeral(r['ion'])}"
        if r['status'] == 'MISSING':
            print(f"| {r['name']} | {elem} | {r['lambda_ref']:.2f} | - | - | - | - | MISSING |")
        else:
            print(f"| {r['name']} | {elem} | {r['lambda_ref']:.2f} | {r['lambda_found']:.2f} | {r['delta']:+.2f} | {r['f_lu']:.4f} | {r['A_ul']:.2e} | {r['status']} |")

    return results

if __name__ == "__main__":
    main()
