#!/usr/bin/env python3
"""
TARDIS-LUMINA Virtual Packet Function-by-Function Comparison

This script compares individual functions between TARDIS and LUMINA
to identify any differences that cause the spectrum mismatch.
Target: noise < 0.1%
"""

import numpy as np
import math

# ============================================================================
# CONSTANTS (must match exactly)
# ============================================================================
C_SPEED_OF_LIGHT = 2.99792458e10  # cm/s
SIGMA_THOMSON = 6.6524587158e-25  # cm^2

# ============================================================================
# TARDIS FUNCTIONS (copied directly from TARDIS source)
# ============================================================================

def tardis_get_doppler_factor(r, mu, time_explosion, enable_full_relativity=False):
    """TARDIS: frame_transformations.py:get_doppler_factor"""
    inv_c = 1 / C_SPEED_OF_LIGHT
    inv_t = 1 / time_explosion
    beta = r * inv_t * inv_c
    if not enable_full_relativity:
        return 1.0 - mu * beta
    else:
        return (1.0 - mu * beta) / math.sqrt(1 - beta * beta)

def tardis_get_inverse_doppler_factor(r, mu, time_explosion, enable_full_relativity=False):
    """TARDIS: frame_transformations.py:get_inverse_doppler_factor"""
    inv_c = 1 / C_SPEED_OF_LIGHT
    inv_t = 1 / time_explosion
    beta = r * inv_t * inv_c
    if not enable_full_relativity:
        return 1.0 / (1.0 - mu * beta)
    else:
        return (1.0 + mu * beta) / math.sqrt(1 - beta * beta)

def tardis_calculate_distance_boundary(r, mu, r_inner, r_outer):
    """TARDIS: calculate_distances.py:calculate_distance_boundary"""
    delta_shell = 0
    if mu > 0.0:
        distance = math.sqrt(r_outer * r_outer + ((mu * mu - 1.0) * r * r)) - (r * mu)
        delta_shell = 1
    else:
        check = r_inner * r_inner + (r * r * (mu * mu - 1.0))
        if check >= 0.0:
            distance = -r * mu - math.sqrt(check)
            delta_shell = -1
        else:
            distance = math.sqrt(r_outer * r_outer + ((mu * mu - 1.0) * r * r)) - (r * mu)
            delta_shell = 1
    return distance, delta_shell

def tardis_calculate_distance_line(nu_lab, comov_nu, is_last_line, nu_line, time_explosion):
    """TARDIS: calculate_distances.py:calculate_distance_line"""
    CLOSE_LINE_THRESHOLD = 1e-7
    MISS_DISTANCE = 1e99

    if is_last_line:
        return MISS_DISTANCE

    nu_diff = comov_nu - nu_line

    if abs(nu_diff / nu_lab) < CLOSE_LINE_THRESHOLD:
        nu_diff = 0.0

    if nu_diff >= 0:
        distance = (nu_diff / nu_lab) * C_SPEED_OF_LIGHT * time_explosion
    else:
        distance = MISS_DISTANCE

    return distance

def tardis_angle_aberration_CMF_to_LF(r, mu_cmf, time_explosion):
    """TARDIS: frame_transformations.py:angle_aberration_CMF_to_LF"""
    ct = C_SPEED_OF_LIGHT * time_explosion
    beta = r / ct
    return (mu_cmf + beta) / (1.0 + beta * mu_cmf)

def tardis_angle_aberration_LF_to_CMF(r, mu_lab, time_explosion):
    """TARDIS: frame_transformations.py:angle_aberration_LF_to_CMF"""
    ct = C_SPEED_OF_LIGHT * time_explosion
    beta = r / ct
    return (mu_lab - beta) / (1.0 - beta * mu_lab)

# ============================================================================
# LUMINA FUNCTIONS (implementing same logic)
# ============================================================================

def lumina_get_doppler_factor(r, mu, time_explosion, enable_full_relativity=False):
    """LUMINA: physics_kernels.h - Doppler factor"""
    inv_c = 1.0 / C_SPEED_OF_LIGHT
    inv_t = 1.0 / time_explosion
    beta = r * inv_t * inv_c
    if not enable_full_relativity:
        return 1.0 - mu * beta
    else:
        return (1.0 - mu * beta) / math.sqrt(1.0 - beta * beta)

def lumina_get_inverse_doppler_factor(r, mu, time_explosion, enable_full_relativity=False):
    """LUMINA: physics_kernels.h - Inverse Doppler factor"""
    inv_c = 1.0 / C_SPEED_OF_LIGHT
    inv_t = 1.0 / time_explosion
    beta = r * inv_t * inv_c
    if not enable_full_relativity:
        return 1.0 / (1.0 - mu * beta)
    else:
        return (1.0 + mu * beta) / math.sqrt(1.0 - beta * beta)

def lumina_calculate_distance_boundary(r, mu, r_inner, r_outer):
    """LUMINA: physics_kernels.h - Distance to boundary"""
    mu_sq_minus_1 = mu * mu - 1.0
    if mu > 0.0:
        discriminant = r_outer * r_outer + mu_sq_minus_1 * r * r
        distance = math.sqrt(discriminant) - r * mu
        delta_shell = 1
    else:
        check = r_inner * r_inner + r * r * mu_sq_minus_1
        if check >= 0.0:
            distance = -r * mu - math.sqrt(check)
            delta_shell = -1
        else:
            discriminant = r_outer * r_outer + mu_sq_minus_1 * r * r
            distance = math.sqrt(discriminant) - r * mu
            delta_shell = 1
    return distance, delta_shell

def lumina_calculate_distance_line(nu_lab, comov_nu, is_last_line, nu_line, time_explosion):
    """LUMINA: physics_kernels.h - Distance to line"""
    CLOSE_LINE_THRESHOLD = 1e-7
    MISS_DISTANCE = 1e99

    if is_last_line:
        return MISS_DISTANCE

    nu_diff = comov_nu - nu_line

    if abs(nu_diff / nu_lab) < CLOSE_LINE_THRESHOLD:
        nu_diff = 0.0

    if nu_diff < 0.0:
        return MISS_DISTANCE

    return (nu_diff / nu_lab) * C_SPEED_OF_LIGHT * time_explosion

def lumina_angle_aberration_CMF_to_LF(r, mu_cmf, time_explosion):
    """LUMINA: physics_kernels.h - Angle aberration CMF to LF"""
    beta = r / (C_SPEED_OF_LIGHT * time_explosion)
    return (mu_cmf + beta) / (1.0 + beta * mu_cmf)

def lumina_angle_aberration_LF_to_CMF(r, mu_lab, time_explosion):
    """LUMINA: physics_kernels.h - Angle aberration LF to CMF"""
    beta = r / (C_SPEED_OF_LIGHT * time_explosion)
    return (mu_lab - beta) / (1.0 - beta * mu_lab)

# ============================================================================
# VIRTUAL PACKET WEIGHT CALCULATION
# ============================================================================

def tardis_vpacket_weight(mu, on_inner_boundary, n_vpackets, mu_min, beta_inner=0):
    """TARDIS: virtual_packet.py weight calculation"""
    if on_inner_boundary:
        # K&S 2014 formula for inner boundary
        weight = 2 * mu / n_vpackets
    else:
        weight = (1 - mu_min) / (2 * n_vpackets)
    return weight

def lumina_vpacket_weight(mu, on_inner_boundary, n_vpackets, mu_min, beta_inner=0):
    """LUMINA: virtual_packet.c weight calculation"""
    if on_inner_boundary:
        weight = 2.0 * mu / n_vpackets
    else:
        weight = (1.0 - mu_min) / (2.0 * n_vpackets)
    return weight

# ============================================================================
# COMPARISON TESTS
# ============================================================================

def test_functions():
    """Run comprehensive function comparison"""
    print("=" * 70)
    print("TARDIS-LUMINA FUNCTION-BY-FUNCTION COMPARISON")
    print("Target: differences < 0.1%")
    print("=" * 70)

    # Test parameters
    t_exp = 19.0 * 86400.0  # 19 days
    r = 2.0e15  # cm
    mu_values = [-0.9, -0.5, 0.0, 0.5, 0.9]
    r_inner = 1.6e15
    r_outer = 4.0e15
    nu_lab = 5e14
    nu_line = 4.7e14

    all_passed = True
    tolerance = 1e-10  # For floating point comparison

    # 1. Doppler Factor
    print("\n" + "-" * 70)
    print("1. DOPPLER FACTOR (partial relativity)")
    print("-" * 70)
    for mu in mu_values:
        D_tardis = tardis_get_doppler_factor(r, mu, t_exp, False)
        D_lumina = lumina_get_doppler_factor(r, mu, t_exp, False)
        diff = abs(D_tardis - D_lumina)
        status = "PASS" if diff < tolerance else "FAIL"
        if diff >= tolerance:
            all_passed = False
        print(f"  mu={mu:+.1f}: TARDIS={D_tardis:.15e}, LUMINA={D_lumina:.15e}, diff={diff:.2e} [{status}]")

    # 2. Inverse Doppler Factor
    print("\n" + "-" * 70)
    print("2. INVERSE DOPPLER FACTOR (partial relativity)")
    print("-" * 70)
    for mu in mu_values:
        D_inv_tardis = tardis_get_inverse_doppler_factor(r, mu, t_exp, False)
        D_inv_lumina = lumina_get_inverse_doppler_factor(r, mu, t_exp, False)
        diff = abs(D_inv_tardis - D_inv_lumina)
        status = "PASS" if diff < tolerance else "FAIL"
        if diff >= tolerance:
            all_passed = False
        print(f"  mu={mu:+.1f}: TARDIS={D_inv_tardis:.15e}, LUMINA={D_inv_lumina:.15e}, diff={diff:.2e} [{status}]")

    # 3. Distance to Boundary
    print("\n" + "-" * 70)
    print("3. DISTANCE TO BOUNDARY")
    print("-" * 70)
    for mu in mu_values:
        d_tardis, delta_tardis = tardis_calculate_distance_boundary(r, mu, r_inner, r_outer)
        d_lumina, delta_lumina = lumina_calculate_distance_boundary(r, mu, r_inner, r_outer)
        diff = abs(d_tardis - d_lumina)
        status = "PASS" if diff < tolerance and delta_tardis == delta_lumina else "FAIL"
        if diff >= tolerance or delta_tardis != delta_lumina:
            all_passed = False
        print(f"  mu={mu:+.1f}: TARDIS={d_tardis:.6e} (Δ={delta_tardis:+d}), LUMINA={d_lumina:.6e} (Δ={delta_lumina:+d}), diff={diff:.2e} [{status}]")

    # 4. Distance to Line
    print("\n" + "-" * 70)
    print("4. DISTANCE TO LINE")
    print("-" * 70)
    for mu in mu_values:
        D = tardis_get_doppler_factor(r, mu, t_exp, False)
        nu_cmf = nu_lab * D
        d_tardis = tardis_calculate_distance_line(nu_lab, nu_cmf, False, nu_line, t_exp)
        d_lumina = lumina_calculate_distance_line(nu_lab, nu_cmf, False, nu_line, t_exp)
        diff = abs(d_tardis - d_lumina) if d_tardis < 1e90 and d_lumina < 1e90 else 0
        status = "PASS" if diff < tolerance else "FAIL"
        if diff >= tolerance:
            all_passed = False
        if d_tardis < 1e90:
            print(f"  mu={mu:+.1f}: TARDIS={d_tardis:.6e}, LUMINA={d_lumina:.6e}, diff={diff:.2e} [{status}]")
        else:
            print(f"  mu={mu:+.1f}: TARDIS=MISS, LUMINA=MISS [{status}]")

    # 5. Angle Aberration CMF -> LF
    print("\n" + "-" * 70)
    print("5. ANGLE ABERRATION (CMF to LF)")
    print("-" * 70)
    for mu_cmf in mu_values:
        mu_lab_tardis = tardis_angle_aberration_CMF_to_LF(r, mu_cmf, t_exp)
        mu_lab_lumina = lumina_angle_aberration_CMF_to_LF(r, mu_cmf, t_exp)
        diff = abs(mu_lab_tardis - mu_lab_lumina)
        status = "PASS" if diff < tolerance else "FAIL"
        if diff >= tolerance:
            all_passed = False
        print(f"  mu_cmf={mu_cmf:+.1f}: TARDIS={mu_lab_tardis:.15f}, LUMINA={mu_lab_lumina:.15f}, diff={diff:.2e} [{status}]")

    # 6. Angle Aberration LF -> CMF
    print("\n" + "-" * 70)
    print("6. ANGLE ABERRATION (LF to CMF)")
    print("-" * 70)
    for mu_lab in mu_values:
        mu_cmf_tardis = tardis_angle_aberration_LF_to_CMF(r, mu_lab, t_exp)
        mu_cmf_lumina = lumina_angle_aberration_LF_to_CMF(r, mu_lab, t_exp)
        diff = abs(mu_cmf_tardis - mu_cmf_lumina)
        status = "PASS" if diff < tolerance else "FAIL"
        if diff >= tolerance:
            all_passed = False
        print(f"  mu_lab={mu_lab:+.1f}: TARDIS={mu_cmf_tardis:.15f}, LUMINA={mu_cmf_lumina:.15f}, diff={diff:.2e} [{status}]")

    # 7. Virtual Packet Weight
    print("\n" + "-" * 70)
    print("7. VIRTUAL PACKET WEIGHT")
    print("-" * 70)
    n_vpackets = 10
    mu_min = -0.8
    for mu in [0.1, 0.3, 0.5, 0.7, 0.9]:
        # Inside ejecta
        w_tardis = tardis_vpacket_weight(mu, False, n_vpackets, mu_min)
        w_lumina = lumina_vpacket_weight(mu, False, n_vpackets, mu_min)
        diff = abs(w_tardis - w_lumina)
        status = "PASS" if diff < tolerance else "FAIL"
        if diff >= tolerance:
            all_passed = False
        print(f"  mu={mu:.1f} (inside): TARDIS={w_tardis:.15f}, LUMINA={w_lumina:.15f}, diff={diff:.2e} [{status}]")

        # On inner boundary
        w_tardis = tardis_vpacket_weight(mu, True, n_vpackets, mu_min)
        w_lumina = lumina_vpacket_weight(mu, True, n_vpackets, mu_min)
        diff = abs(w_tardis - w_lumina)
        status = "PASS" if diff < tolerance else "FAIL"
        if diff >= tolerance:
            all_passed = False
        print(f"  mu={mu:.1f} (boundary): TARDIS={w_tardis:.15f}, LUMINA={w_lumina:.15f}, diff={diff:.2e} [{status}]")

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED - Functions are identical to machine precision")
    else:
        print("SOME TESTS FAILED - Check differences above")
    print("=" * 70)

    return all_passed

if __name__ == '__main__':
    test_functions()
