# LUMINA-SN Documentation

**LUMINA-SN**: A High-Performance Monte Carlo Radiative Transfer Code for Type Ia Supernova Spectral Synthesis

---

## Table of Contents

1. [Overview](#overview)
2. [Physical Foundations](#physical-foundations)
3. [Architecture](#architecture)
4. [File Reference with Physics](#file-reference-with-physics)
   - [simulation_state.h / simulation_state.c](#simulation_stateh--simulation_statec)
   - [plasma_physics.h / plasma_physics.c](#plasma_physicsh--plasma_physicsc)
   - [rpacket.h / rpacket.c](#rpacketh--rpacketc)
   - [physics_kernels.h](#physics_kernelsh)
   - [atomic_loader.c / atomic_data.h](#atomic_loaderc--atomic_datah)
   - [lumina_rotation.h / lumina_rotation.c](#lumina_rotationh--lumina_rotationc)
   - [validation.h / validation.c](#validationh--validationc)
   - [test_integrated.c / test_transport.c](#test_integratedc--test_transportc)
5. [Build Instructions](#build-instructions)
6. [Physical Constants](#physical-constants)

---

## Overview

LUMINA-SN is a Monte Carlo radiative transfer code designed for spectral synthesis of Type Ia supernovae. The code solves the radiative transfer equation in spherically symmetric, homologously expanding ejecta using the Monte Carlo method with the Sobolev approximation for line interactions.

### Key Physical Assumptions

1. **Homologous Expansion**: v(r) = r/t (velocity proportional to radius)
2. **Spherical Symmetry**: 1D radial structure with angle-averaged transport
3. **Sobolev Approximation**: Line interactions localized where ν_cmf = ν_line
4. **LTE Ionization**: Saha-Boltzmann equilibrium with NLTE dilution corrections
5. **Sharp Photosphere**: Inner boundary acts as blackbody source

---

## Physical Foundations

### The Radiative Transfer Problem

In an expanding supernova atmosphere, photons emitted from the photosphere travel through layers of ionized gas, interacting with:

1. **Bound-bound transitions** (spectral lines): Resonant scattering/absorption
2. **Free electrons** (Thomson scattering): Elastic scattering
3. **Bound-free transitions** (photoionization): Continuum absorption
4. **Free-free transitions** (bremsstrahlung): Continuum absorption/emission

### The Sobolev Approximation

In rapidly expanding media (v >> v_thermal), the Doppler shift across a resonance region is much larger than the thermal line width. This allows treating each line interaction as occurring at a single point where:

$$\nu_{\text{cmf}} = \nu_{\text{line}}$$

The Sobolev optical depth is:

$$\tau_{\text{Sobolev}} = \frac{\pi e^2}{m_e c} f_{lu} \lambda n_l t_{\text{exp}}$$

where f_lu is the oscillator strength, n_l is the lower level population, and t_exp is the expansion time.

### Monte Carlo Method

Energy packets represent bundles of photons. Each packet is traced through the ejecta:
1. Sample random optical depth: τ_event = -ln(ξ)
2. Find next interaction (boundary, line, or electron scattering)
3. Process interaction (scatter, absorb, escape)
4. Repeat until packet escapes or is absorbed

---

## Architecture

### Data Flow

```
┌─────────────────┐
│   Atomic Data   │  HDF5 (Kurucz/CHIANTI)
└────────┬────────┘
         │ atomic_loader.c
         ▼
┌─────────────────┐
│  Saha-Boltzmann │  Ionization fractions, level populations
│  (plasma_physics.c)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Sobolev τ      │  Line opacities per shell (simulation_state.c)
│  Continuum κ    │  Bound-free, free-free
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Monte Carlo    │  Packet transport (rpacket.c + physics_kernels.h)
│  Transport      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LUMINA Rotation│  Multi-angle spectrum (lumina_rotation.c)
└─────────────────┘
```

---

## File Reference with Physics

---

### simulation_state.h / simulation_state.c

**Purpose:** Pre-compute and manage plasma state and line opacities for all shells.

#### Key Data Structures

| Structure | Physical Meaning |
|-----------|------------------|
| `PhysicsOverrides` | Configuration for thermalization, continuum opacity, NLTE dilution |
| `ShellState` | Single radial shell: geometry, plasma state, line opacities |
| `SimulationState` | Complete model: all shells, atomic data reference |
| `ActiveLine` | Pre-computed Sobolev τ for efficient transport |

#### Functions with Physics

---

##### `calculate_tau_sobolev()`

**Physics:** Sobolev optical depth for resonant line scattering

$$\tau_{\text{Sob}} = \frac{\pi e^2}{m_e c} f_{lu} \lambda n_l t_{\text{exp}} \times \text{OPACITY\_SCALE}$$

**Implementation:**
- Uses oscillator strength f_lu from atomic data
- n_l from Saha-Boltzmann level populations
- t_exp gives velocity gradient (homologous expansion)
- OPACITY_SCALE (0.05) softens profile to prevent flat-bottomed saturation
- Capped at TAU_MAX_CAP (1000) for numerical stability

**Physical Interpretation:** τ >> 1 means optically thick (photon absorbed); τ << 1 means optically thin (photon passes through).

---

##### `calculate_tau_sobolev_A()`

**Physics:** Alternative formulation using Einstein A coefficient

$$\tau_{\text{Sob}} = \frac{\lambda^3}{8\pi} A_{ul} \frac{g_u}{g_l} n_l t_{\text{exp}} \left(1 - e^{-h\nu/kT}\right)^{-1}$$

**Implementation:**
- Uses spontaneous emission coefficient A_ul
- Includes stimulated emission correction factor
- Equivalent to oscillator strength form via Einstein relations

---

##### `calculate_bf_opacity()`

**Physics:** Bound-free (photoionization) opacity using Kramers' formula

$$\kappa_{\text{bf}} = \sigma_0 \frac{Z^4}{n^5} \left(\frac{\nu_n}{\nu}\right)^3 g_{\text{bf}} n_{\text{ion}}$$

where σ_0 = 7.906×10⁻¹⁸ cm² is the hydrogen ground state cross-section.

**Physical Interpretation:** Photons with ν > ν_threshold can ionize atoms. Cross-section falls off as ν⁻³ above threshold.

---

##### `calculate_ff_opacity()`

**Physics:** Free-free (bremsstrahlung) opacity

$$\kappa_{\text{ff}} = 3.7 \times 10^8 \frac{g_{\text{ff}}}{T^{1/2}} \frac{Z^2 n_e n_{\text{ion}}}{\nu^3} \left(1 - e^{-h\nu/kT}\right)$$

**Physical Interpretation:** Free electrons decelerate in ion Coulomb fields, emitting/absorbing radiation. Dominant in hot, ionized plasma.

---

##### `calculate_dilution_factor()`

**Physics:** Geometric dilution of radiation field in expanding atmosphere

$$W = \frac{1}{2} \left[1 - \sqrt{1 - \left(\frac{R_{\text{ph}}}{r}\right)^2}\right]$$

**Physical Interpretation:**
- At photosphere (r = R_ph): W = 0.5 (hemisphere illuminated)
- Far from photosphere (r >> R_ph): W → (R_ph/2r)² → 0
- Used for NLTE correction: T_rad,eff = W^0.25 × T_rad

---

##### `apply_dilution_to_temperature()`

**Physics:** NLTE correction to ionization balance

The radiation field in outer layers is diluted compared to LTE. The effective radiation temperature driving ionization is reduced:

$$T_{\text{rad,eff}} = W^{1/4} \times T_{\text{rad}}$$

This shifts ionization toward lower stages in outer shells.

---

##### `simulation_set_stratified_abundances()`

**Physics:** Velocity-dependent composition (Task Order #32)

Type Ia SNe have stratified composition from nuclear burning:
- **Inner layers** (v < 11,000 km/s): Si, S, Ca, Fe from incomplete Si burning
- **Outer layers** (v > 11,000 km/s): C, O from unburned progenitor

**Implementation:**
- X_Si = 35% for v < 11,000 km/s
- Linear taper: X_Si → 2% by v = 25,000 km/s
- Prevents Si II "photospheric wall" causing blue-shifted absorption

---

##### `find_lines_in_window()` / `frequency_index_find_start()`

**Physics:** Efficient line search using frequency binning

With 270,000+ lines, linear search is too slow. Uses:
1. Logarithmic frequency bins for O(1) starting position
2. Binary search within bins
3. Lines sorted by frequency for monotonic CMF frequency evolution

---

### plasma_physics.h / plasma_physics.c

**Purpose:** Solve ionization equilibrium and level populations using Saha-Boltzmann equations.

#### Functions with Physics

---

##### `calculate_partition_function()`

**Physics:** Statistical mechanical partition function

$$U(T) = \sum_i g_i \exp\left(-\frac{E_i}{kT}\right)$$

**Physical Interpretation:**
- Sum over all bound states of an ion
- g_i = statistical weight (2J+1)
- E_i = excitation energy above ground state
- High-energy levels (E > 50kT) contribute negligibly (Boltzmann suppression)

---

##### `calculate_saha_factor()`

**Physics:** Saha ionization equilibrium ratio

$$\Phi_{i,i+1} = \frac{n_{i+1} n_e}{n_i} = \frac{2 U_{i+1}}{U_i} \left(\frac{2\pi m_e kT}{h^2}\right)^{3/2} \exp\left(-\frac{\chi_i}{kT}\right)$$

**Physical Interpretation:**
- Balances ionization (photon absorption) vs recombination
- Higher T → more ionization (exponential factor)
- Higher n_e → more recombination (pushes equilibrium to lower ionization)

**Implementation:**
- SAHA_CONST = 2.4146868×10¹⁵ cm⁻³ K^(-3/2)
- χ_i = ionization potential from atomic data

---

##### `solve_ionization_balance()`

**Physics:** Newton-Raphson solution for charge neutrality

$$n_e = \sum_Z \sum_{i=0}^{Z} i \cdot n_{Z,i}$$

**Algorithm:**
1. Guess initial n_e
2. Calculate ion fractions using Saha chain
3. Sum electron contributions from all ions
4. Iterate until n_e converges (charge neutrality satisfied)

---

##### `solve_ionization_balance_diluted()`

**Physics:** NLTE ionization with dilution factor

In diluted radiation field, use effective temperature:

$$T_{\text{eff}} = W^{1/4} T$$

This reduces ionization in outer layers where W < 0.5.

---

##### `calculate_level_population_fraction()`

**Physics:** Boltzmann distribution for level populations

$$\frac{n_i}{n_{\text{ion}}} = \frac{g_i \exp(-E_i/kT)}{U(T)}$$

**Physical Interpretation:** In LTE, level populations follow Boltzmann statistics. This determines the number density of absorbers for each spectral line.

---

##### `abundances_set_type_ia_w7()`

**Physics:** W7 deflagration model composition (Nomoto et al. 1984)

Standard Type Ia composition from 1D deflagration:
- Core: ⁵⁶Ni → ⁵⁶Co → ⁵⁶Fe (powers light curve)
- Middle: Si, S, Ca (incomplete burning)
- Outer: C, O (unburned)

---

### rpacket.h / rpacket.c

**Purpose:** Core Monte Carlo packet transport.

#### Key Data Structures

| Structure | Physical Meaning |
|-----------|------------------|
| `RPacket` | Energy packet: position (r), direction (μ), frequency (ν), energy |
| `Estimators` | MC estimators for mean intensity J and radiation temperature |

#### Functions with Physics

---

##### `rpacket_init()`

**Physics:** Initialize energy packet at photosphere

**Implementation:**
- Position r just above photospheric radius
- Direction μ sampled from μ = √ξ (limb darkening for thermal emission)
- Frequency ν sampled from Planck distribution at T_boundary
- Energy ε = L_bol / N_packets (equal energy packets)

---

##### `move_r_packet()`

**Physics:** Geometric propagation in spherical coordinates

$$r_{\text{new}}^2 = r^2 + d^2 + 2rd\mu$$
$$\mu_{\text{new}} = \frac{r\mu + d}{r_{\text{new}}}$$

**Physical Interpretation:**
- Packet moves in straight line (no gravity, no refraction)
- Both r and μ change along trajectory in spherical coordinates
- Updates estimators with path length contribution

---

##### `thomson_scatter()`

**Physics:** Elastic electron scattering

In comoving frame:
- Scattering is isotropic: μ_cmf,new = 2ξ - 1
- Frequency unchanged: ν_cmf conserved

Transform back to lab frame:
- Direction: μ_lab = (μ_cmf + β)/(1 + βμ_cmf) (relativistic aberration)
- Frequency: ν_lab = ν_cmf × D_inv (Doppler shift from new direction)

**Cross-section:** σ_T = 6.65×10⁻²⁵ cm² (classical electron radius)

---

##### `line_scatter()`

**Physics:** Resonant line scattering (Sobolev approximation)

1. Photon absorbed by atom at resonance (ν_cmf = ν_line)
2. Atom re-emits photon isotropically in CMF
3. For coherent scattering: ν_cmf,new = ν_line

**Modes:**
- `LINE_SCATTER`: Pure resonant scattering (ν conserved in CMF)
- `LINE_DOWNBRANCH`: Fluorescence to lower frequency (simplified cascade)
- `LINE_MACROATOM`: Full NLTE macro-atom treatment

**Transform to lab frame:** Same as Thomson scattering

---

##### `trace_packet()`

**Physics:** Find next interaction point (CORE OF MC TRANSPORT)

**Algorithm:**
1. Sample random optical depth: τ_event = -ln(ξ)
2. Calculate distance to shell boundaries (geometric)
3. Calculate distance to electron scattering: d_e where ∫κ_e ds = τ_event
4. Loop through lines (sorted by frequency):
   - Distance to resonance: d_line where ν_cmf(d) = ν_line
   - Accumulate τ_Sobolev
   - If τ_total > τ_event → line interaction
   - **EARLY EXIT:** if d_line > min(d_boundary, d_electron), stop

**Sobolev Insight:** Each line is a "wall" at fixed CMF frequency. Packet either crosses (τ < τ_event) or interacts (τ > τ_event).

---

##### `single_packet_loop()`

**Physics:** Main MC driver

```
while (packet in process):
    trace_packet() → find next interaction
    move_r_packet() → propagate to interaction point
    if boundary:
        cross_shell() or escape/absorb
    elif line:
        line_scatter()
    elif electron:
        thomson_scatter()
```

---

##### `set_estimators()`

**Physics:** Monte Carlo estimators for radiative quantities

**J estimator** (mean intensity):
$$J \approx \frac{1}{4\pi V} \sum_{\text{packets}} \epsilon \cdot d$$

**ν̄ estimator** (mean frequency):
$$\bar{\nu} \approx \frac{\sum \epsilon \cdot d \cdot \nu}{\sum \epsilon \cdot d}$$

Used for temperature iteration in NLTE calculations.

---

### physics_kernels.h

**Purpose:** Exact numerical implementations of frame transformations and distance calculations.

#### Functions with Physics

---

##### `get_doppler_factor()`

**Physics:** Transform frequency Lab → Comoving frame

$$D = 1 - \beta\mu \quad \text{(partial relativity)}$$
$$D = \frac{1 - \beta\mu}{\sqrt{1-\beta^2}} \quad \text{(full relativity)}$$

where β = v/c = r/(ct_exp) and μ = cos θ.

**Physical Interpretation:**
- μ > 0 (outward): D < 1, redshift in CMF
- μ < 0 (inward): D > 1, blueshift in CMF

---

##### `get_inverse_doppler_factor()`

**Physics:** Transform frequency Comoving → Lab frame

$$D_{\text{inv}} = \frac{1}{1 - \beta\mu} \quad \text{(partial)}$$
$$D_{\text{inv}} = \frac{1 + \beta\mu}{\sqrt{1-\beta^2}} \quad \text{(full)}$$

**Note:** Full relativistic formula has + sign (not simply 1/D).

---

##### `angle_aberration_CMF_to_LF()`

**Physics:** Relativistic aberration of direction

$$\mu_{\text{lab}} = \frac{\mu_{\text{cmf}} + \beta}{1 + \beta \mu_{\text{cmf}}}$$

**Physical Interpretation:** Due to relative motion, directions appear different in different frames. Photons are "beamed" in the direction of motion.

---

##### `calculate_distance_boundary()`

**Physics:** Geometric ray-sphere intersection

Solve |r + d·μ̂|² = R² for distance d:

$$d^2 + 2rd\mu + (r^2 - R^2) = 0$$

$$d = -r\mu \pm \sqrt{R^2 - r^2(1-\mu^2)}$$

**Sign selection:**
- Outward (μ > 0): Always hits outer boundary, use + root
- Inward (μ < 0): Check if ray passes inside inner boundary

---

##### `calculate_distance_line()`

**Physics:** Distance to Sobolev resonance

As packet propagates, its CMF frequency changes due to Doppler shift. Find distance d where:

$$\nu_{\text{cmf}}(d) = \nu_{\text{line}}$$

Using ν_cmf = ν_lab × D(r,μ) and homologous expansion.

**Result:**

$$d_{\text{line}} = \frac{ct_{\text{exp}}(\nu_{\text{cmf}} - \nu_{\text{line}})}{\nu_{\text{line}}}$$

(with corrections for geometry)

---

##### `calculate_distance_electron()`

**Physics:** Distance to Thomson scattering event

Sample from exponential distribution with optical depth:

$$\tau_e = n_e \sigma_T d$$

Given τ_event = -ln(ξ):

$$d_e = \frac{\tau_{\text{event}}}{n_e \sigma_T}$$

---

### atomic_loader.c / atomic_data.h

**Purpose:** Load atomic physics data from HDF5 format.

#### Functions with Physics

---

##### `atomic_data_load_hdf5()`

**Physics:** Load TARDIS-format atomic database

Contains:
- Element data (Z, mass)
- Ion data (ionization energies χ)
- Level data (energies E, statistical weights g)
- Line data (wavelengths λ, oscillator strengths f, Einstein coefficients A)

---

##### `inject_si_ii_6355_lines()`

**Physics:** Add Si II 6347/6371 Å doublet

Critical diagnostic lines for SN Ia:
- Si II 6347.10 Å: f_lu = 0.708 (strong)
- Si II 6371.37 Å: f_lu = 0.419

These lines form the characteristic "Si II 6355" absorption feature used to classify Type Ia supernovae and measure photospheric velocities.

---

### lumina_rotation.h / lumina_rotation.c

**Purpose:** Post-processing multi-angle spectrum synthesis.

#### Functions with Physics

---

##### `lumina_rotate_packet()`

**Physics:** Transform escaped packet to observer frame

**Time delay:**
$$t_{\text{obs}} = t_{\text{exp}} - \frac{r \cdot \mu_{\text{packet}}}{c}$$

Packets escaping toward observer arrive earlier.

**Solid angle weighting:**
$$w = \left(\frac{D_{\text{observer}}}{D_{\text{packet}}}\right)^2$$

Accounts for Doppler beaming and isotropic emission in CMF.

---

##### `spectrum_add_packet()`

**Physics:** Bin packet contribution to spectrum

$$F_\lambda = \frac{\sum_{\text{packets}} \epsilon \cdot w}{\Delta\lambda \cdot N_{\text{total}}}$$

---

### validation.h / validation.c

**Purpose:** Validate C implementation against Python/TARDIS reference.

#### Functions

- `validation_trace_record()`: Record packet state at each step
- `validation_compare_traces()`: Compare trajectories for 10⁻¹⁰ accuracy
- `single_packet_loop_traced()`: Transport with full state recording

---

### test_integrated.c / test_transport.c

**Purpose:** Driver programs for simulation and validation.

#### Pipeline (test_integrated.c)

1. Load atomic data from HDF5
2. Configure shell geometry (v_inner to v_outer)
3. Set stratified abundances (Task Order #32)
4. Compute Saha-Boltzmann ionization
5. Calculate Sobolev opacities for all lines
6. Run Monte Carlo transport (OpenMP parallel)
7. Apply LUMINA rotation for observer spectrum
8. Write spectrum to CSV

---

## Build Instructions

```bash
# Prerequisites: GCC, HDF5, OpenMP (optional)
make clean
make

# Run simulation
./test_integrated atomic/kurucz_cd23_chianti_H_He.h5 100000 spectrum.csv

# With custom parameters
./test_integrated --v-inner 10500 --T 13500 atomic/kurucz_cd23_chianti_H_He.h5 500000 output.csv
```

---

## Physical Constants

| Constant | Symbol | Value | Units |
|----------|--------|-------|-------|
| Speed of light | c | 2.99792458×10¹⁰ | cm/s |
| Thomson cross-section | σ_T | 6.6524587158×10⁻²⁵ | cm² |
| Sobolev constant | πe²/m_e c | 2.6540281×10⁻² | cm² s⁻¹ |
| Saha constant | (2πm_e k/h²)^(3/2) | 2.4146868×10¹⁵ | cm⁻³ K^(-3/2) |
| Boltzmann constant | k_B | 1.380649×10⁻¹⁶ | erg/K |
| Planck constant | h | 6.62607015×10⁻²⁷ | erg s |
| Electron mass | m_e | 9.1093837015×10⁻²⁸ | g |
| eV to erg | - | 1.602176634×10⁻¹² | erg/eV |

---

## References

1. Lucy, L.B. (2002, 2003) - Monte Carlo radiative transfer
2. Mazzali, P.A. & Lucy, L.B. (1993) - Sobolev approximation for SNe
3. Nomoto, K. et al. (1984) - W7 deflagration model
4. Kerzendorf, W.E. & Sim, S.A. (2014) - TARDIS code

---

*Documentation generated: 2026-01-30*
*LUMINA-SN with Task Order #30 (Continuum Opacity) and #32 (Si II Velocity Calibration)*
