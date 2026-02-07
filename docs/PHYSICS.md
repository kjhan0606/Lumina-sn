# LUMINA-SN Physics Documentation

## 1. Monte Carlo Radiative Transfer Overview

LUMINA-SN uses the Monte Carlo (MC) method to solve the radiative transfer problem
in Type Ia supernova ejecta. The radiation field is represented by a large number
of discrete photon packets, each carrying equal energy:

    E_packet = L_inner / n_packets

where L_inner is the luminosity at the inner boundary (photosphere) and n_packets
is the total number of packets launched per iteration.

Packets are initialized at the inner boundary with frequencies sampled from a
blackbody distribution at temperature T_inner. They propagate outward through a
1D spherically-symmetric ejecta grid undergoing homologous expansion, where the
velocity is proportional to radius:

    v(r) = r / t_exp

with t_exp being the time since explosion. This linear velocity law is an
excellent approximation for SN Ia ejecta at the epochs of interest (days to
weeks post-explosion), because the ejecta have entered free expansion where
pressure forces are negligible.

As packets traverse the ejecta, they interact with:

- **Spectral lines** via the Sobolev approximation (dominant opacity source)
- **Free electrons** via Thomson scattering (frequency-independent)

Bound-free and free-free continuum opacities are negligible in SN Ia ejecta at
optical wavelengths, because the typical photon energies (1.5-2.5 eV) are far
below the ionization thresholds of the dominant ions (Si II: 16.35 eV,
Fe II: 16.19 eV, Ca II: 11.87 eV).

Packets that reach the outer boundary escape and contribute to the emergent
spectrum. Packets that fall back through the inner boundary are reabsorbed by the
photosphere. The simulation iterates over multiple cycles, updating the radiation
field estimates (W, T_rad) and plasma state (ionization, level populations) until
convergence.


## 2. Sobolev Approximation and Line Interaction

### Comoving-Frame Frequency Evolution

In homologous expansion, a photon packet traveling along a ray experiences a
linearly changing comoving-frame (CMF) frequency:

    nu_cmf(s) = nu_lab * (1 - (r0 * mu0 + s) / (c * t_exp))

where s is the distance traveled along the ray, r0 is the starting radius, mu0
is the cosine of the angle with the radial direction, and c is the speed of
light. This linear dependence means that as a packet crosses a shell, it sweeps
through a continuous range of CMF frequencies, potentially resonating with
multiple spectral lines.

### Sobolev Optical Depth

The Sobolev approximation treats each line as a point interaction. The optical
depth for a line transition from lower level l to upper level u is:

    tau_Sobolev = (pi * e^2 / (m_e * c)) * f_lu * n_l * lambda_lu * t_exp

where:
- e is the electron charge
- m_e is the electron mass
- f_lu is the oscillator strength
- n_l is the number density of the lower level
- lambda_lu is the rest wavelength of the transition
- t_exp is the time since explosion

The stimulated emission correction modifies the effective lower-level population:

    n_l_eff = n_l * (1 - g_l * n_u / (g_u * n_l))

where g_l, g_u are the statistical weights of the lower and upper levels, and
n_u is the upper level population.

### Full Frequency Sweep

LUMINA-SN performs a full Sobolev sweep across each shell, computing the CMF
frequency at entry (nu_entry) and exit (nu_exit) and considering ALL lines in
that frequency range. This is critical because the frequency shift across a
thick shell can be much larger than a fixed percentage window. An earlier
implementation used a +/-1% frequency window, which caused packets to miss
lines entirely during shell crossings.

### Cumulative Optical Depth Sampling

The interaction point is determined by sampling a random optical depth:

    tau_event = -ln(xi)

where xi is a uniform random number in (0, 1). The code then accumulates
tau_line contributions from each line encountered along the packet path. When
the cumulative optical depth exceeds tau_event, the packet interacts with the
current line. The interaction probability for a single line is:

    P_interact = 1 - exp(-tau_Sobolev)


## 3. Line Interaction Types

When a packet interacts with a line, three outcomes are possible depending on
the simulation configuration:

### Resonance Scattering

The simplest treatment: the packet is re-emitted at the same comoving-frame
frequency as the absorbed line. A new propagation direction is chosen
isotropically. In the lab frame, the re-emitted frequency differs due to the
Doppler shift from the new direction. This produces the classical P-Cygni
profile shape but does not redistribute photons across different transitions.

### Downbranching (Fluorescence)

After absorption into the upper level, the packet is re-emitted via a different
transition selected from a pre-computed table of branching probabilities. This
allows photons absorbed at one wavelength to be re-emitted at a completely
different wavelength. Downbranching is the primary mechanism that redistributes
UV photons into the optical band and is crucial for producing the correct
depth of absorption features such as Si II 6355 A.

### Macro-Atom (Full Treatment)

The most physically complete treatment, where the packet activates a macro-atom
level and undergoes a chain of internal transitions before eventually being
re-emitted. This is described in detail in the next section.


## 4. Macro-Atom Model

### Theoretical Basis

The macro-atom formalism, developed by Lucy (2002, 2003), provides an exact
treatment of line emission within the Sobolev approximation. Rather than
treating each line interaction independently, it considers the atom as a
complete system with interconnected energy levels.

### Activation and Internal Transitions

When a packet is absorbed by a line transition (l -> u), it activates the
upper level u of the macro-atom. The packet's identity is temporarily
suspended; instead, the macro-atom holds an amount of energy equal to the
packet energy.

At each activated level i, the macro-atom chooses one of three actions:

1. **Radiative emission**: the atom de-excites via a bound-bound transition
   (i -> j, where j < i), emitting the packet at the corresponding line
   frequency. The probability is proportional to the Einstein A coefficient
   times the Sobolev escape probability for that transition.

2. **Internal transition (up or down)**: the atom transitions to a different
   level without emitting a packet. The transition probability depends on
   radiative rates and the local radiation field (stimulated absorption/emission).

3. **Continuum process**: the atom ionizes or recombines, potentially leading
   to a k-packet (thermal pool) that is re-emitted at a frequency sampled
   from the local thermal distribution.

### Branching Probabilities

The transition probabilities at each level are pre-computed from atomic data
(Einstein coefficients, energy levels, statistical weights) and the local
radiation field parameters (W, T_rad). In LUMINA-SN, these probabilities are
frozen at their initial values for stability. Updating them during the
simulation can cause UV divergence in outer shells where the radiation field
estimates have large statistical noise.

The branching probability tables are stored as cumulative probability arrays.
For a given activated level, a uniform random number is drawn and a binary
search selects the transition. The emission line frequency is looked up from
the destination transition.

### Physical Importance

The macro-atom treatment is essential for producing realistic SN Ia spectra
because:

- It allows efficient downbranching from UV to optical wavelengths. UV photons
  absorbed by iron-group elements can cascade through multiple levels and
  emerge in the optical, contributing to the pseudo-continuum.

- It correctly handles fluorescence, where absorption at one wavelength leads
  to emission at a completely different wavelength through intermediate level
  transitions.

- It produces the correct depth of key absorption features. For Si II 6355 A,
  the macro-atom treatment increased the absorption depth from 45% to over 74%
  (compared to pure resonance scattering), much closer to the TARDIS reference
  of 93%.

### Safety Limit

A maximum of 500 internal transitions is imposed per macro-atom activation.
If this limit is reached, the packet is emitted from the current level via
the most probable radiative transition. This prevents rare infinite loops in
the level network.


## 5. Convergence: W, T_rad, and T_inner

### Monte Carlo Estimators

During packet propagation, two volume-based estimators are accumulated for
each radial shell:

- **j_estimator**: proportional to the mean intensity J, accumulated as
  E_packet * path_length / V_shell for each packet traversal

- **nu_bar_estimator**: frequency-weighted mean intensity, accumulated as
  E_packet * nu * path_length / V_shell

These estimators are summed over all packets in an iteration and normalized
at the end.

### Radiation Temperature

The radiation temperature is derived from the ratio of the two estimators:

    T_rad = (H_PLANCK / (4 * K_BOLTZMANN)) * (nu_bar / j)

This comes from the first moment of the Planck function. For a dilute
blackbody radiation field J_nu = W * B_nu(T_rad), the ratio nu_bar/j
depends only on T_rad (not on W), providing a clean temperature diagnostic.

### Dilution Factor

The dilution factor W quantifies how much the radiation field is diluted
relative to a full blackbody at temperature T_rad:

    W = j / (4 * SIGMA_SB * T_rad^4 * t_sim * V_shell)

where SIGMA_SB is the Stefan-Boltzmann constant, t_sim is the simulation
time, and V_shell is the volume of the shell. In the optically thin limit
near the photosphere, W approaches 0.5 (geometric dilution). It decreases
outward as the radiation field becomes more dilute.

### Damped Updates

All radiation field quantities are updated using 50% damping to prevent
oscillations:

    X_new = X_old + 0.5 * (X_estimated - X_old)

This is the TARDIS convention (damping_constant = 0.5). Without damping,
the estimates can oscillate between iterations, especially in optically
thin outer shells with poor photon statistics.

### Inner Boundary Temperature

The inner boundary temperature T_inner controls the luminosity injected at
the photosphere. It is updated to drive the emergent luminosity toward the
requested luminosity:

    T_inner_new = T_inner * (L_emitted / L_requested)^(-0.5)

Key aspects of this update:

- **L_emitted** is computed from the actual energies of escaped packets
  (summing E_packet for all packets that reach the outer boundary), not
  from an escape fraction approximation.

- The **-0.5 exponent** (TARDIS convention) accounts for the non-linear
  coupling between T_inner and L_emitted. Since L ~ T^4 but the opacity
  and escape fraction also depend on T, the effective sensitivity is
  steeper than the naive T^4 scaling.

- **hold_iterations = 3**: the first 3 iterations run without updating
  T_inner, allowing the radiation field estimates to stabilize before
  adjusting the boundary condition.

- T_inner is also damped at 50% to prevent overshoot.


## 6. Plasma Solver

The plasma solver computes the ionization state and level populations in
each shell, given the radiation field parameters (W, T_rad) and the
elemental composition.

### Partition Functions

The partition function for each ion is computed as a Boltzmann sum over
all known energy levels:

    Z = sum_i g_i * exp(-E_i / (k_B * T_rad))

where g_i is the statistical weight and E_i is the excitation energy of
level i. Importantly, T_rad (not the electron temperature T_e) is used
for ALL levels, following the TARDIS convention. Using T_e for metastable
levels was identified as a source of systematic error in early versions.

### Ionization Equilibrium

Ion populations are determined by the nebular ionization equation with
dilution factors (Mazzali & Lucy 1993):

    n_{i+1} * n_e / n_i = W * Phi(T_rad) * zeta(T_rad) * delta

where:
- Phi(T_rad) is the Saha factor (from partition functions and ionization energy)
- zeta(T_rad) is the recombination correction factor (tabulated)
- delta accounts for the departure from LTE

The zeta factors encode the ratio of recombination rates to photoionization
rates and are critical for getting the correct ionization balance in the
dilute radiation field of the SN ejecta.

### Electron Density Iteration

The electron density n_e is solved iteratively:

1. Start with an initial guess for n_e
2. Compute ion populations using the Saha equation
3. Sum contributions from all ions: n_e_new = sum(Z_i * n_i)
4. Apply 50% damping: n_e = n_e_old + 0.5 * (n_e_new - n_e_old)
5. Repeat until convergence (5% relative change threshold)

The 50% damping and 5% threshold follow the TARDIS convention. Earlier
versions used undamped iteration with a 1e-6 threshold, which could
oscillate or converge to incorrect values.

### Sobolev Optical Depth Update

After updating the plasma state, tau_Sobolev is recomputed for all active
lines using the new W, T_rad, n_e, and ion populations. This feeds back
into the next MC iteration.

Note: updating the macro-atom transition probabilities from the new tau
values can cause divergence in outer shells due to statistical noise.
LUMINA-SN keeps the transition probabilities frozen at their initial values
by default. This can be overridden with the LUMINA_TAU_UPDATE environment
variable.


## 7. Spectrum Modes

LUMINA-SN supports three methods for constructing the emergent spectrum,
each with different trade-offs between noise and computational cost.

### Real Packet Spectrum

The most straightforward method: packets that escape through the outer
boundary are binned by their lab-frame wavelength into a histogram.

- Wavelength range: 500 - 20000 Angstroms
- Number of bins: 1000
- Bin width: 19.5 Angstroms

The luminosity in each bin is:

    L_lambda = (number of packets in bin) * E_packet / (delta_lambda * t_sim)

This spectrum has Poisson noise proportional to 1/sqrt(N_bin), where N_bin
is the number of packets per bin. For a typical run with 2M-20M packets,
the noise is acceptable in the optical but can be significant at UV and
IR wavelengths where fewer packets escape.

### Virtual Packet Spectrum (GPU only)

At each line interaction point, a set of virtual (probe) packets is emitted
in random directions and traced to the outer boundary. Each virtual packet
accumulates optical depth along its path and contributes to the spectrum
with a weight:

    w = exp(-tau_total)

where tau_total is the integrated optical depth from the interaction point
to the outer boundary. This dramatically reduces the noise compared to
real packets because every interaction contributes to the spectrum, not
just the fraction that happens to escape.

Virtual packets do not affect the radiation field estimators or the
simulation state. They are purely diagnostic.

### Rotation Packet Spectrum

For modeling the effects of ejecta rotation or asymmetry, LUMINA-SN can
store the radius r and direction cosine mu of each escaping packet. These
parameters allow post-processing with arbitrary Doppler weighting functions:

    nu_obs = nu_lab * (1 + v_rot * sin(theta) / c)

where v_rot is the rotational velocity and theta is the viewing angle.
This avoids re-running the full MC simulation for each viewing angle.


## References

- Lucy, L.B. (2002). "Monte Carlo techniques for time-dependent radiative
  transfer in homologous flows." A&A, 384, 725-735.

- Lucy, L.B. (2003). "Monte Carlo transition probabilities." A&A, 403,
  261-275.

- Mazzali, P.A. & Lucy, L.B. (1993). "The application of Monte Carlo
  methods to the synthesis of early-time supernovae spectra." A&A, 279,
  447-456.

- Kerzendorf, W.E. & Sim, S.A. (2014). "A spectral synthesis code for
  rapid modelling of supernovae." MNRAS, 440, 387-404. (TARDIS)

- Long, K.S. & Knigge, C. (2002). "Modeling the spectral signatures of
  accretion disk winds: a new Monte Carlo approach." ApJ, 579, 725-740.
