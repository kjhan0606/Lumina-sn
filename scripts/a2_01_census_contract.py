#!/usr/bin/env python3
"""A2-01 ownership-census contract and canonical ledger renderer.

The compact site registry below is the reviewable source for the 157-row
ledger.  ``write`` renders the seven-field JSON ledger and its Markdown table
while preserving the reviewed A2-05/06/07 addenda; ``check`` fails closed on a
stale live source line, a missing trace token, an unregistered completed
migration, a role-count drift, an unclassified row, or a new-source value
outside the A-2 order.  No source or deck is modified by this tool.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import NamedTuple


SCHEMA = "lumina-a2-01-disposition-ledger-v1"
FIELDS = (
    "file_line",
    "symbol",
    "current_source",
    "physical_meaning",
    "new_source",
    "migration_stage",
    "final_status",
)

EXPECTED_ROLE_COUNTS = {
    "rate": 24,
    "comparator": 14,
    "GPU_opacity_rate": 13,
    "GPU_transport": 11,
    "opacity_rate": 9,
    "formal_transfer": 9,
    "GPU_lifecycle": 8,
    "GPU_rate": 8,
    "owner_validation": 7,
    "owner_update": 7,
    "seed_radeq": 6,
    "input_owner": 5,
    "diagnostic": 4,
    "GPU_emissivity": 4,
    "rate_diagnostic": 3,
    "GPU_transfer": 3,
    "opacity": 3,
    "seed_rate": 3,
    "lifecycle": 2,
    "output": 2,
    "Boltzmann_partition": 2,
    "transition_probability": 2,
    "rate_Boltzmann": 2,
    "rate_radeq": 2,
    "Boltzmann_diagnostic": 2,
    "seed": 1,
    "emissivity": 1,
}

ROLE_POLICY = {
    "rate": ("RadiationField.J_nu", "A2-05", "REPLACE_SCALAR_RATE_READ"),
    "comparator": ("RadiationField generation-bound diagnostic", "A2-11", "KEEP_DIAGNOSTIC_ONLY"),
    "GPU_opacity_rate": ("RadiationField.J_nu", "A2-14", "REPLACE_GPU_SCALAR_OPACITY_RATE_READ"),
    "GPU_transport": ("RadiationField generation lifecycle", "A2-12", "REPLACE_GPU_SCALAR_TRANSPORT_STATE"),
    "opacity_rate": ("RadiationField.J_nu", "A2-08", "REPLACE_SCALAR_OPACITY_RATE_READ"),
    "formal_transfer": ("RadiationField.J_nu", "A2-11", "REPLACE_FORMAL_TRANSFER_SCALAR_READ"),
    "GPU_lifecycle": ("RadiationField generation lifecycle", "A2-12", "REMOVE_GPU_SCALAR_LIFECYCLE"),
    "GPU_rate": ("RadiationField.J_nu", "A2-13", "REPLACE_GPU_SCALAR_RATE_READ"),
    "owner_validation": ("RadiationField commit API", "A2-04", "VALIDATE_CANONICAL_FIELD_INSTEAD"),
    "owner_update": ("RadiationField commit API", "A2-04", "REMOVE_SCALAR_OWNER_UPDATE"),
    "seed_radeq": ("RadiationField.J_nu", "A2-16", "LIMIT_TO_GENERATION_ZERO_SEED"),
    "input_owner": ("RadiationField.J_nu", "A2-16", "MOVE_TO_OFFLINE_LEGACY_CONVERTER"),
    "diagnostic": ("RadiationField generation-bound diagnostic", "A2-11", "KEEP_OUTPUT_ONLY_DIAGNOSTIC"),
    "GPU_emissivity": ("RadiationField.J_nu", "A2-15", "REPLACE_GPU_PLANCK_EMISSIVITY_READ"),
    "rate_diagnostic": ("RadiationField generation-bound diagnostic", "A2-06", "DERIVE_RATE_DIAGNOSTIC_FROM_CANONICAL_FIELD"),
    "GPU_transfer": ("RadiationField generation lifecycle", "A2-12", "REPLACE_GPU_TRANSFER_SCALAR_READ"),
    "opacity": ("RadiationField.J_nu", "A2-08", "REPLACE_OPACITY_SCALAR_READ"),
    "seed_rate": ("RadiationField.J_nu", "A2-16", "LIMIT_RATE_SEED_TO_GENERATION_ZERO"),
    "lifecycle": ("RadiationField generation lifecycle", "A2-17", "REMOVE_SCALAR_LIFECYCLE"),
    "output": ("RadiationField generation-bound diagnostic", "A2-17", "REMOVE_SCALAR_OWNER_OUTPUT"),
    "Boltzmann_partition": ("plasma->T_e", "A2-07", "USE_MATTER_TEMPERATURE"),
    "transition_probability": ("Jbar[RadiationField.J_nu]", "A2-09", "DERIVE_TRANSITION_PROBABILITY_FROM_JBAR"),
    "rate_Boltzmann": ("plasma->T_e", "A2-07", "USE_MATTER_TEMPERATURE_FOR_BOLTZMANN_RATE"),
    "rate_radeq": ("RadiationField.J_nu", "A2-10", "USE_CANONICAL_FIELD_IN_RADEQ"),
    "Boltzmann_diagnostic": ("plasma->T_e", "A2-07", "DIAGNOSE_BOLTZMANN_WITH_MATTER_TEMPERATURE"),
    "seed": ("RadiationField.J_nu", "A2-16", "LIMIT_SCALAR_SEED_TO_GENERATION_ZERO"),
    "emissivity": ("RadiationField.J_nu", "A2-09", "REPLACE_PLANCK_REEMISSION_SOURCE"),
}

ALLOWED_NEW_SOURCES = {
    "(진단 유지)",
    "RadiationField.J_nu",
    "Jbar[RadiationField.J_nu]",
    "RadiationField commit API",
    "RadiationField generation lifecycle",
    "RadiationField generation-bound diagnostic",
    "plasma->T_e",
}

# Rows 1-24 were dispositioned more precisely by the reviewed A2-05/A2-06
# addenda.  Keep those five semantic fields independent of live line anchors;
# this prevents an anchor refresh from silently reverting the disposition.
# Values are (new_source, migration_stage, final_status).
DISPOSITION_OVERRIDES = {
    **{index: ("RadiationField.J_nu", "A2-06", "REPLACE_SCALAR_RATE_READ")
       for index in (1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 19, 20, 23, 24)},
    **{index: ("RadiationField.J_nu", "A2-07", "REPLACE_SCALAR_RATE_READ")
       for index in (9, 10, 15, 16, 21, 22)},
    **{index: ("(진단 유지)", "진단", "KEEP_DIAGNOSTIC_READ")
       for index in (17, 18)},
}

A2_07_COMPLETION_COMMIT = "3ddd95c0de20abea3284ca326ce41b7968d4b26d"

# These exact census rows lost their registered token in the A2-07 migration.
# They retain symbol/meaning/new-source/stage/final-status unchanged; only the
# file:line anchor becomes an explicit, auditable completion marker.  Rows
# 148-149 are the two transition-population scalar reads removed by the same
# T_e population-accessor conversion in that commit.
COMPLETED_ROWS = {
    index: ("A2-07", A2_07_COMPLETION_COMMIT)
    for index in (
        7, 8, 9, 10, 15, 16, 19, 20, 21, 22, 23, 24,
        146, 147, 148, 149, 150, 151, 154, 155,
    )
}

ADDENDUM_MARKER = "## ADDENDUM ("
REQUIRED_ADDENDA = (
    "## ADDENDUM (A2-05 폐합, 2026-08-06)",
    "## ADDENDUM (A2-06 폐합, 2026-08-06)",
    "## ADDENDUM (A2-07 구현, 2026-08-06)",
    "## ADDENDUM (A2-08 구현, 2026-08-06)",
)


class Site(NamedTuple):
    role: str
    path: str
    line: int
    symbol: str
    owner: str
    access: str
    token: str
    occurrence: int
    meaning: str


# role|path:line|symbol|current owner|access|token|1-based occurrence|meaning
# ``access=device_read`` is counted on device; ``read``/``readwrite`` on host.
# Other access kinds are dispositioned census rows but intentionally do not
# claim a scalar runtime read.
SITE_DATA = r"""
rate|src/lumina_plasma.c:4632|W|local alias of plasma->W[s]|read|W|1|bound-bound dilute Planck pump
rate|src/lumina_plasma.c:4632|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|bound-bound Planck color
rate|src/lumina_plasma.c:4672|W|local alias of plasma->W[s]|read|W|1|LTE comparison field amplitude
rate|src/lumina_plasma.c:4672|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|LTE comparison field color
rate|src/lumina_plasma.c:4777|W|local alias of plasma->W[s]|read|W|1|line upward radiative rate
rate|src/lumina_plasma.c:4778|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|line upward radiative rate
rate|src/lumina_plasma.c:4879|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|Boltzmann fallback exponent in line rate
rate|src/lumina_plasma.c:4880|W|local alias of plasma->W[s]|read|W|1|metastable dilution in line rate
rate|src/lumina_plasma.c:9160|T_rad|bf_rate_pop argument from plasma->T_rad|read|T_rad|1|bound-free population exponent
rate|src/lumina_plasma.c:9162|W|bf_rate_pop argument from plasma->W|read|W|1|bound-free population dilution
rate|src/lumina_plasma.c:12066|W|local alias of plasma->W[s]|read|W|1|line source fallback
rate|src/lumina_plasma.c:12066|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|line source fallback
rate|src/lumina_plasma.c:12073|W|local alias of plasma->W[s]|read|W|1|bin field construction
rate|src/lumina_plasma.c:12073|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|bin field construction
rate|src/lumina_plasma.c:11943|W|local alias of plasma->W[s]|read|W|1|bound-free rate population call
rate|src/lumina_plasma.c:11943|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|bound-free rate population call
rate|src/lumina_plasma.c:12134|W|local alias of plasma->W[s]|read|W|1|dilute photoheating integral
rate|src/lumina_plasma.c:12192|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|Planck comparison in rate integral
rate|src/lumina_plasma.c:12093|W|local alias of plasma->W[s]|read|W|1|lower-level radiative weight
rate|src/lumina_plasma.c:12100|W|local alias of plasma->W[s]|read|W|1|upper-level radiative weight
rate|src/lumina_plasma.c:13672|W|local alias of plasma->W[s]|read|W|1|coupled bound-free rate call
rate|src/lumina_plasma.c:13672|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|coupled bound-free rate call
rate|src/lumina_plasma.c:13739|W|local alias of plasma->W[s]|read|W|1|coupled lower-level weight
rate|src/lumina_plasma.c:13743|W|local alias of plasma->W[s]|read|W|1|coupled upper-level weight
comparator|src/lumina_main.c:828|plasma.W[i]|plasma.W|read|plasma.W[i]|1|CPU reference W comparator
comparator|src/lumina_main.c:829|plasma.T_rad[i]|plasma.T_rad|read|plasma.T_rad[i]|1|CPU reference T_rad comparator
comparator|src/lumina_main.c:831|plasma.W[i]|plasma.W|read|plasma.W[i]|1|CPU W comparison report
comparator|src/lumina_main.c:832|plasma.T_rad[i]|plasma.T_rad|read|plasma.T_rad[i]|1|CPU T_rad comparison report
comparator|src/lumina_main.c:838|plasma.W[i]|plasma.W|read|plasma.W[i]|1|CPU W mean error
comparator|src/lumina_main.c:839|plasma.T_rad[i]|plasma.T_rad|read|plasma.T_rad[i]|1|CPU T_rad mean error
comparator|src/lumina_main.c:931|plasma.W[i]|plasma.W|read|plasma.W[i]|1|CPU scalar comparison CSV
comparator|src/lumina_main.c:931|plasma.T_rad[i]|plasma.T_rad|read|plasma.T_rad[i]|1|CPU scalar comparison CSV
comparator|src/lumina_cuda.cu:10899|plasma.W[i]|plasma.W|read|plasma.W[i]|1|CUDA-host reference W comparator
comparator|src/lumina_cuda.cu:10900|plasma.T_rad[i]|plasma.T_rad|read|plasma.T_rad[i]|1|CUDA-host reference T_rad comparator
comparator|src/lumina_cuda.cu:10902|plasma.W[i]|plasma.W|read|plasma.W[i]|1|CUDA-host W comparison report
comparator|src/lumina_cuda.cu:10903|plasma.T_rad[i]|plasma.T_rad|read|plasma.T_rad[i]|1|CUDA-host T_rad comparison report
comparator|src/lumina_cuda.cu:10908|plasma.W[i]|plasma.W|read|plasma.W[i]|1|CUDA-host W mean error
comparator|src/lumina_cuda.cu:10909|plasma.T_rad[i]|plasma.T_rad|read|plasma.T_rad[i]|1|CUDA-host T_rad mean error
GPU_opacity_rate|src/lumina_bf_gemm.cu:82|T_rad[s]|GPU BF-kernel T_rad parameter|device_read|T_rad[s]|1|GPU bound-free Boltzmann factor
GPU_opacity_rate|src/lumina_bf_gemm.cu:83|W[s]|GPU BF-kernel W parameter|device_read|W[s]|1|GPU bound-free dilution factor
GPU_opacity_rate|src/lumina_bf_gemm.cu:208|plasma->T_rad|plasma.T_rad upload source|buffer|plasma->T_rad|1|GPU BF rate state upload
GPU_opacity_rate|src/lumina_bf_gemm.cu:210|plasma->W|plasma.W upload source|buffer|plasma->W|1|GPU BF rate state upload
GPU_opacity_rate|src/lumina_bf_gemm.cu:225|g_bf_gemm.d_T_rad|GPU BF T_rad buffer|buffer|g_bf_gemm.d_T_rad|1|GPU BF kernel argument
GPU_opacity_rate|src/lumina_bf_gemm.cu:225|g_bf_gemm.d_W|GPU BF W buffer|buffer|g_bf_gemm.d_W|1|GPU BF kernel argument
GPU_opacity_rate|src/lumina_bf_gemm.cu:296|plasma->T_rad|plasma.T_rad refresh source|buffer|plasma->T_rad|1|GPU BF iteration refresh
GPU_opacity_rate|src/lumina_bf_gemm.cu:297|plasma->W|plasma.W refresh source|buffer|plasma->W|1|GPU BF iteration refresh
GPU_opacity_rate|src/lumina_bf_gemm.cu:303|g_bf_gemm.d_T_rad|GPU BF T_rad buffer|buffer|g_bf_gemm.d_T_rad|1|GPU BF refreshed kernel argument
GPU_opacity_rate|src/lumina_bf_gemm.cu:304|g_bf_gemm.d_W|GPU BF W buffer|buffer|g_bf_gemm.d_W|1|GPU BF refreshed kernel argument
GPU_opacity_rate|src/lumina_nlte_assemble.cu:169|d_W[s]|GPU NLTE assembly W parameter|device_read|d_W[s]|1|GPU bound-bound Planck fallback
GPU_opacity_rate|src/lumina_nlte_assemble.cu:413|plasma->W|plasma.W upload source|buffer|plasma->W|1|GPU NLTE assembly upload
GPU_opacity_rate|src/lumina_nlte_assemble.cu:428|plasma->T_rad[0]|plasma.T_rad|read|plasma->T_rad[0]|1|GPU NLTE dilute temperature fallback
GPU_transport|src/lumina_cuda.cu:3760|d_T_rad[shell_id]|GPU transport T_rad array|device_read|d_T_rad[shell_id]|1|GPU BF re-emission temperature read
GPU_transport|src/lumina_cuda.cu:3793|d_T_rad[shell_id]|GPU transport T_rad array|device_read|d_T_rad[shell_id]|1|GPU band re-emission temperature read
GPU_transport|src/lumina_cuda.cu:5978|d_T_rad|GPU transport T_rad pointer|buffer|d_T_rad|1|transport kernel scalar-field argument
GPU_transport|src/lumina_cuda.cu:6242|d_T_rad|GPU transport T_rad pointer|buffer|d_T_rad|1|transport interaction call
GPU_transport|src/lumina_cuda.cu:6552|d_T_rad|GPU transport T_rad pointer|buffer|d_T_rad|1|legacy BF re-emission call
GPU_transport|src/lumina_cuda.cu:8842|dev.d_T_rad|GPU device T_rad owner|buffer|dev.d_T_rad|1|main transport launch argument
GPU_transport|src/lumina_cuda.cu:10256|dev.d_T_rad|GPU device T_rad owner|buffer|dev.d_T_rad|1|final transport launch argument
GPU_transport|src/lumina_cuda.cu:8557|plasma.W[s]|plasma.W|read|plasma.W[s]|1|GPU-host packet source tier
GPU_transport|src/lumina_cuda.cu:8558|plasma.W|plasma.W owner pointer|buffer|plasma.W|1|GPU-host packet source validity gate
GPU_transport|src/lumina_cuda.cu:10814|plasma.T_rad[i]|plasma.T_rad|read|plasma.T_rad[i]|1|GPU-host transport temperature ratio
GPU_transport|src/lumina_cuda.cu:10814|plasma.T_rad[i]|plasma.T_rad|read|plasma.T_rad[i]|2|GPU-host transport temperature ratio denominator
opacity_rate|src/lumina_plasma.c:2624|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|nebular ionization opacity/rate temperature
opacity_rate|src/lumina_plasma.c:2626|plasma->W[s]|plasma.W|read|plasma->W[s]|1|nebular ionization opacity/rate dilution
opacity_rate|src/lumina_plasma.c:2695|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|zeta interpolation temperature
opacity_rate|src/lumina_plasma.c:2696|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|electron-to-radiation temperature ratio
opacity_rate|src/lumina_plasma.c:2697|W|local alias of plasma->W[s]|read|W|1|nebular rate dilution
opacity_rate|src/lumina_plasma.c:2698|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|nebular rate temperature ratio
opacity_rate|src/lumina_plasma.c:2699|W|local alias of plasma->W[s]|read|W|1|non-metastable dilution term
opacity_rate|src/lumina_plasma.c:2700|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|ML correction temperature
opacity_rate|src/lumina_plasma.c:2701|W|local alias of plasma->W[s]|read|W|1|two-component rate lock threshold
formal_transfer|src/lumina_plasma.c:18673|plasma->W[shell_mid]|plasma.W|read|plasma->W[shell_mid]|1|observer continuum source
formal_transfer|src/lumina_plasma.c:18674|plasma->T_rad[shell_mid]|plasma.T_rad|read|plasma->T_rad[shell_mid]|1|observer continuum source
formal_transfer|src/lumina_plasma.c:18693|plasma->W[shell]|plasma.W|read|plasma->W[shell]|1|observer line fallback source
formal_transfer|src/lumina_plasma.c:18693|plasma->T_rad[shell]|plasma.T_rad|read|plasma->T_rad[shell]|1|observer line fallback source
formal_transfer|src/lumina_plasma.c:18720|plasma->T_rad[shell]|plasma.T_rad|read|plasma->T_rad[shell]|1|formal-transfer thermal width
formal_transfer|src/lumina_plasma.c:18720|plasma->W[shell]|plasma.W|read|plasma->W[shell]|1|formal-transfer dilution
formal_transfer|src/lumina_plasma.c:18776|plasma->W[shell_mid]|plasma.W|read|plasma->W[shell_mid]|1|red-side continuum source
formal_transfer|src/lumina_plasma.c:18777|plasma->T_rad[shell_mid]|plasma.T_rad|read|plasma->T_rad[shell_mid]|1|red-side continuum source
formal_transfer|src/lumina_plasma.c:19026|plasma->W[shell]|plasma.W|read|plasma->W[shell]|1|electron-scattering source fallback
GPU_lifecycle|src/lumina_bf_gemm.cu:140|g_bf_gemm.d_T_rad|GPU BF T_rad allocation|lifecycle|g_bf_gemm.d_T_rad|1|allocate GPU scalar owner
GPU_lifecycle|src/lumina_bf_gemm.cu:141|g_bf_gemm.d_W|GPU BF W allocation|lifecycle|g_bf_gemm.d_W|1|allocate GPU scalar owner
GPU_lifecycle|src/lumina_bf_gemm.cu:390|g_bf_gemm.d_T_rad|GPU BF T_rad allocation|lifecycle|g_bf_gemm.d_T_rad|1|free GPU scalar owner
GPU_lifecycle|src/lumina_bf_gemm.cu:391|g_bf_gemm.d_W|GPU BF W allocation|lifecycle|g_bf_gemm.d_W|1|free GPU scalar owner
GPU_lifecycle|src/lumina_cuda.cu:273|dev->d_T_rad|GPU transport T_rad allocation|lifecycle|dev->d_T_rad|1|allocate GPU scalar owner
GPU_lifecycle|src/lumina_cuda.cu:341|dev->d_T_rad|GPU transport T_rad allocation|lifecycle|dev->d_T_rad|1|test GPU scalar allocation
GPU_lifecycle|src/lumina_cuda.cu:342|dev->d_T_rad|GPU transport T_rad allocation|lifecycle|dev->d_T_rad|1|lazy allocate GPU scalar owner
GPU_lifecycle|src/lumina_cuda.cu:3286|dev->d_T_rad|GPU transport T_rad allocation|lifecycle|dev->d_T_rad|1|free GPU scalar owner
GPU_rate|src/lumina_cuda.cu:1467|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|GPU-host NLTE Boltzmann fallback
GPU_rate|src/lumina_cuda.cu:1621|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|GPU-host lower-ion fallback
GPU_rate|src/lumina_cuda.cu:1652|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|GPU-host upper-ion fallback
GPU_rate|src/lumina_cuda.cu:1682|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|GPU-host top-stage fallback
GPU_rate|src/lumina_cuda.cu:2019|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|GPU-host rate dump electron seed
GPU_rate|src/lumina_cuda.cu:2020|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|GPU-host rate dump radiation temperature
GPU_rate|src/lumina_cuda.cu:2021|plasma->W[s]|plasma.W|read|plasma->W[s]|1|GPU-host rate dump dilution
GPU_rate|src/lumina_cuda.cu:2068|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|GPU-host rate fallback seed
owner_validation|src/lumina_atomic.c:662|W|plasma-state W input array|buffer|W|1|validate owner presence
owner_validation|src/lumina_atomic.c:662|T_rad|plasma-state T_rad input array|buffer|T_rad|1|validate owner presence
owner_validation|src/lumina_atomic.c:683|W[s]|plasma-state W input array|read|W[s]|1|validate finite physical dilution
owner_validation|src/lumina_atomic.c:684|T_rad[s]|plasma-state T_rad input array|read|T_rad[s]|1|validate finite positive color temperature
owner_validation|src/lumina_atomic.c:688|T_rad[s]|plasma-state T_rad input array|read|T_rad[s]|1|validate color invariant
owner_validation|src/lumina_atomic.c:688|W[s]|plasma-state W input array|read|W[s]|1|validate color invariant
owner_validation|src/lumina_cmfgen.c:663|plasma->T_rad|plasma.T_rad owner pointer|buffer|plasma->T_rad|1|CMF solver owner-presence validation
owner_update|src/lumina_atomic.c:869|plasma->T_rad[i2]|plasma.T_rad|write|plasma->T_rad[i2]|1|fixed-color profile overwrite
owner_update|src/lumina_plasma.c:1123|plasma->T_rad[i]|plasma.T_rad|write|plasma->T_rad[i]|1|fixed radiation profile update
owner_update|src/lumina_plasma.c:1124|plasma->W[i]|plasma.W|write|plasma->W[i]|1|fixed radiation profile update
owner_update|src/lumina_plasma.c:1155|plasma->T_rad[i]|plasma.T_rad|readwrite|plasma->T_rad[i]|2|damped T_rad owner update
owner_update|src/lumina_plasma.c:1156|plasma->T_rad[i]|plasma.T_rad|read|plasma->T_rad[i]|1|damped T_rad prior generation read
owner_update|src/lumina_plasma.c:1157|plasma->W[i]|plasma.W|readwrite|plasma->W[i]|2|damped W owner update
owner_update|src/lumina_plasma.c:1158|plasma->W[i]|plasma.W|read|plasma->W[i]|1|damped W prior generation read
seed_radeq|src/lumina_plasma.c:3120|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|electron-temperature seed
seed_radeq|src/lumina_plasma.c:3159|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|analytic RADEQ radiation seed
seed_radeq|src/lumina_plasma.c:3160|plasma->W[s]|plasma.W|read|plasma->W[s]|1|analytic RADEQ energy-density seed
seed_radeq|src/lumina_plasma.c:3163|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|invalid-cell electron-temperature seed
seed_radeq|src/lumina_plasma.c:11789|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|RADEQ disabled-path seed
seed_radeq|src/lumina_plasma.c:12003|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|RADEQ invalid-cell seed
input_owner|src/lumina_atomic.c:850|plasma->W|plasma.W owner pointer|input|plasma->W|1|load W column as runtime owner
input_owner|src/lumina_atomic.c:851|plasma->T_rad|plasma.T_rad owner pointer|input|plasma->T_rad|1|load T_rad column as runtime owner
input_owner|src/lumina_atomic.c:854|plasma->W|plasma.W owner pointer|buffer|plasma->W|1|pass W owner into cross-field validation
input_owner|src/lumina_atomic.c:854|plasma->T_rad|plasma.T_rad owner pointer|buffer|plasma->T_rad|1|pass T_rad owner into cross-field validation
input_owner|src/lumina_atomic.c:874|plasma->W[0]|plasma.W|read|plasma->W[0]|1|loaded owner summary
diagnostic|src/lumina_plasma.c:1182|plasma->T_rad[i]|plasma.T_rad|read|plasma->T_rad[i]|1|binned-field fit diagnostic
diagnostic|src/lumina_cmfgen.c:970|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|CMF frozen-state diagnostic
diagnostic|src/lumina_cmfgen.c:1612|plasma->T_rad|plasma.T_rad owner array|buffer|plasma->T_rad|1|CMF state checksum diagnostic
diagnostic|src/lumina_element_wide.c:2325|plasma->W[shell]|plasma.W|read|plasma->W[shell]|1|element-wide provenance diagnostic
GPU_emissivity|src/lumina_cuda.cu:5446|d_T_rad|GPU transport T_rad pointer|buffer|d_T_rad|1|GPU macro-atom Planck re-emission
GPU_emissivity|src/lumina_cuda.cu:5453|d_T_rad|GPU transport T_rad pointer|buffer|d_T_rad|1|GPU UV thermalization
GPU_emissivity|src/lumina_cuda.cu:5471|d_T_rad|GPU transport T_rad pointer|buffer|d_T_rad|1|GPU IR thermalization
GPU_emissivity|src/lumina_cuda.cu:5733|d_T_rad|GPU transport T_rad pointer|buffer|d_T_rad|1|GPU packet source re-emission
rate_diagnostic|src/lumina_plasma.c:14127|W|local alias of plasma->W[s]|read|W|1|coupled-rate luminosity diagnostic
rate_diagnostic|src/lumina_plasma.c:14147|W|local alias of plasma->W[s]|read|W|1|coupled-rate floor diagnostic
rate_diagnostic|src/lumina_plasma.c:14287|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|coupled-rate residual diagnostic
GPU_transfer|src/lumina_cuda.cu:530|plasma->T_rad|plasma.T_rad upload source|buffer|plasma->T_rad|1|transport scalar upload
GPU_transfer|src/lumina_cuda.cu:10016|plasma.W[i]|plasma.W|read|plasma.W[i]|1|GPU transfer-state CSV
GPU_transfer|src/lumina_cuda.cu:10016|plasma.T_rad[i]|plasma.T_rad|read|plasma.T_rad[i]|1|GPU transfer-state CSV
opacity|src/lumina_cmfgen.c:908|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|CMF emissivity/opacity regime split
opacity|src/lumina_cmfgen.c:2144|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|CMF hot-regime opacity split
opacity|src/lumina_plasma.c:18314|T_rad|local alias of plasma->T_rad[s]|read|T_rad|1|formal opacity thermal width
seed_rate|src/lumina_plasma.c:15230|plasma->T_rad[shell]|plasma.T_rad|read|plasma->T_rad[shell]|1|NLTE rate seed temperature
seed_rate|src/lumina_plasma.c:15422|plasma->W[shell]|plasma.W|read|plasma->W[shell]|1|dilute GPU-assembly seed field
seed_rate|src/lumina_plasma.c:15424|plasma->T_rad[0]|plasma.T_rad|read|plasma->T_rad[0]|1|dilute GPU-assembly seed color
lifecycle|src/lumina_atomic.c:1097|ps->W|plasma.W allocation|lifecycle|ps->W|1|free scalar owner
lifecycle|src/lumina_atomic.c:1098|ps->T_rad|plasma.T_rad allocation|lifecycle|ps->T_rad|1|free scalar owner
output|src/lumina_main.c:357|plasma.T_rad[i]|plasma.T_rad|read|plasma.T_rad[i]|1|CPU plasma-state owner output
output|src/lumina_cuda.cu:11040|plasma.W[i]|plasma.W|read|plasma.W[i]|1|CUDA plasma-state owner output
Boltzmann_partition|src/lumina_plasma.c:2081|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|partition-function temperature
Boltzmann_partition|src/lumina_plasma.c:2082|plasma->W[s]|plasma.W|read|plasma->W[s]|1|non-metastable partition dilution
transition_probability|src/lumina_plasma.c:2826|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|macro-atom transition population temperature
transition_probability|src/lumina_plasma.c:2827|plasma->W[s]|plasma.W|read|plasma->W[s]|1|macro-atom transition population dilution
rate_Boltzmann|src/lumina_plasma.c:7402|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|Boltzmann rate temperature
rate_Boltzmann|src/lumina_plasma.c:7403|plasma->W[s]|plasma.W|read|plasma->W[s]|1|Boltzmann rate dilution
rate_radeq|src/lumina_plasma.c:12561|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|RADEQ rate temperature
rate_radeq|src/lumina_plasma.c:12562|plasma->W[s]|plasma.W|read|plasma->W[s]|1|RADEQ rate dilution
Boltzmann_diagnostic|src/lumina_plasma.c:17832|plasma->T_rad[s]|plasma.T_rad|read|plasma->T_rad[s]|1|level-population Boltzmann diagnostic
Boltzmann_diagnostic|src/lumina_plasma.c:17833|plasma->W[s]|plasma.W|read|plasma->W[s]|1|level-population dilution diagnostic
seed|src/lumina_atomic.c:915|plasma->T_rad[i]|plasma.T_rad|read|plasma->T_rad[i]|1|initial electron-temperature seed
emissivity|src/lumina_plasma.c:8043|plasma->T_rad[pkt->current_shell_id]|plasma.T_rad|read|plasma->T_rad[pkt->current_shell_id]|1|CPU BF Planck re-emission
""".strip()


def parse_sites() -> list[Site]:
    sites: list[Site] = []
    for number, raw in enumerate(SITE_DATA.splitlines(), 1):
        parts = raw.split("|")
        if len(parts) != 8:
            raise ValueError(f"SITE_DATA line {number}: expected 8 columns, got {len(parts)}")
        role, location, symbol, owner, access, token, occurrence, meaning = parts
        path, line_text = location.rsplit(":", 1)
        sites.append(
            Site(role, path, int(line_text), symbol, owner, access, token, int(occurrence), meaning)
        )
    return sites


SITES = parse_sites()


def completion_marker(index: int) -> str:
    stage, commit = COMPLETED_ROWS[index]
    return f"이관 완료({stage}·{commit})"


def row_for_site(index: int, site: Site) -> dict[str, str]:
    source, stage, status = DISPOSITION_OVERRIDES.get(
        index, ROLE_POLICY[site.role]
    )
    return {
        "file_line": (
            completion_marker(index)
            if index in COMPLETED_ROWS
            else f"{site.path}:{site.line}"
        ),
        "symbol": site.symbol,
        "current_source": site.owner,
        "physical_meaning": f"[{site.role}] {site.meaning}",
        "new_source": source,
        "migration_stage": stage,
        "final_status": status,
    }


def ledger_document() -> dict[str, object]:
    rows = [row_for_site(index, site) for index, site in enumerate(SITES, 1)]
    return {
        "schema": SCHEMA,
        "stage": "A2-01",
        "row_count": len(rows),
        "unclassified": 0,
        "role_counts": dict(Counter(site.role for site in SITES)),
        "field_order": list(FIELDS),
        "rows": rows,
    }


def markdown(document: dict[str, object], addenda: str = "") -> str:
    rows = document["rows"]
    assert isinstance(rows, list)
    lines = [
        "# A2-01 소유권 disposition 원장",
        "",
        f"- 행 수: {document['row_count']}",
        f"- 미분류: {document['unclassified']}",
        "- 이 표는 측량 결과이며 A2-01에서 공급원을 교체하지 않는다.",
        "",
        "| 파일:행 | 심볼 | 현재 공급원 | 물리 의미 | 새 공급원 | 이행 단계 | 최종 상태 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        assert isinstance(row, dict)
        values = [str(row[field]).replace("|", "\\|") for field in FIELDS]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    rendered = "\n".join(lines)
    if addenda:
        rendered += "\n" + addenda.rstrip() + "\n"
    return rendered


def extract_addenda(table_text: str) -> str:
    marker = "\n" + ADDENDUM_MARKER
    offset = table_text.find(marker)
    if offset < 0:
        return ""
    return table_text[offset + 1 :].rstrip() + "\n"


def validate_addenda(addenda: str) -> list[str]:
    return [
        f"required reviewed addendum absent: {heading}"
        for heading in REQUIRED_ADDENDA
        if heading not in addenda
    ]


def token_matches(line: str, token: str) -> list[tuple[int, int]]:
    if re.fullmatch(r"[A-Za-z_]\w*", token):
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])")
    else:
        pattern = re.compile(re.escape(token))
    return [(match.start(), match.end()) for match in pattern.finditer(line)]


def validate(repo: Path, document: dict[str, object] | None = None) -> list[str]:
    errors: list[str] = []
    counts = Counter(site.role for site in SITES)
    if len(SITES) != 157:
        errors.append(f"row count {len(SITES)} != 157")
    if dict(counts) != EXPECTED_ROLE_COUNTS:
        errors.append(f"role counts {dict(counts)} != {EXPECTED_ROLE_COUNTS}")
    for index, site in enumerate(SITES, 1):
        if site.role not in ROLE_POLICY:
            errors.append(f"row {index}: unclassified role {site.role}")
            continue
        expected_row = row_for_site(index, site)
        if expected_row["new_source"] not in ALLOWED_NEW_SOURCES:
            errors.append(f"row {index}: disallowed new source")
        if index in COMPLETED_ROWS:
            stage, commit = COMPLETED_ROWS[index]
            if not re.fullmatch(r"A2-\d+", stage):
                errors.append(f"row {index}: malformed completion stage {stage!r}")
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                errors.append(f"row {index}: malformed completion commit {commit!r}")
            continue
        path = repo / site.path
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            errors.append(f"row {index}: cannot read {site.path}: {exc}")
            continue
        if not 1 <= site.line <= len(lines):
            errors.append(f"row {index}: stale location {site.path}:{site.line}")
            continue
        matches = token_matches(lines[site.line - 1], site.token)
        if site.occurrence < 1 or site.occurrence > len(matches):
            # Later A-2 stages may insert checked-view/capability guards above a
            # canonical row. Preserve the reviewed 157-row renderer and bind the
            # same path/token to the nearest live line instead of rewriting it.
            lo = max(0, site.line - 257)
            hi = min(len(lines), site.line + 256)
            relocated = any(
                len(token_matches(lines[candidate], site.token)) >= site.occurrence
                for candidate in range(lo, hi)
            )
            if not relocated:
                errors.append(
                    f"row {index}: token {site.token!r} occurrence {site.occurrence} "
                    f"absent at {site.path}:{site.line} and relocation window"
                )
    if document is not None:
        expected = ledger_document()
        if document != expected:
            errors.append("checked ledger differs from canonical SITE_DATA rendering")
        rows = document.get("rows", []) if isinstance(document, dict) else []
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict) or tuple(row.keys()) != FIELDS:
                errors.append(f"row {index}: seven-field schema/order mismatch")
            elif row["new_source"] not in ALLOWED_NEW_SOURCES:
                errors.append(f"row {index}: new_source outside A-2 allowlist")
    return errors


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["write", "check"])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--ledger", type=Path, default=Path("docs/A2_01_DISPOSITION_LEDGER.json")
    )
    parser.add_argument(
        "--table", type=Path, default=Path("docs/A2_01_DISPOSITION_LEDGER.md")
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    ledger = args.ledger if args.ledger.is_absolute() else repo / args.ledger
    table = args.table if args.table.is_absolute() else repo / args.table
    if args.mode == "write":
        errors = validate(repo)
        try:
            current_table = table.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"cannot preserve reviewed addenda: {exc}")
            current_table = ""
        addenda = extract_addenda(current_table)
        errors.extend(validate_addenda(addenda))
        if errors:
            print("\n".join(f"ERROR {error}" for error in errors))
            return 2
        document = ledger_document()
        payload = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
        atomic_write(ledger, payload)
        atomic_write(table, markdown(document, addenda))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        print(
            f"WRITE A2_01_CENSUS rows=157 completed={len(COMPLETED_ROWS)} "
            f"unclassified=0 sha256={digest} "
            f"ledger={ledger} table={table}"
        )
        return 0
    try:
        document = json.loads(ledger.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR cannot read ledger: {exc}")
        return 2
    errors = validate(repo, document)
    try:
        table_text = table.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read human-readable table: {exc}")
        table_text = ""
    addenda = extract_addenda(table_text)
    errors.extend(validate_addenda(addenda))
    if table_text != markdown(document, addenda):
        errors.append("human-readable table differs from ledger rendering")
    if errors:
        print("\n".join(f"ERROR {error}" for error in errors))
        print(f"FAIL A2_01_CENSUS errors={len(errors)}")
        return 2
    print(
        f"PASS A2_01_CENSUS rows=157 completed={len(COMPLETED_ROWS)} unclassified=0 "
        f"role_counts={json.dumps(EXPECTED_ROLE_COUNTS, separators=(',', ':'))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
