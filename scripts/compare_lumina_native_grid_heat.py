#!/usr/bin/env python3
"""같은 고정 J·물질에서 Lumina 열원을 원래/native 격자에 나란히 적분한다.

같은 source producer·같은 population·같은 fine-grid J를 유지해 fine/native
격자 효과를 분리하고, 실제 EDDFACTOR native J로도 재적분해 native STEQ_T와
비교한다. 격자 source 적분은 SCL_LN 비적용 진단으로 표시하고, native SCL_LN을
포함한 보존 열장부를 별도로 native STEQ_T와 비교한다. 수송, SE/RE 해법,
물질 갱신이나 합격선은 추가하지 않는다.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np

import cmfgen_model_sources as S
from extract_cmfgen_line_net_fixture import CMFGEN_RE_INTERNAL_TO_CGS
from prepare_cmfgen_fixedj_heat import read_heat_row


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_native_heat_controls(path):
    text = Path(path).read_text()
    controls = {}
    for name in ('COH_ES', 'GLOBAL_LINE', 'SCL_LN', 'SCL_LN_FAC'):
        values = re.findall(r'^\s*(\S+)\s+\['+name+r'\]', text, re.M | re.I)
        if len(values) != 1:
            raise ValueError('native VADAT control missing/duplicated: '+name)
        value = values[0].strip("'\" ").upper()
        if name == 'SCL_LN_FAC':
            try:
                value = float(value)
            except ValueError as error:
                raise ValueError('native SCL_LN_FAC is not numeric') from error
            if not math.isfinite(value) or value < 0:
                raise ValueError('native SCL_LN_FAC is invalid')
        controls[name] = value
    return controls


def parse_native_cont_freq_text(text):
    """Read only the NCF-row integration block; later diagnostic rows are not grid."""
    match = re.search(r'^\s*!NCF,\s*(\d+)\s*$', text, re.M)
    if not match:
        raise ValueError('CONT_FREQ의 NCF 선언 누락')
    ncf = int(match.group(1))
    if ncf < 2:
        raise ValueError('CONT_FREQ 격자 길이 오류')
    rows = []
    for line in text[match.end():].splitlines():
        fields = line.split()
        if not fields or fields[0].startswith('!'):
            continue
        try:
            index = int(fields[0])
        except ValueError:
            if len(rows) == ncf:
                break
            continue
        if len(rows) == ncf:
            break
        if len(fields) < 4 or index != len(rows) + 1:
            raise ValueError('CONT_FREQ의 native NCF 행 주소/열 오류')
        try:
            rows.append((float(fields[1]), float(fields[2])))
        except ValueError as error:
            raise ValueError('CONT_FREQ 주파수 숫자 오류') from error
    if len(rows) != ncf:
        raise ValueError('CONT_FREQ native NCF 행 일부 누락')
    descending = np.asarray(rows, dtype=np.float64)
    if (not np.isfinite(descending).all() or np.any(descending <= 0) or
            not np.all(np.diff(descending[:, 0]) < 0)):
        raise ValueError('CONT_FREQ의 내림차순 주파수/유한성 오류')
    ascending = descending[::-1] * 1.0e15
    nu, nu_cont = ascending[:, 0], ascending[:, 1]
    if np.any(np.diff(nu) <= 0):
        raise ValueError('native integration frequency is not strictly ascending')
    return nu, nu_cont, dict(ncf=ncf,
        printed_frequency_precision='CONT_FREQ decimal output (8 significant digits); not binary internal NU')


def native_trapuneq_weights(nu_hz):
    """TRAPUNEQ nodal weights, with native frequency units converted to Hz."""
    nu = np.asarray(nu_hz, dtype=np.float64)
    if (nu.ndim != 1 or len(nu) < 2 or not np.isfinite(nu).all() or
            np.any(nu <= 0) or np.any(np.diff(nu) <= 0)):
        raise ValueError('TRAPUNEQ requires a positive ascending frequency grid')
    weights = np.empty_like(nu)
    weights[0] = (nu[1] - nu[0]) * 0.5
    weights[-1] = (nu[-1] - nu[-2]) * 0.5
    weights[1:-1] = (nu[2:] - nu[:-2]) * 0.5
    return weights


def fine_grid_trapezoid_weights(nu_hz):
    """Lumina 원 주파수 적분의 composite-trapezoid nodal weights."""
    nu = np.asarray(nu_hz, dtype=np.float64)
    if (nu.ndim != 1 or len(nu) < 2 or not np.isfinite(nu).all() or
            np.any(nu <= 0) or np.any(np.diff(nu) <= 0)):
        raise ValueError('Lumina fine-grid quadrature requires ascending positive frequencies')
    weights = np.empty_like(nu)
    delta = np.diff(nu)
    weights[0] = delta[0] * 0.5
    weights[-1] = delta[-1] * 0.5
    weights[1:-1] = (nu[2:] - nu[:-2]) * 0.5
    return weights


def read_eddfactor(path, info_path):
    """Map the 14 native headers and return ascending source frequencies and J."""
    info = Path(info_path).read_text().splitlines()
    if len(info) < 3:
        raise ValueError('EDDFACTOR_INFO is truncated')
    fields = info[2].split()
    if (len(fields) < 6 or list(map(int, fields[:5])) != [90, 728, 8, 1, 4] or
            fields[5].upper() != 'T'):
        raise ValueError('EDDFACTOR layout differs from sealed 90-depth little-endian record')
    records = Path(path).stat().st_size // 728
    if Path(path).stat().st_size != records * 728 or records <= 14:
        raise ValueError('EDDFACTOR record length/count invalid')
    nfreq = records - 14
    header = np.memmap(path, dtype='<f8', mode='r', offset=0, shape=(14, 91))
    data = np.memmap(path, dtype='<f8', mode='r', offset=14 * 728, shape=(nfreq, 91))
    try:
        if header[4, 0] != 1.0:
            raise ValueError('EDDFACTOR FINISH record is not complete')
        frequency = np.asarray(data[:, 90], dtype=np.float64) * 1.0e15
        order = np.argsort(frequency)
        frequency = frequency[order]
        j_by_frequency = np.asarray(data[order, :90], dtype=np.float64)
    finally:
        header._mmap.close()
        data._mmap.close()
    if (not np.isfinite(frequency).all() or np.any(frequency <= 0) or
            np.any(np.diff(frequency) <= 0) or not np.isfinite(j_by_frequency).all() or
            np.any(j_by_frequency < 0)):
        raise ValueError('EDDFACTOR frequency/J values invalid')
    return frequency, j_by_frequency


def interpolate_native_j(source_nu, j_by_frequency, target_nu):
    source_nu = np.asarray(source_nu, dtype=np.float64)
    j = np.asarray(j_by_frequency, dtype=np.float64)
    target = np.asarray(target_nu, dtype=np.float64)
    if (source_nu.ndim != 1 or j.ndim != 2 or j.shape[0] != len(source_nu) or
            target.ndim != 1 or np.any(np.diff(source_nu) <= 0) or
            np.any(target <= 0) or not np.isfinite(source_nu).all() or
            not np.isfinite(target).all() or not np.isfinite(j).all() or np.any(j < 0)):
        raise ValueError('RD_CONT_J linear-frequency interpolation input invalid')
    return np.vstack([np.interp(target, source_nu, j[:, depth])
                      for depth in range(j.shape[1])])


def heat_components(source_npz, j, weights, expected_nu):
    names = (('BF', 'bf_chi', 'bf_eta'), ('FF', 'ff_chi', 'ff_eta'),
             ('BB', 'bb_chi', 'bb_eta'))
    result = {}
    gross = np.zeros(j.shape[0], dtype=np.float64)
    with np.load(source_npz, allow_pickle=False) as source:
        if not np.array_equal(source['nu_Hz'], expected_nu):
            raise ValueError('source coefficient frequency array differs from native grid')
        for label, chi_name, eta_name in names:
            chi = np.asarray(source[chi_name], dtype=np.float64)
            eta = np.asarray(source[eta_name], dtype=np.float64)
            if chi.shape != j.shape or eta.shape != j.shape:
                raise ValueError('source coefficient depth/frequency shape mismatch')
            absorbed = chi * j
            local = (absorbed - eta) * weights[None, :]
            result[label] = (4.0 * math.pi) * np.sum(local, axis=1)
            gross += (4.0 * math.pi) * np.sum(
                (np.abs(absorbed) + np.abs(eta)) * weights[None, :], axis=1)
        # Coherent ES is transport opacity, not a thermal source in this VADAT.
        if 'es_chi' not in source.files:
            raise ValueError('source output lacks the separately owned ES opacity')
        es = np.asarray(source['es_chi'], dtype=np.float64)
        if es.shape != j.shape:
            raise ValueError('ES opacity depth/frequency shape mismatch')
        result['total_no_coherent_ES'] = result['BF'] + result['FF'] + result['BB']
    return result, gross


def metrics(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError('heat comparison arrays invalid')
    difference = candidate - reference
    scale = np.maximum(np.abs(reference), np.abs(candidate))
    relative = np.divide(np.abs(difference), scale,
        out=np.zeros_like(difference), where=scale > 0)
    return dict(max_absolute_difference=float(np.max(np.abs(difference))),
        rms_difference=float(np.sqrt(np.mean(difference * difference))),
        sign_mismatch_points=int(np.count_nonzero(reference * candidate < 0)),
        median_symmetric_net_relative_difference=float(np.median(relative)),
        p95_symmetric_net_relative_difference=float(np.quantile(relative, 0.95)))


def field_input(manifest, basename, base=None):
    matches = []
    for raw_path, digest in manifest.get('input_sha256', {}).items():
        path = Path(raw_path)
        if path.name != basename:
            continue
        if not path.is_absolute() and base is not None:
            path = Path(base)/path
        matches.append((path, digest))
    if len(matches) != 1 or sha(matches[0][0]) != matches[0][1]:
        raise ValueError('fixed-J manifest does not identify one unchanged '+basename)
    return matches[0][0], matches[0][1]


def compare_fine_field_to_eddfactor(fine_nu, fine_j, edd_nu, edd_j):
    """Bound the field mismatch without making a second full-size J copy."""
    if (fine_nu.shape != edd_nu.shape or fine_j.shape != (90, len(fine_nu)) or
            edd_j.shape != (len(edd_nu), 90)):
        raise ValueError('fine and EDDFACTOR field dimensions differ')
    max_frequency_relative = float(np.max(np.abs(fine_nu-edd_nu)/fine_nu))
    max_j_absolute = 0.0
    max_j_symmetric_relative = 0.0
    exact_j_values = 0
    total_j_values = fine_j.size
    for first in range(0, len(fine_nu), 32768):
        end = min(first+32768, len(fine_nu))
        left = np.asarray(fine_j[:, first:end])
        right = edd_j[first:end, :].T
        if not np.isfinite(left).all() or np.any(left < 0):
            raise ValueError('fine-grid J contains invalid values')
        difference = np.abs(left-right)
        scale = np.maximum(np.abs(left), np.abs(right))
        relative = np.divide(difference, scale, out=np.zeros_like(difference), where=scale > 0)
        max_j_absolute = max(max_j_absolute, float(np.max(difference)))
        max_j_symmetric_relative = max(max_j_symmetric_relative, float(np.max(relative)))
        exact_j_values += int(np.count_nonzero(left == right))
    return dict(max_frequency_relative_difference=max_frequency_relative,
        max_J_absolute_difference=max_j_absolute,
        max_J_symmetric_relative_difference=max_j_symmetric_relative,
        exact_J_values=exact_j_values, total_J_values=total_j_values)


def locate_balance(manifest):
    found = [(Path(path), digest) for path, digest in manifest.get('input_sha256', {}).items()
             if Path(path).name == 'balance.npz']
    if len(found) != 1 or sha(found[0][0]) != found[0][1]:
        raise ValueError('accepted material does not identify one unchanged balance.npz')
    return found[0][0], found[0][1]


def run(input_root, phot_library, kernel_library, out):
    input_root = Path(input_root).resolve()
    phot_library, kernel_library = Path(phot_library).resolve(), Path(kernel_library).resolve()
    out = Path(out).resolve()
    if (out.exists() or out == input_root or input_root in out.parents or
            out in input_root.parents):
        raise FileExistsError('native-grid diagnostic output must be new and outside sealed inputs')
    native, field, material = input_root/'native', input_root/'field', input_root/'material'
    field_manifest_path = field/'fixedj-field-input.json'
    field_manifest = json.loads(field_manifest_path.read_text())
    if (field_manifest.get('schema') != 'lumina-native-current-fixedj-input-v1' or
            field_manifest.get('layout', {}).get('frequencies') != 1115020):
        raise ValueError('fixed-J field manifest is not the sealed complete current field')
    fine_nu_path, fine_nu_sha = field_input(field_manifest, 'frequency.npy', field)
    fine_j_path, fine_j_sha = field_input(field_manifest, 'J.npy', field)
    fine_nu = np.load(fine_nu_path, mmap_mode='r', allow_pickle=False)
    fine_j = np.load(fine_j_path, mmap_mode='r', allow_pickle=False)
    if (fine_nu.shape != (1115020,) or fine_j.shape != (90, 1115020) or
            not np.isfinite(fine_nu).all() or np.any(fine_nu <= 0) or
            np.any(np.diff(fine_nu) <= 0)):
        raise ValueError('sealed Lumina fine-grid J/frequency dimensions or order invalid')
    for name in ('EDDFACTOR', 'EDDFACTOR_INFO'):
        expected = field_manifest['output_sha256'][name]
        if sha(field/name) != expected or sha(native/name) != expected:
            raise ValueError('native and field fixed-J inputs differ from the field manifest')
    accepted_path = material/'accepted-material.json'
    accepted = json.loads(accepted_path.read_text())
    if (accepted.get('schema') != 'lumina-accepted-nodal-material-v1' or
            accepted.get('status') != 'EXPORTED_NOT_TRANSPORT_READY'):
        raise ValueError('accepted physical material manifest missing')
    accepted_policy = accepted.get('thermal_energy_policy')
    if accepted_policy not in ('physical_stored', 'physical_instantaneous_scenario',
                               'native_stored', 'native_instantaneous_scenario'):
        raise ValueError('accepted material has no explicit supported thermal energy policy')
    for name, expected in accepted['output_sha256'].items():
        if sha(material/name) != expected:
            raise ValueError('accepted material output changed: '+name)
    full_population = material/'C-imported-full.bin'
    balance_path, balance_sha = locate_balance(accepted)
    v6_path = input_root/'comparison-policy-v6.json'
    v6 = json.loads(v6_path.read_text())
    if (v6.get('exact_injected_state') is not True or v6.get('heat_row_same_state') is not True or
            v6.get('final_goal') != 'NOT_MET' or
            v6.get('native_heat_comparison', {}).get('lumina_balance_sha256') != balance_sha):
        raise ValueError('native/Lumina comparison is not tied to the accepted same-state v6 record')
    heat_row = read_heat_row(native/'NATIVE_HEAT_ROW.bin')
    if heat_row['frequencies'] != v6['native_heat_comparison']['native_frequency_count']:
        raise ValueError('native heat-row frequency count differs from v6')
    if heat_row['evaluation_counter'] != v6['native_evaluation_counter']:
        raise ValueError('native heat row is from a different evaluation')
    controls = read_native_heat_controls(native/'VADAT')
    if controls != {'COH_ES': 'T', 'GLOBAL_LINE': 'BLANK',
                    'SCL_LN': 'T', 'SCL_LN_FAC': 0.5}:
        raise ValueError('thermal comparison requires the sealed COH_ES/GLOBAL_LINE/SCL_LN contract')
    cont_freq = native/'CONT_FREQ'
    nu, nu_cont, grid_meta = parse_native_cont_freq_text(cont_freq.read_text())
    if len(nu) != heat_row['frequencies']:
        raise ValueError('CONT_FREQ NCF differs from native heat row')
    if (not phot_library.is_file() or not kernel_library.is_file() or
            not full_population.is_file()):
        raise FileNotFoundError('accepted source-generation input is missing')
    edd_nu, edd_j = read_eddfactor(field/'EDDFACTOR', field/'EDDFACTOR_INFO')
    field_agreement = compare_fine_field_to_eddfactor(fine_nu, fine_j, edd_nu, edd_j)
    if field_agreement['max_frequency_relative_difference'] > 1.e-12:
        raise ValueError('fine-grid frequency table and EDDFACTOR do not describe the same grid')
    j_native_same_field = interpolate_native_j(fine_nu, fine_j.T, nu)
    j_native_cmfgen = interpolate_native_j(edd_nu, edd_j, nu)
    fine_weights = fine_grid_trapezoid_weights(fine_nu)
    native_weights = native_trapuneq_weights(nu)
    out.mkdir(parents=True)
    fine_dir = out/'fine-grid'
    fine_dir.mkdir()
    native_dir = out/'native-grid'
    native_dir.mkdir()
    nu_path, cont_path = native_dir/'frequency.npy', native_dir/'continuum_frequency.npy'
    np.save(nu_path, nu, allow_pickle=False)
    np.save(cont_path, nu_cont, allow_pickle=False)
    started = time.monotonic()
    fine_started = time.monotonic()
    fine_source_out = fine_dir/'sources'
    S.prepare(native, material/'native_material_nodes.bin', material/'atomic',
        phot_library, kernel_library, full_population, fine_nu, fine_source_out,
        frequency_grid=fine_nu_path, free_free_policy='NATIVE_UPPER_IONS')
    fine_source_seconds = time.monotonic()-fine_started
    fine_source_manifest = json.loads((fine_source_out/'sources.json').read_text())
    fine_components, fine_gross = heat_components(
        fine_source_out/'sources.npz', fine_j, fine_weights, fine_nu)
    lumina_fine = fine_components['total_no_coherent_ES']
    native_started = time.monotonic()
    native_source_out = native_dir/'sources'
    S.prepare(native, material/'native_material_nodes.bin', material/'atomic',
        phot_library, kernel_library, full_population, nu, native_source_out,
        frequency_grid=nu_path, continuum_frequency_grid=cont_path,
        free_free_policy='NATIVE_UPPER_IONS')
    native_source_seconds = time.monotonic()-native_started
    native_source_manifest = json.loads((native_source_out/'sources.json').read_text())
    native_same_components, native_same_gross = heat_components(
        native_source_out/'sources.npz', j_native_same_field, native_weights, nu)
    native_cmfgen_components, native_cmfgen_gross = heat_components(
        native_source_out/'sources.npz', j_native_cmfgen, native_weights, nu)
    lumina_native_same_field = native_same_components['total_no_coherent_ES']
    lumina_native_cmfgen_j = native_cmfgen_components['total_no_coherent_ES']
    native_heat = heat_row['steq_t_native'] * CMFGEN_RE_INTERNAL_TO_CGS
    with np.load(balance_path, allow_pickle=False) as balance:
        old_lumina = np.asarray(balance['native_stored_net'], dtype=np.float64)
        if 'native_instantaneous_scenario_net' not in balance.files:
            raise ValueError('accepted balance lacks separately identified native instantaneous scenario')
        native_instantaneous = np.asarray(
            balance['native_instantaneous_scenario_net'], dtype=np.float64)
    if (old_lumina.shape != native_heat.shape or
            native_instantaneous.shape != native_heat.shape or
            lumina_fine.shape != native_heat.shape or
            lumina_native_same_field.shape != native_heat.shape or
            lumina_native_cmfgen_j.shape != native_heat.shape):
        raise ValueError('accepted, native, and new heat vectors do not share 90 points')
    rows = []
    for i in range(len(native_heat)):
        rows.append(dict(depth_1based=i+1,
            native_steq_t_cgs=float(native_heat[i]),
            lumina_native_scl_stored_balance_net=float(old_lumina[i]),
            lumina_native_scl_instantaneous_scenario_net=float(native_instantaneous[i]),
            lumina_fine_grid_BF=float(fine_components['BF'][i]),
            lumina_fine_grid_FF=float(fine_components['FF'][i]),
            lumina_fine_grid_BB=float(fine_components['BB'][i]),
            lumina_fine_grid_net=float(lumina_fine[i]),
            lumina_fine_grid_gross=float(fine_gross[i]),
            lumina_native_same_field_BF=float(native_same_components['BF'][i]),
            lumina_native_same_field_FF=float(native_same_components['FF'][i]),
            lumina_native_same_field_BB=float(native_same_components['BB'][i]),
            lumina_native_same_field_net=float(lumina_native_same_field[i]),
            lumina_native_same_field_gross=float(native_same_gross[i]),
            lumina_native_EDDFACTOR_J_net=float(lumina_native_cmfgen_j[i]),
            lumina_native_EDDFACTOR_J_gross=float(native_cmfgen_gross[i])))
    with (out/'depths.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    inputs = {name: sha(path) for name, path in (
        ('CONT_FREQ', cont_freq), ('EDDFACTOR', field/'EDDFACTOR'),
        ('EDDFACTOR_INFO', field/'EDDFACTOR_INFO'),
        ('NATIVE_HEAT_ROW.bin', native/'NATIVE_HEAT_ROW.bin'),
        ('VADAT', native/'VADAT'),
        ('fixedj-field-input.json', field_manifest_path),
        ('fine-frequency.npy', fine_nu_path), ('fine-J.npy', fine_j_path),
        ('accepted-material.json', accepted_path),
        ('C-imported-full.bin', full_population),
        ('comparison-policy-v6.json', v6_path), ('balance.npz', balance_path),
        ('phot-library', phot_library), ('kernel-library', kernel_library))}
    result = dict(schema='lumina-native-grid-heat-comparison-v3',
        scope='same accepted material/full population and same fine-grid J; direct Lumina source generation on both sealed 1,115,020-point and printed native NCF grids; separates grid/quadrature sensitivity from native-J and source-policy differences; diagnostic only',
        grid=grid_meta, fine_frequency_count=len(fine_nu), native_frequency_count=len(nu),
        continuum_evaluation_unique_count=int(len(np.unique(nu_cont))),
        interpolation='linear-frequency interpolation with endpoint clamp, matching CMFGEN RD_CONT_J',
        quadrature=dict(fine_grid='composite trapezoid nodal weights on sealed original frequency.npy',
            native_grid='TRAPUNEQ nodal weights reconstructed from printed NU in CONT_FREQ; converted to Hz'),
        fine_grid_input_sha256=dict(frequency=fine_nu_sha, J=fine_j_sha),
        fine_field_vs_EDDFACTOR=field_agreement,
        source_components=dict(bound_free='direct evaluation at requested grid; native run uses CONT_FREQ mapping',
            free_free='NATIVE_UPPER_IONS',
            bound_bound='local line coefficients are unscaled for transport/source-grid diagnostics',
            coherent_electron_scattering='excluded from thermal heat', native_VADAT_controls=controls),
        thermal_policy=dict(
            accepted_material_policy=accepted_policy,
            native_line_scaling_enabled=controls['SCL_LN'] == 'T',
            native_line_scaling_ratio_limit=controls['SCL_LN_FAC'],
            scaling_scope='native RE line heating/cooling ledger; do not scale transport chi/eta',
            accepted_balance='native_stored_net includes the native SCL_LN line-energy factor',
            comparison_to_native_STEQ_T=('compare native_stored_net without the separate instantaneous-scenario term; '
                'captured CMFGEN STEQ_T routes explicit decay separately')),
        bound_bound_comparison_note=('Raw Lumina local BB coefficients omit CMFGEN SCL_LN and are not native thermal parity. '
            'The separately reported native_stored_net includes SCL_LN, but still is not termwise LINEHEAT parity.'),
        luminosity_units='erg cm^-3 s^-1', native_sign='STEQ_T positive=net heating',
        native_steq_t_to_cgs_factor=CMFGEN_RE_INTERNAL_TO_CGS,
        comparison=dict(
            same_J_fine_grid_vs_native_grid=metrics(lumina_fine, lumina_native_same_field),
            native_scl_stored_balance_vs_raw_fine_grid_source_diagnostic=metrics(old_lumina, lumina_fine),
            native_steq_t_vs_unscaled_native_grid_source_using_EDDFACTOR_J=metrics(native_heat, lumina_native_cmfgen_j),
            native_steq_t_vs_native_scl_stored_balance=metrics(native_heat, old_lumina),
            native_same_field_J_vs_EDDFACTOR_J=metrics(lumina_native_same_field, lumina_native_cmfgen_j)),
        depth_count=len(rows), evaluation_counter=heat_row['evaluation_counter'],
        source_model=dict(
            fine_grid=dict(ions=fine_source_manifest['model_ions'],
                full_levels=fine_source_manifest['full_levels'], routes=fine_source_manifest['routes'],
                lines=fine_source_manifest['lines'], memory=fine_source_manifest['allocation_inventory'],
                output_sha256=fine_source_manifest['output_sha256'],
                source_manifest_sha256=sha(fine_source_out/'sources.json')),
            native_grid=dict(ions=native_source_manifest['model_ions'],
                full_levels=native_source_manifest['full_levels'], routes=native_source_manifest['routes'],
                lines=native_source_manifest['lines'], memory=native_source_manifest['allocation_inventory'],
                output_sha256=native_source_manifest['output_sha256'],
                source_manifest_sha256=sha(native_source_out/'sources.json'))),
        input_sha256=inputs, elapsed_seconds=time.monotonic()-started,
        source_generation_seconds=dict(fine_grid=fine_source_seconds,
            native_grid=native_source_seconds),
        new_transport=0, new_SE_RE_solves=0, material_updates=0,
        physical_equivalence='DIAGNOSTIC_ONLY_NO_PREDECLARED_THRESHOLD', final_goal='NOT_MET')
    with (out/'native-grid-heat-comparison.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(out=out.name, comparison=result['comparison'],
        source_generation_seconds=result['source_generation_seconds'],
        elapsed_seconds=result['elapsed_seconds'], final_goal='NOT_MET'), ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--phot-library', type=Path, required=True)
    parser.add_argument('--kernel-library', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.input_root, args.phot_library, args.kernel_library, args.out)
