# CODEX 구현 보고 — A2-08

기준/최종 HEAD는 모두 `694d9cdc297c97082d2c1fa731c5a9fc7ba591ce`이며 commit/push는 하지 않았다. source diff hash는 `12f9593c2057bc74d7e74d15bd4d7e4ce655f80d92e550c80166431020670dec`다. 동시 저작 대상 `docs/SPEC_A2_09_10_V1.md`, `docs/SPEC_A2_12_V1.md`는 수정하지 않았다.

## 구현 결과

signed Sobolev direct-difference helper, value/status line publication, ES/BB/BF/FF/total component owner와 원자 commit, BF signed-net/gross-event API 분리, total-variation 경계 measure, transport/formal/heating/transition capability BLOCK을 구현했다. 단위는 `cm^-1`, frame은 comoving, grid는 BF 1000-bin log-frequency projection이며 total 순서는 `((es+bb)+bf)+ff`다. 합성 selftest worst closure는 0이다.

## 54행 처분 결과

| ID | 구현 후 anchor | 처분 | capability | 후속 |
|---|---|---|---|---|
| T01 | src/lumina_transport.c:200 `tau_sobolev` | blocked | BLOCK_UNSUPPORTED | A2-11M |
| T02 | src/lumina_transport.c:574 `chi_e` | migrate | SEPARATE_NONNEG_EVENT_MEASURE | A2-08 |
| T03 | src/lumina_transport.c:580 `T03` | blocked | BLOCK_UNSUPPORTED | A2-11M |
| F01 | src/lumina_cmf_field.c:227 `chi` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| F02 | src/lumina_cmf_field.c:302 `delta_tau` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| F03 | src/lumina_cmf_field.c:710 `chi_up` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| F04 | src/lumina_cmf_field.c:945 `chi_u` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| F05 | src/lumina_cmf_field.c:1908 `chi_coherent` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| F06 | src/lumina_cmf_field.c:2182 `chi_total` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G01 | src/lumina_cmfgen.c:281 `chi_tot` | migrate | SIGNED_EQUATION | A2-08 |
| G02 | src/lumina_cmfgen.c:736 `tau` | migrate | SIGNED_EQUATION | A2-08 |
| G03 | src/lumina_cmfgen.c:1169 `tau` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G04 | src/lumina_cmfgen.c:1287 `tau` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G05 | src/lumina_cmfgen.c:1626 `tau_sobolev` | migrate | SIGNED_EQUATION | A2-08 |
| G06 | src/lumina_cmfgen.c:1789 `chi_line` | migrate | SIGNED_EQUATION | A2-08 |
| G07 | src/lumina_cmfgen.c:1797 `tau` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G08 | src/lumina_cmfgen.c:1976 `acc_tau` | migrate | SIGNED_EQUATION | A2-08 |
| G09 | src/lumina_cmfgen.c:2091 `chi_es` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G10 | src/lumina_cmfgen.c:2469 `dtau` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G11 | src/lumina_cmfgen.c:2585 `chi_tot` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G12 | src/lumina_cmfgen.c:2757 `tau_r` | keep_allowed | OUTPUT_ONLY | A2-08 |
| G13 | src/lumina_cmfgen.c:2890 `chi_tot` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G14 | src/lumina_cmfgen.c:2980 `chi_tot` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G15 | src/lumina_cmfgen.c:3078 `chi_tot` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G16 | src/lumina_cmfgen.c:3108 `dtau` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G17 | src/lumina_cmfgen.c:3223 `tau_sobolev` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G18 | src/lumina_cmfgen.c:3581 `chi_tot` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G19 | src/lumina_cmfgen.c:3786 `chi_tot` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G20 | src/lumina_cmfgen.c:3841 `line_source_S` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G21 | src/lumina_cmfgen.c:4111 `chi_es` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| G22 | src/lumina_cmfgen.c:4474 `line_source_S` | keep_allowed | OUTPUT_ONLY | A2-08 |
| G23 | src/lumina_cmfgen.c:5176 `tau_sobolev` | blocked | BLOCK_UNSUPPORTED | A2-11M |
| P01 | src/lumina_plasma.c:2148 `tau_sobolev` | migrate | SIGNED_EQUATION | A2-08 |
| P02 | src/lumina_plasma.c:2978 `tau_validity` | migrate | SIGNED_EQUATION | A2-08 |
| P03 | src/lumina_plasma.c:7023 `chi_bf` | migrate | SEPARATE_NONNEG_EVENT_MEASURE | A2-08 |
| P04 | src/lumina_plasma.c:7673 `stim_route_begin` | migrate | SEPARATE_NONNEG_EVENT_MEASURE | A2-08 |
| P05 | src/lumina_plasma.c:7124 `chi_bf` | migrate | SEPARATE_NONNEG_EVENT_MEASURE | A2-08 |
| P06 | src/lumina_plasma.c:4505 `chi_ff_nnionpart` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| P07 | src/lumina_plasma.c:5677 `tau` | keep_allowed | OUTPUT_ONLY | A2-08 |
| P08 | src/lumina_plasma.c:8921 `tau_sobolev` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| P09 | src/lumina_plasma.c:12049 `bl_tau` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| P10 | src/lumina_plasma.c:12049 `bl_tau` | migrate | SIGNED_EQUATION | A2-08 |
| P11 | src/lumina_plasma.c:12280 `chi_nu` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| P12 | src/lumina_plasma.c:13949 `tau_rec` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| P13 | src/lumina_plasma.c:11497 `chi` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| P14 | src/lumina_plasma.c:15438 `tau` | blocked | BLOCK_UNSUPPORTED | A2-11M |
| P15 | src/lumina_plasma.c:17498 `tau` | migrate | SIGNED_EQUATION | A2-08 |
| P16 | src/lumina_plasma.c:18188 `tau_sobolev` | blocked | BLOCK_UNSUPPORTED | A2-11M |
| P17 | src/lumina_plasma.c:18656 `tau` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| P18 | src/lumina_plasma.c:19048 `chi_l` | blocked | BLOCK_UNSUPPORTED | A2-11 |
| E01 | src/lumina_element_wide.c:1783 `tau` | migrate | SEPARATE_NONNEG_EVENT_MEASURE | A2-08 |
| E02 | src/lumina_element_wide.c:2243 `tau_all` | migrate | SEPARATE_NONNEG_EVENT_MEASURE | A2-08 |
| E03 | src/lumina_element_wide.c:2269 `tau_boundary` | keep_allowed | SEPARATE_NONNEG_EVENT_MEASURE | A2-08 |
| M01 | src/lumina_main.c:728 `tau` | migrate | SIGNED_EQUATION | A2-08 |

checker 실측은 raw_hits=2635, classified_hits=2635, unknown=0, sites=54, migrate=16, keep=4, blocked=34, duplicate=0, silent clamp/floor=0, numeric line-source sentinel=0이다.

## allowlist seal

- baseline HEAD: `694d9cdc297c97082d2c1fa731c5a9fc7ba591ce`
- object-only seal commit(브랜치/ref 미갱신): `78e1879877fd6dbe435a24aabfb21c66ef3ef158`
- JSON blob: `b65e5d8ac121f9c1b2feadd160ab0eebc1c4a129`
- JSON/current/sidecar/sealed SHA-256: `e0f89c1b9851543f995d459890d7d32bdf62bf642d7e8bdb6957e7f5a482fafb` (3중 일치)
- seal diff: allowlist JSON과 sidecar 두 파일만 포함

## 게이트 산출물

- `make lumina`: PASS
- A2-03/04/05/06/07 selftests: PASS
- A2-08 positive + N1-N8: PASS(8/8); negative tau=1, negative BF route=1, closure=0
- A2-01 canonical: PASS rows=157 completed=20 unclassified=0
- Z 전용: 네 hard-coded build가 새 TU를 직접 link; runner Z=7, PASS=7/FAIL=0
- canonical Z: active_lines=2211572, bit_differences=0, signed hash=`4a80c65d9c37fad9`
- L-4 CHAIN/ORACLE_INPUT: `BLOCKED_MISSING_CHI_DATA`, child rc 3. PASS로 승격하지 않음.

## 운전석 실행 명령

```bash
make lumina
make selftest_a2_08_signed_opacity
python3 scripts/a2_08_verify_allowlist_seal.py
python3 scripts/a2_08_signed_consumer_census.py check
python3 scripts/a2_01_census_contract.py check
python3 scripts/run_gate_battery.py --verify-equivalence
```

마지막 명령은 대형 병렬/배터리이므로 구현자는 실행하지 않았고 운전석(lageunha)에 남겼다.

## 남은 위험

CHI_DATA/CHI_DATA_INFO가 없어 truth coverage와 L-4 물리 비교는 실행 불가다. canonical Z에서 legacy floor 대비 1,074,495 bit 변화가 관측됐으며 signed oracle로 재기준화한 뒤 허용 밖 차이는 0이다. 실제 대형 모델의 component artifact/음수 identity는 운전석 실행에서 재생성·확인해야 한다. 기존 warning(`strdup` feature declaration, legacy OpenMP pragma/indent)은 이 단계 밖이며 빌드 rc는 0이다.

## A2-09 인계

`A2-01:old7897:T_rad`, BF Planck 재방출, `eta_reemit`, CDF builder/sampler와 RNG draw count는 diff 0으로 A2-09에 남겼다. G03/G04/G07/G09/P06의 emissivity/source/transition 의미도 A2-09 소유다. A2-10은 signed heating/RADEQ, A2-11은 formal 수식, 제안 A2-11M은 MC maser/overlap 의미, A2-12+는 GPU를 인계받는다.
