# Gate B dual oracle Phase 1.6 — Codex A 구현 보고서

## 1. 결론

C15의 잔존 FAIL ③(실질 커버리지)과 ④(동결 생산값 재현), B15의 경미
2건을 정면 수정했다. 기존 PASS 축인 관측 전용, gate OFF 무접촉,
결정론, `n_e` 단위 실증, s43 속도 대응은 유지된다.

Phase 1.6의 핵심 변화는 다음과 같다.

1. 생산 epoch를 `rate/thermal consume iter=11`과 이를 공급한
   `C1/C2 producer iter=10`으로 정렬하고, 정확히 1 iteration lag가 아니면
   하니스가 중단하도록 했다.
2. 원 raw-Jbar가 보존된 Si II/III와, 보존되지 않은 여섯 이온의 생산
   C1-fallback replay를 구분했다. 원 raw 값을 위조하지 않으면서 계산 가능한
   대표 J, bb rate, collisional rate를 생산 함수에서 수치화했다.
3. parity50에서 thermal 등록 입력이 실제로 소비됐는지 resolved config와
   소비 게이트를 함께 검사했다. tail color, global-Newton tri response,
   line-RE, lagged photoion-MC는 모두 비활성이므로 NULL 등록이 정확한 복원값이다.
   향후 활성 구성은 archive가 없으면 fail-closed한다.
4. 모든 결손 사유를 disposition으로 분류하고 미등록/빈 사유가 하나라도
   있으면 비교자가 중단하는 coverage gate를 추가했다.
5. 향후 재실행용 `LUMINA_GATEB_ORACLE_CAPTURE=1`을 추가했다. 이 gate는
   기존 Jbar observer를 8개 대표 이온 전체로 확장하고, MA line-destruction의
   per-shell event/rate ledger를 쓴다. default OFF이며 물리값을 읽기만 한다.

## 2. 생산 epoch 정렬

생산 순서는 outer iteration 초반에 이전 transport field로 thermal/rate solve를
수행하고, iteration 끝 transport 뒤 C1/C2를 다음 iteration용으로 발행한다.
따라서 마지막 rate block `iter=11`의 입력은 C1/C2 `iter=10`이다.

각 셀 CSV에는 다음 계약을 독립 행으로 기록한다.

| 항목 | s0 | s8 | s43 |
|---|---:|---:|---:|
| `rate_consume_iteration` | 11 | 11 | 11 |
| `field_producer_iteration` | 10 | 10 | 10 |
| `producer_to_consumer_lag` | 1 | 1 | 1 |
| C1 coarse bins | 24/24 | 24/24 | 24/24 |
| C2 fine bins | 1000/1000 | 1000/1000 | 1000/1000 |
| exact raw-Jbar ions | 2/8 | 2/8 | 2/8 |

C1, C2, Jbar loader는 더 이상 각 파일의 독립 최대 iteration을 고르지 않는다.
Jbar의 authoritative consumer epoch를 먼저 정한 뒤 C1/C2는 정확히
`consumer-1` 블록만 읽으며, bin 완결성과 중심 주파수를 검사한다.

## 3. Jbar 6이온 결손과 수치 회수

원 parity50의 `LUMINA_JBAR_DUMP_IONS=14:1,14:2` 필터 때문에 S II/III,
Fe II/III/IV, Co III의 raw `jbar_line`, count, tau는 소급 복원이 불가능하다.
Phase 1.6은 두 층을 분리한다.

- `jbar_input_raw`: 원 계측값이므로 계속 unavailable이다. 사유에는 필터 전
  유실, tau 포함 비복원성, capture gate를 명시한다.
- `jbar_representative`, `R_lu`, `R_ul`, `C_lu`, `C_ul`: 동결 C1 field를
  실제 `nlte_assemble_rate_matrix`에 넣은 생산 fallback replay 값을 낸다.
  producer는 `C1_fallback_replay`로 표시해 원 raw 계측과 혼동하지 않는다.
- frozen population에 양의 lower-level flow가 없는 셀/이온만 실제
  snapshot 결손으로 남긴다.

그 결과 bb의 Lumina unavailable은 120→70, collisional은 40→20으로
감소했다. bb strict comparison은 8→18로 늘었다. 원 raw-Jbar/tau 재실행
복구 대상은 현재 대표 전이가 존재하는 10행이며
`LUMINA_GATEB_ORACLE_CAPTURE=1` disposition으로 모인다.

## 4. Thermal 등록 입력과 MA line destruction

parity50 resolved config의 실제 소비 상태:

| 등록 입력 | 소비 gate | 값 | Phase 1.6 복원 |
|---|---|---:|---|
| tail color | `LUMINA_A4_TAIL_COLOR` | OFF | `radeq_set_tail_color(NULL,0)` |
| tri response | `LUMINA_COUPLED_NEWTON` | 0 | NULL registration |
| line response/source | `LUMINA_RADEQ_LINE_RE` | 0 | NULL registration |
| lagged photoion MC | `LUMINA_COEVOLVE_PHOTOION_MC` | OFF | NULL registration |

`LUMINA_RADEQ_LINE_RESPOND=1`은 별도 등록 배열이 아니라
`radeq_simul_all`이 동결 population/atomic table에서 직접 만드는 생산
line-response 경로이므로 기존 생산 호출에서 재현된다.

향후 `LUMINA_COEVOLVE_PHOTOION_MC=1` 구성은 C2 dump의 `J_raw`와
`j_nu_count`, resolved alpha로 등록값을 복원한다. tail/tri/line-RE가 활성인데
전용 archive가 없으면 하니스는 0으로 대체하지 않고 중단한다.

기존 archive에서 MA line destruction은 producer iter 10의 전역
`terminals=506,685,974`, `destroyed=554,208`까지 수치 회수했다. 하지만
shell ownership과 packet-energy/volume 정규화가 저장되지 않아 셀별
`heating_MA_LINE_DESTRUCT`는 진짜 unavailable이다. 새 capture gate는
`lumina_ma_line_destruct.csv`에 iter/shell별 terminals, destroyed,
`erg s^-1 cm^-3`를 기록하므로 재실행 시 이 3행을 채운다.

## 5. 커버리지

| 지표 | Phase 1.5 | Phase 1.6 | 변화 |
|---|---:|---:|---:|
| 전체 행 | 546 | 582 | +36 provenance |
| strict `compared` | 79 | 89 | +10 |
| context numeric | 9 | 9 | 0 |
| Lumina unavailable | 241 | 171 | -70 |
| Lumina 값이 존재하는 행 | 305 | 411 | +106 |
| strict 비율 | 14.47% | 15.29% | +0.82%p |

`coverage_disposition.csv`는 모든 non-compared 행을 category/status/사유별로
집계하고 다음 중 하나로 폐쇄한다.

- 선택 CMFGEN 출력에 동일 수량 앵커 없음
- frozen NLTE topology상 lower pair 부재
- 해당 snapshot의 zero population/no positive flow
- 원 archive 유실이나 capture gate로 재실행 복구 가능

빈 사유나 등록되지 않은 Lumina 결손 사유는 비교 생성 자체를 실패시킨다.

## 6. B15 경미 2건

- Fortran form-feed를 `splitlines()`가 행으로 세던 오류를 제거했다. newline
  기준 source evidence는 `cmfgen_sub.f:4421`, `cmfgen_sub.f:4423`,
  `mod_cmfgen.f:211`이며 Phase 1.5 저장 CSV도 정정했다.
- snapshot 설명은 s0/s8의 16행만 `same_snapshot`이고, s43의 8행은
  상대차 `-1.8805437716e-3`의
  `different_snapshot_or_output_generation`임을 명시한다.

## 7. 재현성과 불변축

동일 바이너리와 동일 입력으로 CPU single-thread 하니스를 두 번 실행했다.
각 CSV는 194 data row이며 두 실행의 `cmp`는 모두 0이다.

| 셀 | SHA-256 |
|---:|---|
| s0 | `8cbccb2cac2fb7b860eac45edd8479f36f5f5b010e0dd3708d463eff389332b6` |
| s8 | `dad29ce6b39a00609f6b63aa06cb85c8fb323212921081d434d8ca5510115767` |
| s43 | `432952ec471323a7d164a31792c21d117cbc3221af3ac63d753696f57f182112` |

비교자 산출 8종도 독립 두 번 생성해 모두 byte-identical이었다.

관측 전용/OFF 확인:

- 기본 object와 명시적 `-ULUMINA_FROZEN_ORACLE` object는 byte-identical
  (`849c48bb...61a3d7`)이다.
- 두 object 모두 `lumina_oracle`/`g_oracle` symbol이 0개다.
- capture gate OFF에서는 Jbar 필터 확장, MA per-shell counter readback,
  CSV I/O가 실행되지 않는다.
- CMFGEN 단위 실증과 s43 속도 대응 산출은 그대로 유지된다.

GPU 실행, `git` 명령은 사용하지 않았다. CUDA capture observer는 이번
지시상 GPU로 실행 검증하지 않았고, CPU oracle 및 OFF object 검증만 수행했다.

## 8. 산출물

- Phase 1.6 oracle/comparator: `validation/gate_b_dual_oracle/phase1_6/`
- coverage 원장: `coverage_disposition.csv`
- 스모크 hash: `smoke_sha256.txt`
- 입력 hash: `frozen_input_sha256.txt`
- OFF audit: `off_object_sha256.txt`, `off_compile_audit.txt`
- 구현: `bench_frozen_oracle.c`, `src/lumina_plasma.c`,
  `src/lumina_cuda.cu`, `src/lumina.h`
- 비교자: `scripts/oracle_compare_cmfgen.py`

