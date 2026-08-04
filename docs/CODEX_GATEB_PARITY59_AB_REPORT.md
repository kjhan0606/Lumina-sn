# Gate B oracle parity59 캡처 재실행 — Codex A+B 폐합 보고

작성일: 2026-07-31  
입력: `logs/coevolve_consume_parity59/`  
산출: `validation/gate_b_dual_oracle/parity59/`

## 1. 판정

- ③ 캡처 커버리지 결손: **해소분 PASS**. parity50에서 유실이던 raw-J̄
  비교 10행을 모두 회수했고 archive 유실은 13→3행으로 감소했다.
  49GB 원장은 한 번에 메모리로 올리지 않고 단일 순차 패스로 iter 11,
  s0/s8/s43, 8개 대상 이온만 선별했다.
- ④ 생산값 재현: **부분 PASS / exact-double 폐합은 보류**.
  캡처에 기록된 raw-J̄는 14/14, β는 14/14, 직접 소비 mode-3 J는
  11/11 일치했다. 그러나 writer가 C1/C2/J̄를 여전히 6자리로 기록하고,
  mode-0 C1 소비 J 3건과 C2에서 조립된 생산 R_bf 결과는 별도 보존하지
  않아 원 생산 double과의 독립 exact 비교는 불가능하다.
- 재현성: oracle 3셀 2회와 비교자 8종 2회가 모두 byte-identical.
- OFF: 기본 object와 명시적 `-ULUMINA_FROZEN_ORACLE` object가
  byte-identical이고 두 object 모두 oracle symbol 0개.
- GPU와 git 명령은 사용하지 않았다.

## 2. 49GB 스트리밍 선별 및 완결성

| 항목 | 값 |
|---|---:|
| 원본 크기 | 49,272,071,814 bytes |
| 원본 data rows | 683,984,500 |
| 선별본 크기 | 298,840,263 bytes |
| 선별 data rows | 4,103,907 |
| malformed | 0 |
| consumer iter | 11 |
| 선별 셸 | s0, s8, s43 |
| 완결성 sentinel | s49 |

각 셸/이온의 line-id `count/first/last/sum/FNV-1a/order`를 s49와
대조했다. 실재하는 7개 이온은 네 셸에서 모두 동일했다. Fe IV는 네 셸
모두 0행이며, 현 frozen topology에서 lower member pair가 없어 생산
J̄ 소비점이 생기지 않는 구조적 正결손이다.

원본 SHA-256:
`beebb3b65ae8a3e9d03ae5b811c4c37c7f9282fdadab43c450676e32e7fde5d0`.

## 3. Oracle 재실행

세 셀 모두 다음 epoch 계약을 만족했다.

| 셸 | consume | producer | lag | C1 | C2 | J̄ rows | exact raw-J̄ ions |
|---:|---:|---:|---:|---:|---:|---:|---:|
| s0 | 11 | 10 | 1 | 24/24 | 1000/1000 | 1,367,969 | 7/8 |
| s8 | 11 | 10 | 1 | 24/24 | 1000/1000 | 1,367,969 | 7/8 |
| s43 | 11 | 10 | 1 | 24/24 | 1000/1000 | 1,367,969 | 7/8 |

각 CSV는 194 data row이며 두 실행의 `cmp`는 전부 0이다.

| 셸 | SHA-256 |
|---:|---|
| s0 | `7a79f4f345d5c8500b48530a7f859bbe05a57771f43e52f983639ef9a0839381` |
| s8 | `2ee175c6be4a0ab6ae07034722180c6312f0a6e55e1009fcdda23b72310693eb` |
| s43 | `f75b84a314e85831825aea3e2ef64d9bdbc1729c50e83132b6e02c2151b5cd8e` |

## 4. 최종 커버리지

### 4.1 Phase 1 strict 및 Phase 1.6 delta

| 기준 | Phase 1 | Phase 1.6 | parity59 최종 | 최종−P1 | 최종−P1.6 |
|---|---:|---:|---:|---:|---:|
| strict identical | 33/484 (6.82%) | 89/582 (15.29%) | **99/582 (17.01%)** | +66, +10.19%p | +10, +1.72%p |
| nonidentical context 포함 | — | 98/582 (16.84%) | **108/582 (18.56%)** | — | +10, +1.72%p |

### 4.2 수치 보유 / 正결손 / archive 유실

| 구분 | Phase 1.6 | parity59 최종 | delta |
|---|---:|---:|---:|
| 수치 보유·대응 가능 | 411 (70.62%) | **421 (72.34%)** | +10, +1.72%p |
| 正결손 | 158 (27.15%) | **158 (27.15%)** | 0 |
| archive 유실 | 13 (2.23%) | **3 (0.52%)** | −10, −1.72%p |
| 합계 | 582 | **582** | 0 |

strict 증분 10행은 모두 새로 회수된 `jbar_input_raw`와 CMFGEN
EDDFACTOR 비교다. CMFGEN 동일 수량이 없는 Lumina 수치 행은 삭제하지
않고 `unavailable`로 유지했다.

## 5. 생산 소비값 재현 증빙

대표 전이 14건을 선별 raw 원장과 oracle CSV 사이에서 다시 조인했다.

| 검증 | 일치 |
|---|---:|
| captured raw-J̄ → oracle loaded raw-J̄ | **14/14 (100%) exact** |
| captured β → tau 역산 → oracle β | **14/14 (100%)**, 최대 상대오차 `2.223e-16` |
| raw-J̄를 직접 쓰는 mode 3 생산 J → oracle production J | **11/11 (100%) exact** |
| mode 0 C1 fallback production J | 0/3 independent exact check; 원 생산 per-line J 미보존 |

여기서 “exact”는 CSV에 보존된 decimal token을 double로 읽은 뒤의
일치다. 원 생산 double 자체는 writer의 6자리 반올림 때문에 bit-exact
복원 대상이 아니다. C2는 셀당 1000/1000 bin을 정확한 iter에서
소비했지만, 원 생산 `R_bf` 출력이 별도 보존되지 않아 재계산 결과와의
독립 일치율을 만들 수 없다.

thermal 등록 상태는 tail color, coupled-Newton tri response, line-RE,
lagged photoion-MC가 모두 OFF여서 NULL 복원이 정확하다. 반면 parity59
디렉터리에는 `lumina_ma_line_destruct.csv`가 없고 stdout의 전역
terminals/destroyed만 남아 있으므로 셸별 MA 체적 열화율은 복원할 수
없다. 또한 MA transport 열화는 생산 thermal residual에 등록·소비되는
입력이 아니므로 `thermal_net` 일치 근거로 합산하지 않았다.

## 6. 잔여 unavailable 사유

### 6.1 Lumina 수치 결손 161행

| 사유 | 행 |
|---|---:|
| lower pair가 없는 frozen NLTE topology | 60 |
| positive lower-population flow/대표 전이 없음 | 80 |
| non-sentinel 대표 b_k 없음 | 10 |
| element-stage census가 0 | 8 |
| 셸별 MA ledger archive 유실 | 3 |

앞의 158행은 snapshot/topology의 正결손이고 마지막 3행만 archive
유실이다.

### 6.2 선택 CMFGEN 출력에 동일 수량 앵커 없음 313행

| 사유군 | 행 |
|---|---:|
| matched bb rate/β 미노출 | 56 |
| α spont/stim split 및 monochromatic bf coefficient 미노출 | 114 |
| 대표 collisional coefficient 미노출 | 28 |
| monochromatic ff coefficient 미노출 | 12 |
| Gate-B input provenance에 대응 CMFGEN 수량 없음 | 72 |
| ion census/dilute-field scalar 미노출 | 22 |
| thermal 개별 항 미노출 | 9 |

## 7. 재현성과 OFF 불변

- oracle s0/s8/s43 2회: 3/3 `cmp=0`.
- 비교자 산출 8종 2회: 8/8 `cmp=0`.
- OFF object compile: 둘 다 exit 0, `cmp=0`.
- OFF object SHA-256:
  `123316ee167d01ecd560a78d2e5f8d36d0afeb355cefbd89aba7be8c8982a5dc`.
- `lumina_oracle|g_oracle` symbol: 기본 0, 명시 OFF 0.

## 8. 증빙 파일

- `validation/gate_b_dual_oracle/parity59/jbar_capture_manifest.csv`
- `validation/gate_b_dual_oracle/parity59/production_replay_consistency.csv`
- `validation/gate_b_dual_oracle/parity59/oracle_vs_cmfgen.csv`
- `validation/gate_b_dual_oracle/parity59/coverage_disposition.csv`
- `validation/gate_b_dual_oracle/parity59/oracle_byte_identity.txt`
- `validation/gate_b_dual_oracle/parity59/comparator_byte_identity.txt`
- `validation/gate_b_dual_oracle/parity59/off_compile_audit.txt`
- `validation/gate_b_dual_oracle/parity59/frozen_input_sha256.txt`
