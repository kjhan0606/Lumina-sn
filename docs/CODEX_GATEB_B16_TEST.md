Gate B 1.6 검증 판정: **PASS**.

### 핵심 결과

- `make -B bench_frozen_oracle`: 성공
- CPU oracle 2회: 모두 exit 0
- s0/s8/s43: 각 194행, 회차 간 및 제출본과 byte-identical
- 비교자: 구문 검사·실행 성공, 8종 산출물 모두 제출본과 byte-identical
- iter 정렬 결손 주입: exit 1, CSV 0개 — fail-closed 확인
- OFF object: byte-identical, oracle 심볼 0개
- 소스 수정·GPU·git 명령: 사용하지 않음

### Oracle 해시

| 셀 | SHA-256 |
|---|---|
| s0 | `8cbccb2c…89332b6` |
| s8 | `dad29ce6…81015767` |
| s43 | `432952ec…f182112` |

정상 입력은 모두 `consume=11 / producer=10 / lag=1`, C1 `24/24`, C2 `1000/1000`, exact raw-J̄ ions `2/8`입니다.

### 최종 커버리지

3분할은 전체 행을 “Lumina 수치 보유 / 正결손 / archive 유실”로 완전히 분류합니다.

| 구분 | 1.5 | 1.6 | delta |
|---|---:|---:|---:|
| 전체 | 546 | 582 | +36 |
| 수치 보유·대응 가능 | 305 (55.86%) | 411 (70.62%) | **+106, +14.76%p** |
| 正결손 | 94 (17.22%) | 158 (27.15%) | **+64, +9.93%p** |
| archive 유실 | 147 (26.92%) | 13 (2.23%) | **−134, −24.69%p** |

1.6 유실 13행은 raw-J̄ 10행과 MA per-shell ledger 3행입니다. 正결손 증가는 과거 유실로 뭉쳤던 zero-flow/구조적 결손을 정확히 재분류한 결과입니다.

엄격한 CMFGEN 동일 수량 비교는 `79→89`행, `14.47→15.29%`로 **+10행, +0.82%p**입니다. 비동일 문맥 수치 9행을 포함하면 `88→98`행입니다. 커버리지 합계는 582이고 빈 결손 사유는 0건입니다.

### Fail-closed와 OFF

- C1 producer iter-10의 대상 72행을 iter-9로 이동:
  - 세 셀 모두 `frozen input load failed`
  - exit 1
  - oracle CSV 0개
- 추가로 s0 C1을 `24→23 bins`로 만든 시험도 exit 1
- OFF object:
  - 기본/`-ULUMINA_FROZEN_ORACLE` 모두 exit 0
  - `cmp=0`
  - 공통 SHA-256 `849c48bb…61a3d7`
  - `lumina_oracle|g_oracle` 심볼 각각 0개

제출 산출물은 [phase1_6 검증 디렉터리](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_6)에서 확인할 수 있습니다.