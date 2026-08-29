# VERDICT_SLB_DPX

## 머리말
- **단**: 판정 (fresh 컨텍스트)
- **계약 둘**: `docs/RUNG_SUPERLEVEL_B_2026-08-28.md`(SLB-*) · `docs/PREREG_DPROXY_JBAR_ATTRIBUTION_2026-08-29.md`(DPX-*)
- **판정자**: Codex fresh (운전석 판정 이탈 원칙 준수)
- **판독 출처**: `/gpfs/kjhan/lumina/slb/READOUT_SLB_DPX.md`
- **표기**: 판정식/판정식 입력값은 원문에서 재적용·재산출
- **판본 사슬**: `VERDICT_SLB_DPX.md`(base) → `VERDICT_SLB_DPX.md.rev1`(이번 개정)
- **개정 사유**: `§11`/`§9 G-4` 직행 재적용 및 `run_det_convergence_2026-08-08.slurm:223-231`/계약 충돌 기반 B-1 귀속 정리

## §계약별 판정(판정식 1:1)

### 1) SLB-*/§11 상위 분할
- **판정식**
  - `SLB-A`: `g_fail == 0 and hz == 0`
  - `SLB-PR`: `g_fail == 0 and hz > 0`
  - `SLB-F`: `g_fail > 0`  (어느 `G-*` 이름 있는 FAIL)
  - `SLB-X`: 위 어느 분기에도 안 듦
- **게이트/기대치 재적용**
  - G-0: PASS (preflight 통과)
  - G-S: PASS (selftest 통과)
  - G-RC: PASS (봉인 R-II 변조-사본 1바이트 훼손=이름 있는 FAIL 시연)
  - G-1: PASS (RUN_SEAL 무결성 통과)
  - G-2~G-4: 산정 대상. B-2에서 G-4 완주 사전조건 미충족 (`iter=0` 종료, `iter1` 미도달)
  - G-5~G-7: 미집행(런 산출 부재)
  - `E-1` 스테이징·구성: PASS (`G-1`/`G-2` 전건 충족)
  - `E-2` 배너: `ACTIVE(13015 FL -> 1173 SL across ions)`, `off(identity)=0`, `K=100: 21919 levels lumped`, `[A2-07] within-super-level blocked=0`
  - `E-3` 완주( R7 iter0/1 완료 ): B-2에서 완주 미충족 (`iter=0` FATAL, `summary complete=1` 부재)
  - `E-4` hazard 소멸: 미착지
  - `E-5` F1b 잔여: 미착지
  - `E-6` 결정론 미러(B-1≡B-2): 미착지
  - `E-7` 새 앵커 bake: 실행 없음(측정 항목이 아님)
  - `E-8` 봉인 사후 checksum: 미착지(사후 봉인 검사 경로 미실행)
- **판정 결과**
  - B-1: 게이트 전 런처 가드 실패(`DET_FLIGHT_FATAL ... capture is not armed`)는 `G-*` 판정식 전단 바깥이다
  - B-2: `stderr` 1,252행에서 `[A2-10][LINE-SATURATION-BLOCKED] reason=UPSTREAM_LINE_SCAN_INCOMPLETE ... complete=0`,
    `[R7][FATAL] lane=DET iter=0 rc=4`, `[A2-10][BLOCKED] event=R7_MATERIAL_UPDATE_BLOCKED` 확인으로 `G-4 완주` 미충족 → `g_fail=1`
  - 상위 규칙 적용 결론: **`SLB-F`**

### 2) DPX-*/§4 상위 분할
- **판정식**
  - `DPX-A`: `P-1` ∧ `P-2`
  - `DPX-B`: (`P-1` 미충족 또는 `P-2` 미충족) ∧ 축약비 `>= 10.0`
  - `DPX-C`: `2.0 < ratio <= 10.0`
  - `DPX-X`: 위 어느 판정에도 안 듦(예: B-2 미완주·짝 0·필드 부재)
- **사전등록 P-1~P-4 재적용**
  - `P-1`: Δn_upper 상이값 수(타깃 ≥800) → 짝 산출 대상 자체 미형성(짝 0)로 판정 불가
  - `P-2`: 축약비(≤2.0) → 짝 0으로 판정 불가
  - `P-3`: ρ(> +0.4107) → 짝 0으로 판정 불가
  - `P-4`: 73=SL 수 판독 항목 → SL id 미기록으로 직접 확인 미실행, 배너는 부분 참고 값만 존재
- **판정 결과**
  - P-1~P-3은 `B-2 iter=0` 사망 때문에 판정식 입력이 형성되지 않음
  - `P-4`도 관측 항목 미실행, 배너 `11.1:1`은 간접 지표일 뿐
  - 따라서 §4 상위 분할은 **`DPX-X`**

## §갈림 5건 결론

(a) **SLB 분기**
- `SLB-F` 는 **어느 G-* 이름 있는 FAIL**이 존재해야 성립
- `§11` `SLB-F = g_fail > 0` 을 직접 적용하면, B-2에서 `G-4 완주`가 미달되어 `g_fail=1`이 성립한다.
- 근거: `stderr`에서 `iter=0` 단계 `[R7][FATAL] ... rc=4`로 종료했고 `iter=1`, `summary complete=1`가 관측되지 않으며 `complete=0` 블록을 동반한다.
- `G-2~G-7` 미집행은 자동으로 `SLB-X` 사유가 아니다. `SLB-F`는 `g_fail`이 0인지 여부로 분기한다는 점(계약 원문 `§11`).
- 결론: **`SLB-F`**(게이트 실패 분기)

(b) **DPX 분기**
- `B-2`는 iter=0 사망 → 짝 0, `ROW 0행`
- 사전등록 예시가 명시한 `DPX-X` 조건(미완주·짝 0·필드 부재)에 부합
- 승격 전 대조군(R-II/R-III)은 참조 가능한 **판정 가능한 관측 자료**이나, 이것은 DPX 후속 판정의 교차대조로만 사용되고 `DPX` 분기 직접 산정값은 아님
- 결론: **`DPX-X`**, 또한 승격 후 대조 부재로 인해 `DPX-A/B/C` 확정은 유보 아님(직접 미착지)이고 분할 밖 고정

(c) **B-2 실패의 성격**
- `RADEQ_SIGN_MISMATCH`/`UNRESOLVED_CANCELLATION`/`RADEQ_FIXED_T_BUNDLE_BUILD_FAILED`는 `B-2`에서만 출현하고 R-II·R-III은 0건
- 그러나 이것만으로 “승격이 새로 만들었다”를 직접 단정할 자료는 없음
- 남는 설명축: 입력 분기 차이( B-2 vs pre-contrast ), 런루트/타이밍(단일노드 반복 없음), 스테이징 경로의 차이(사전 가드·실험 모드), 물리 사멸 지점의 차이
- 결론: 이번 단에서 물리 소득으로 확정하려면 추가 대조가 필요, 현재 단정 불가

(d) **B-1 귀속(계약/스테이징/런처)**
- `run_det_convergence_2026-08-08.slurm`의 `A210_Z2A_PROBE`는 `LUMINA_A210_SPRODUCER_CAPTURE=1`, `LUMINA_A210_INDEPENDENT_CAPTURE=1` 을 강제한다(223-231행).
- 계약 `§7-1`의 B-1 구성은 `ZO`/`R-II` 입력을 각각 `0/1`에 맞춰야 하고(`B-1`에서 capture=0 조건), `run` 스크립트 `A210_Z2A_PROBE` 입력-provenance(1~3의 봉인 타깃)와의 접합에서 계약 내부 비호환이 발생한다.
- 판독의 `B-1` 실패(`DET_FLIGHT_FATAL ... capture is not armed`)는 런처 불변량 집행이 아니라 계약 경로 상호배타 조건 충돌의 정합 동작으로 본다.
- 판정 효과: 귀속은 **런처 가드 유죄 아님**, 사유는 계약 내부 비호환, 즉 `SLB` 분기 라벨 자체를 가드 실패로 끌어내리지 않는다.

(e) **운전석 §2 주장(73 = SL 수) 지위**
- 판독 §1이 분명히 밝힌 대로, SL id 직접 로그 미포함으로 `P-4` 직접 항목은 미실행
- 허용 가능한 해석은 “`73`이 되는 군-수준 축약 구조라는 운영 가설”의 기록에 한정
- 과소/과대 금지 하의 상한: `73=SL`을 판정적으로 확정할 근거 부재(하한 0, 상한 미정) — 즉 **판정상 보수적으로는 미착지**
- 수치 보강 근거: 배너 전역 비 11.1:1, 승격 전 국소 비 21.9:1은 `73=SL` 직접 확인을 대체하지 못함

`(비차단 반영)` 
- (c) 과소/과대 정합: `73`은 관측 재현으로 증명된 값이나, 이번 단은 수리 제안 없음; B-2 사망 선의 추가 오프라인 기전은 `관측 기재`로만 남김.
- **E-8**: 판독의 `봉인 read-only` 사전·사후 모두 `rc=0`(사후 경로 미실행 지표 포함)임을 `E-8` 항목에 동시 기재해야 함.
- **E-2 충족 누락**: `ACTIVE(13015 FL -> 1173 SL across ions)` 및 `off(identity)=0`이 B-2에서 확인되나 본문에서 판정 라벨로 반영되지 않았음.
- **E-3**: `필연`은 판정 등급의 단정이 아니라, 이번 단에서 `G-4` 미완주를 기록하는 축으로 처리해야 함.

## §분기 확정
- **SLB 분기**: `SLB-F`
- **DPX 분기**: `DPX-X`

## §하지 않은 것
- 라벨 부여(`SLB-*`, `DPX-*`)는 판독에서 0건
- B-3 예비런: 조건이 `SLB-F-D`(G-7 미러 실패)인데 G-7 미집행이므로 자격 없음
- `REVIEW2_REPORT`/`REVIEW3_REPORT`에서 지적된 검수 보완이 남아 있으나 본 판정은 실측 대조로만 판단
- 봉인 라우트는 `read-only`로 운용되고, R-II/R-III 비교군의 d-proxy 항목은 판정 보조 관측으로만 사용
- 운전석이 사용자 지시로 정정한 `c19ccf4`(운전석 계약 정정)는 본 단 소유권 외 항목

## §부록 실행 명장
- `rg -n "§11\|SLB-A\|SLB-PR\|SLB-F\|SLB-X" docs/RUNG_SUPERLEVEL_B_2026-08-28.md docs/PREREG_DPROXY_JBAR_ATTRIBUTION_2026-08-29.md`
- `sed -n '330,390p' docs/RUNG_SUPERLEVEL_B_2026-08-28.md`
- `sed -n '455,485p' docs/RUNG_SUPERLEVEL_B_2026-08-28.md`
- `sed -n '1,220p' /gpfs/kjhan/lumina/slb/READOUT_SLB_DPX.md`
- `ls -1 /gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260829T101641Z_census_ion1 /gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260829T101641Z_probe_ion1`
- `rg -n "R7\]\[FATAL\]|LINE-SATURATION-BLOCKED|RADEQ_SIGN_MISMATCH|active|off \(identity\)" /gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260829T101641Z_probe_ion1/stderr.log /gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260829T101641Z_probe_ion1/stdout.log /gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260829T101641Z_probe_ion1/RUN_FOOTER.txt`

SLB 분기 = SLB-F
DPX 분기 = DPX-X
