# Wave 3 명세 — D-5/SE 구조 수리: element-wide NLTE 파일럿

상태: **DESIGN ONLY — 구현 금지**  
작성일: 2026-07-31  
대상 단계: Wave 3 / 동등화 로드맵 v2 Stage 2A  
정본 범위: frozen-`n_e` 1셀 파일럿. 전 원소 global charge는 Stage 2B로 분리한다. 이 분리는 로드맵의 Stage 2A/2B 계약이다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:177-198`).

## 0. 결정문

Wave 3의 D-5 수리는 인접 ion pair를 더 반복하거나 damping하는 패치가 아니다. 한 원소의 S II–IV 또는 Fe II–IV에 속한 모든 활성 super-level(SL)을 한 벡터에 넣고, II→III와 III→IV의 bound-free(bf) 연결을 같은 선형계에 동시에 조립해 원소 총량 보존행 하나로 푸는 **element-wide statistical-equilibrium(SE) 파일럿**이다. 현행 pair 경로가 element-wide 단일 행렬을 잔여로 명시하고 있고(`src/lumina_plasma.c:16909-16920`), 공유 lo-ion을 저장했다가 뒤 pair 해를 복원으로 버리며(`src/lumina_plasma.c:16944-16994`), 감사는 이 구조가 상위 stage drain을 모르는 앞 pair를 최종 소유자로 만든다고 D-5로 확정했다(`docs/V3_FUNCTION_AUDIT_VERIFICATION_2026-07-31.md:185-194`, `docs/V3_FUNCTION_AUDIT_VERIFICATION_2026-07-31.md:489-510`).

방법 참조는 ARTIS다. ARTIS는 원소의 사용 ion 전부에 대해 하나의 차원을 만들고 ion별 bb와, 최상단을 제외한 각 ion의 ionization/NT/autoionization 항을 같은 행렬에 더한 뒤(`../artis-ref/nltepop.cc:1218-1247`), 한 행을 원소 총량 정규화로 바꾼다(`../artis-ref/nltepop.cc:1249-1260`). 최종 acceptance target은 ARTIS가 아니라 CMFGEN이다. Gate B 자체도 Lane A를 방법 reference, Lane C를 CMFGEN acceptance lane으로 분리한다(`docs/GATE_B_DUAL_ORACLE_SPEC.md:9-18`).

이 파일럿은 Stage 2A까지만 구현 대상으로 정의한다. `n_e`, `T_e`, 복사장과 원소 총밀도는 입력 스냅숏에서 동결하며, 전하행은 넣지 않는다. 모든 원소를 확장한 뒤 공유 `n_e` 전하식 하나를 결합하는 일은 Stage 2B다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:177-188`).

## 1. 범위

### 1.1 파일럿의 정확한 대상

1. 셀은 **s8 한 셀**이다. Gate B 최종 캡처가 s0/s8/s43을 소비 셀로 보존하며(`docs/CODEX_GATEB_PARITY59_AB_REPORT.md:44-60`), 이 명세는 그중 광구 셀 s8을 Stage 2A의 단일 판정 셀로 고정한다. 최초 Gate B 명세도 s8을 광구 셀로 정의했다(`docs/GATE_B_DUAL_ORACLE_SPEC.md:34-49`).
2. 원소는 서로 독립적으로 **S II–IV(Z=16, 내부 stage 1–3)**와 **Fe II–IV(Z=26, 내부 stage 1–3)**다. 로드맵이 이 두 원소를 Stage 2A의 필수 파일럿으로 지정한다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:177-183`); ARTIS 비교 진단도 같은 최소 대상을 요구한다(`docs/ARTIS_CMFGEN_PARITY_DIAGNOSIS_2026-07-30.md:426-435`).
3. 각 원소는 II–IV를 한 행렬로 푼다. S와 Fe를 한 행렬로 합치지 않는다. Stage 2A에는 원소보존행만 있고 원소별 전하행은 없다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:179-188`).
4. 미지수는 full level이 아니라 현행 CMFGEN model-atom의 활성 SL population이다. full-level 과정은 SL 행/열에 투영하고 해 뒤 현행 within-SL 분율로 복원한다. 이 선택은 CMFGEN 동등 정의가 모든 활성 SL을 미지수로 두고 full-level 과정을 투영·복원한다고 명시한 데 따른다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:46-55`); 현행 Lumina에도 SL 행렬과 full-level 재분배가 이미 있다(`src/lumina_plasma.c:16410-16422`, `src/lumina_plasma.c:16585-16593`).
5. 파일럿의 권위 경로는 CPU double-precision reference solve다. GPU 이식·성능 최적화·전 셸 생산 전환은 이 명세의 비목표다. 로드맵도 권위 코어는 CPU에서 먼저 완성하고 GPU는 동등성 뒤 활성화하도록 요구한다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:200-214`).

### 1.2 동결 입력 계약

동일한 s8 스냅숏에서 `T_e`, `n_e`, `J_nu`/`Jbar`, 원소 총밀도, Sobolev/escape 입력, 원자자료와 SL 분율을 pair-wise 기준선과 element-wide 후보가 같이 읽는다. Gate B의 목적 자체가 한 셀의 동결 `T_e,n_e,J,population`에서 생산 산술을 비교하는 것이다(`docs/GATE_B_DUAL_ORACLE_SPEC.md:7-16`). 입력 디렉터리, 소비 iteration, 셸, 원자자료 checksum과 effective gate manifest는 두 레인에서 같아야 한다.

Stage 2A는 다음 값을 갱신하지 않는다.

- `n_e`와 전하보존: Stage 2B 대상이다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:185-195`).
- `T_e`와 복사평형(RE): Stage 4 대상이다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:217-235`).
- `J_nu/Jbar` 생산자와 frequency coupling: Stage 3 대상이다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:200-214`).
- opacity/formal/packet fate의 최종 개선 판정: Stage 5–6 대상이며, ion fraction 하나가 스펙트럼을 자동으로 고친다고 가정할 수 없다(`docs/ARTIS_CMFGEN_PARITY_DIAGNOSIS_2026-07-30.md:256-270`).

### 1.3 확장 조건

다음 조건을 순서대로 만족해야 범위를 넓힌다.

1. **s8/S와 s8/Fe 구조 게이트:** §4.1–§4.3의 identity, topology, 보존, 잔차, 정칙성 검사를 둘 다 통과한다.
2. **경계 ion 활성 검사:** 제외한 인접 stage(I 또는 V)가 양쪽 자료의 합집합에서 이온분율 `>10^-8`이거나 rate/opacity/heating 기여 `>10^-4`이면 그 stage를 행렬에 포함하기 전에는 해 acceptance를 금지한다. 이는 로드맵의 active-set 규약이다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:70-78`). 즉 II–IV는 무조건 고정된 물리 절단이 아니라 s8에서만 허용되는 최초 창이다.
3. **역방향 축 검사:** s8 통과 뒤 동일 assembler를 s0 Fe II–IV와 s20 S II–IV에 shadow-run한다. 기존 지도에서 s0 Fe/Co/Ni 전 성분과 s20 S 전 성분이 앵커 역방향으로 이동했다(`docs/CODEX_ABS_STATE_60_OVERLAY.md:9-32`). 이 파일럿은 Fe와 S만 다루므로 s0에서는 Fe만 필수 판정하고, Co/Ni 회복 주장은 해당 원소가 element-wide로 확장될 때까지 금지한다.
4. **전 원소/전 셸:** S/Fe 및 역방향 축을 통과한 뒤에만 toy06 활성 원소와 전 셸로 확장한다. 그 뒤 공유 global charge equation을 붙이는 것이 Stage 2B다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:185-198`).

## 2. 제거할 현행 구조와 유지할 자산

### 2.1 파일럿 ON 영역에서 제거할 구조

현행 행렬은 `ion_idx_lo/ion_idx_hi` 두 stage만 받아 조립한다(`src/lumina_plasma.c:14479-14486`). bf 블록은 lo ion의 ionization energy 하나를 사용하고(`src/lumina_plasma.c:15406-15425`), 각 lo-level의 ionization/recombination을 `ground_hi`로만 보낸다(`src/lumina_plasma.c:15638-15650`). 따라서 II/III pair 안의 III는 III→IV drain을 같은 계에서 보지 못한다. 감사에서 지적한 옛 좌표 `plasma.c:14484-14487`의 의미도 이 단일 lo→hi 블록이며, D-5 기전은 감사문에 보존되어 있다(`docs/V3_FUNCTION_AUDIT_VERIFICATION_2026-07-31.md:185-194`).

파일럿 ON의 대상 `(Z,s)`에서는 아래를 전부 우회한다.

- pair 순회 및 pair별 보존행(`src/lumina_plasma.c:16944-16994`, `src/lumina_plasma.c:16333-16358`),
- 공유 lo-ion `saved_lo` save/restore(`src/lumina_plasma.c:16948-16993`),
- `pair_shares_slot`에 의한 per-ion pin과 후처리 rescale(`src/lumina_plasma.c:16489-16500`, `src/lumina_plasma.c:16595-16627`),
- pair owner 결과를 별도 opacity ion-stage writeback으로 고치는 경로. 이 함수는 기본적으로 환경변수가 없으면 즉시 return하고 공유 pair를 skip한다(`src/lumina_plasma.c:2624-2644`).

현재의 `LUMINA_TOPSTAGE_IV`는 element-wide 대체물이 아니다. 이 gate는 pair의 hi stage에 IV reservoir를 외부 Saha 값으로 두고 대각 sink와 RHS source를 넣는 방식이며(`src/lumina_plasma.c:15736-15775`, `src/lumina_plasma.c:15875-15895`), IV population 자체를 같은 미지수 벡터에서 풀지 않는다. 파일럿 ON 대상에서는 이 보정도 중복 적용하지 않는다.

### 2.2 재사용할 자산

다음 자산은 산술을 복제하지 말고 element-wide assembler의 입력으로 재사용한다.

- 현행 column-major 규약 `A(row i,col j)`(`src/lumina_plasma.c:14508-14519`).
- full-level→SL mapping과 within-SL Boltzmann fraction(`src/lumina_plasma.c:14511-14519`).
- bb radiative/충돌 rate 생산 함수와 실제 Omega 경로. 현행 bb 배치는 상향/하향을 off-diagonal inflow와 source-column diagonal outflow로 넣는다(`src/lumina_plasma.c:15003-15048`).
- per-level CMFGEN `sigma_bf` 격자(`src/lumina.h:428-438`).
- bf upper-target CSR과 route probability. v2 schema는 `offset/targets/probability`를 보유한다(`src/lumina.h:440-457`, `src/lumina_atomic.c:1095-1113`).
- 현행 Milne, collisional ionization/3-body inverse 산술. 현재 pair 블록의 배치 형태는 `src/lumina_plasma.c:15638-15707`에 있다.

## 3. 행렬 계약

### 3.1 행·열 identity

원소 `Z`와 셸 `s`에 대해 미지수 벡터를 다음 고정 순서로 만든다.

```text
x = [stage II의 SL 0..nII-1,
     stage III의 SL 0..nIII-1,
     stage IV의 SL 0..nIV-1]^T
N = nII + nIII + nIV
```

각 index는 `(matrix_index, Z, spectroscopic_stage, internal_stage, sl_id, anchor_global_level, member_full_level_ids, energy, g_or_SL_partition, source_atomic_checksum)`을 가진다. row와 column은 같은 identity table을 공유하며, 임의의 암묵적 `ground_hi=n_lo_super` 계산을 금지한다. 현행은 pair-local `ground_hi=n_lo_super`를 쓴다(`src/lumina_plasma.c:15412-15426`); ARTIS는 전 원소 index helper로 ion/level을 한 벡터에 배치한다(`../artis-ref/nltepop.cc:450-464`, `../artis-ref/nltepop.cc:574-585`).

identity manifest의 ion/SL/full-level/line/continuum/target 수와 ID는 비교 모델과 100% 일치하고 energy, `g`, threshold, mapping checksum 불일치가 0이어야 한다. 이것은 M-동등 공통 문턱이다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:56-68`). 하나라도 불일치하면 행렬 solve를 시작하지 않는다.

### 3.2 부호와 공통 배치 규약

미정규화 SE 계는 `A x = 0`이다. column `j`는 source state, row `i`는 target state balance다. 모든 물리 전이 `j -> i`의 비음수 rate coefficient `P(j->i)`는 반드시 다음 두 항을 한 함수에서 원자적으로 더한다.

```text
A[i,j] += P(j->i)      # i로의 inflow
A[j,j] -= P(j->i)      # j에서의 outflow
```

따라서 정규화행을 덮기 전 각 물리 채널의 column sum은 roundoff 안에서 0이어야 한다. 현행 bb 조립도 이 부호쌍을 사용한다(`src/lumina_plasma.c:15037-15048`), ARTIS bb와 bf도 source 대각에 음수, target off-diagonal에 양수를 넣는다(`../artis-ref/nltepop.cc:515-559`, `../artis-ref/nltepop.cc:587-615`).

각 채널은 별도 dense reference plane에 조립한 뒤 합한다.

```text
A_total = A_rad_bb + A_coll_bb + A_nt_bb
        + A_rad_bf + A_coll_bf + A_nt_bf
        + A_autoion_DR
```

ARTIS도 같은 계열의 rate matrix들을 별도로 보유하고 합산한다(`../artis-ref/nltepop.cc:54-119`). 파일럿에서 비활성인 항은 manifest에 `inactive`와 판정 근거를 남겨야 하며, 구현되지 않은 활성 항을 0으로 조용히 대체하면 실패다.

### 3.3 bound-bound와 충돌 항

1. bb 전이는 같은 ion stage 안에서만 연결한다.
2. 각 full-level 전이의 radiative upward/downward와 thermal collisional upward/downward rate를 계산하고, source full level의 within-SL fraction을 곱해 해당 source SL column에 투영한다. 현행 SL 조립은 source 쪽 fraction으로 흡수/방출을 투영한다(`src/lumina_plasma.c:15037-15048`).
3. 동일 SL 안의 두 full level 사이 전이는 총 SL population을 바꾸지 않으므로 두 부호항이 정확히 상쇄돼야 한다. 상쇄 뒤 남는 값이 channel column-sum tolerance를 넘으면 실패다.
4. 실제 collision-strength 표, fallback, floor 사용 여부는 transition별 provenance로 덤프한다. 이 파일럿은 새 Omega floor를 만들지 않는다. 레지스트리는 collision 자료 결손/floor를 별도 Stage 1→2 항목으로 두고 있다(`docs/CLAMP_FIX_PRIORITY_REGISTRY.md:45-47`).

### 3.4 level-resolved bound-free 항

각 lower full level `l`과 target CSR의 **모든** route `t`를 순회한다. route는 `(lower_global_level, upper_global_level, target_probability, threshold, sigma identity)`를 가져야 한다. 현재 데이터 구조가 v2 multi-target CSR을 보유한다(`src/lumina.h:440-457`); ARTIS도 lower level마다 모든 photoionization target을 순회하고 specific upper level index를 얻는다(`../artis-ref/nltepop.cc:563-590`).

lower SL `L=SL(l)`와 upper SL `U=SL(t)`에 대해 다음을 배치한다.

```text
P_rad_ion = p_t * Gamma_base(l, t)
P_col_ion = p_t * C_ion_base(l, t)
A_rad_bf[U,L] += f_l * P_rad_ion
A_rad_bf[L,L] -= f_l * P_rad_ion
A_coll_bf[U,L] += f_l * P_col_ion
A_coll_bf[L,L] -= f_l * P_col_ion

P_rad_rec = p_t * alpha_rad_base(t, l) * n_e   # 사용 rate 함수의 단위 계약을 manifest에 기록
P_col_rec = p_t * C_3body_base(t, l)
A_rad_bf[L,U] += f_t * P_rad_rec
A_rad_bf[U,U] -= f_t * P_rad_rec
A_coll_bf[L,U] += f_t * P_col_rec
A_coll_bf[U,U] -= f_t * P_col_rec
```

`f_l/f_t`는 full-level이 SL에 속할 때 source population을 full level로 내리는 투영 계수이며 identity SL이면 1이다. 여기서 `*_base`는 target probability를 아직 곱하지 않은 rate다. 생산 helper가 이미 target별 cross-section/probability를 포함한 rate를 반환하면 외부 `p_t`는 다시 곱하지 않고, manifest에 `probability_applied=inside`를 기록한다. ARTIS의 target별 collisional helper는 threshold cross-section에 target probability를 내부에서 한 번 곱하며(`../artis-ref/macroatom.cc:630-681`), ARTIS matrix는 그 반환값을 다시 가중하지 않는다(`../artis-ref/nltepop.cc:581-615`). route probability는 어떤 경로에서도 정확히 한 번만 적용돼야 한다.

이 규약으로 S II→III와 S III→IV, Fe II→III와 Fe III→IV가 한 번에 들어간다. III→IV가 별도 외부 reservoir/RHS가 아니라 IV population column과 연결되므로 D-5의 hi-ion drain 결손을 직접 제거한다. pair 구조의 lo→hi 한 블록과 ground-only routing은 `src/lumina_plasma.c:15406-15426`, `src/lumina_plasma.c:15638-15650`에 있다.

target map이 없거나 target이 행렬 밖이면 다음 중 하나만 허용한다.

- 제외 stage가 §1.3 active-set 문턱 아래임을 manifest로 증명하고 `inactive-boundary`로 분류한다.
- target stage/level을 행렬 범위에 추가한다.

ground target fallback, target clamp 또는 RHS reservoir로 조용히 바꾸는 것은 파일럿 PASS 경로에서 금지한다. target/lower/upper/SL mapping coverage 100%가 Stage 1 acceptance이기 때문이다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:160-173`).

### 3.5 보존행

모든 물리 rate를 조립한 뒤 **row 0 하나만** 다음 원소 총량 식으로 덮는다.

```text
sum_j x_j = n_element(s8)
A[0,j] = 1
b[0]   = n_element(s8)
b[i>0] = 0
```

departure/scaling 변수 `x_j = d_j y_j`를 쓰면 `A_scaled[0,j]=d_j`로 쓰되, 물리적으로 복원한 `x`의 합이 동일한 `n_element`여야 한다. ARTIS는 zeroth row를 전부 1로 채우고 RHS 0번에 원소밀도를 둔다(`../artis-ref/nltepop.cc:1249-1260`). Stage 2A는 원소별 전하행이나 stage별 보존행을 추가하지 않는다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:179-188`).

### 3.6 조건수·정칙성 대책

조건 대책은 해를 바꾸는 clamp가 아니라 동일 선형계의 scaling, pivoting과 refinement로 제한한다.

1. **사전 진단:** 정규화행 삽입 전 channel별 column-sum, zero row/column, 연결성 component를 기록한다. 정규화행 삽입 후 numerical rank가 `N`이어야 한다.
2. **ARTIS식 equilibration:** 최대 10회, 각 index의 row/column 2-norm에서 `f=sqrt(col_norm/row_norm)`를 구해 row에 `f`, column에 `1/f`를 곱하고 `|f-1|<=1e-3`이면 멈춘다. 해 복원 factor를 별도 보존한다. 이는 ARTIS의 실제 정규화 알고리즘이다(`../artis-ref/nltepop.cc:713-765`).
3. **reference solve:** scaled matrix를 partial-pivot LU로 풀고 최대 10회 iterative refinement한다. ARTIS도 partial-pivot LU 뒤 잔차 refinement를 최대 10회 수행한다(`../artis-ref/nltepop.cc:913-980`, `../artis-ref/nltepop.cc:1026-1096`).
4. **조건수:** SVD 기반 `kappa_2`와 solver의 reciprocal condition estimate를 모두 덤프한다. Wave 3 PASS는 `kappa_2 <= 1e12`, numerical rank `N`, pivot growth `<=1e8`로 사전등록한다. `1e12 < kappa_2 <= 1e14`는 **CONDITIONING FAIL**이며 permutation/scaling 원인분석만 허용한다. `kappa_2 > 1e14` 또는 rank 결손은 즉시 **SINGULAR FAIL**이다. 현행 주석은 raw cold-`T_e` matrix의 `cond~1e15`가 garbage를 만들고 변환 뒤 `4e3`으로 낮아진 사례를 기록한다(`src/lumina_plasma.c:16266-16275`); rate 기반 pin 뒤에도 `3.6e11`을 보고한다(`src/lumina_plasma.c:16305-16329`). 위 `1e12`는 이 두 실측 영역 사이에 둔 Wave 3 사전등록 한계다.
5. **금지:** negative-pop floor, LTE-relative repair, `b_k` cap, Boltzmann row anchor, stage-IV fixed reservoir, ion 제거 재시도는 PASS 경로에서 0회여야 한다. 현행에는 음수 floor/repair와 `b_k` cap 계열이 존재한다(`src/lumina_plasma.c:16504-16583`), 레지스트리는 이를 Stage 2→4 제거 군집으로 분류한다(`docs/CLAMP_FIX_PRIORITY_REGISTRY.md:20-21`, `docs/CLAMP_FIX_PRIORITY_REGISTRY.md:40-41`, `docs/CLAMP_FIX_PRIORITY_REGISTRY.md:71-73`).

## 4. Acceptance 정의

### 4.1 구조·수치 절대 게이트 — 즉시 판정 가능

아래는 S와 Fe 각각, s8에서 모두 만족해야 한다.

| 항목 | PASS 문턱 |
|---|---:|
| row/column identity, ion/SL/full-level/line/continuum/target coverage | 100%, checksum mismatch 0 |
| channel column-sum | `max_j |sum_i A_channel[i,j]| / max_i,j |A_channel[i,j]| <= 1e-12` |
| 정규화 뒤 rank | `N` |
| `kappa_2` | `<=1e12` |
| pivot growth | `<=1e8` |
| scaled SE residual | `<=1e-10` |
| 원소보존 상대오차 | `<=1e-12` |
| non-finite | 0 |
| `x_i/n_element < -1e-14` | 0 |
| clamp/floor/freeze/anchor/repair/fallback 발화 | 0 |
| index permutation 3종의 ion fraction 최대 절대차 | `<=1e-10` |

로드맵 Stage 2의 필수 잔차는 scaled SE `<=1e-10`, 원소보존 `<=1e-12`, permutation/hot-cold 동일해다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:190-197`). population이 음수/비유한이면 ARTIS도 solution validation 실패 대상으로 다룬다(`../artis-ref/nltepop.cc:796-858`).

### 4.2 ARTIS 방법 대조 — 즉시 판정 가능

같은 atomic projection, 같은 s8 frozen state와 같은 stage window를 양쪽 assembler에 주고, identity permutation을 적용한 뒤 다음을 비교한다. ARTIS는 방법 reference이지 최종 물리 oracle이 아니다(`docs/GATE_B_DUAL_ORACLE_SPEC.md:13-16`).

| 대상 | PASS 문턱 |
|---|---:|
| `N`, row/column identity, conservation row | 100% 동일 |
| 각 channel의 nonzero support | 100% 동일; support/sign mismatch 0 |
| 활성 양수 matrix/rate 항 `|log10(Lumina/ARTIS)|` | median `<=0.03 dex`, p95 `<=0.10 dex`, max `<=0.20 dex` |
| normalized residual-vector 상대차 | `<=1e-6` |
| 해의 ion fraction 최대 절대차 | `<=0.01` |
| 활성 SL population `|Delta log10 x|` | median `<=0.05 dex`, p95 `<=0.15 dex` |

rate와 population 문턱은 로드맵 공통 계약에서 가져온다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:56-68`). 한쪽만 0/음수인 항은 floor로 비율을 만들지 않고 support/sign mismatch로 실패 처리한다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:70-74`). ARTIS dump와 다르면 먼저 identity/target/rate provenance를 고치며 population을 튜닝하지 않는다.

### 4.3 pair-wise 대비 개선 — Wave 3의 임시 채택 게이트

현재 수렴 released-T CMFGEN 앵커가 없으므로, 아래는 **CMFGEN parity PASS가 아니라 구조 수리의 상대 개선 판정**이다. 로드맵은 현 fixed-T 계산을 staging checkpoint로만 허용하고(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:16-22`), released-T 앵커가 없으면 CMFGEN rate/J/state/spectrum PASS 선언을 금지한다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:140-156`).

동일 frozen input에서 pair-wise 기준 `p_pair`와 element-wide 후보 `p_elem`을 만든다. CMFGEN 조건부 snapshot `p_ref`에 대해 active stage별

```text
d_k(p) = |log10(p_k / p_ref,k)|
D(p)   = active stage의 d_k 동일가중 평균
improvement = 1 - D(p_elem)/D(p_pair)
```

를 쓴다. `p_ref,k` 또는 후보가 0/음수면 실패이며 floor를 쓰지 않는다. active stage 규약은 `docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:70-78`을 따른다.

**pair-wise 대비 개선 인정 조건은 다음 모두다.**

1. s8 S와 s8 Fe에서 각각 `improvement >= 25%`.
2. 각 원소의 active stage 중 어느 하나도 `d_k`가 pair-wise보다 `>10%` 증가하지 않음.
3. 지배 ion fraction의 CMFGEN 절대차가 pair-wise보다 줄고, 후보의 절대차 `<=0.05` 또는 ratio 오차 `<=0.10 dex` 중 하나를 만족.
4. 활성 `b_k`는 pair-wise 대비 median log-error를 `>=25%` 줄이며, 후보 자체가 median `<=0.05 dex`, p95 `<=0.15 dex`이면 “absolute provisional pass”로 별도 표기.
5. Gate B의 strict/context eligible 동일 행에서는 element-wide 전후 변화가 재현되고, 예상 효과가 `>=1%`인 표적만 “oracle-measured”로 부른다. Gate B는 1% 효과를 계량할 정밀도는 있지만(`docs/CODEX_GATEB_C_FINAL.md:42-50`), strict 99/582와 context 9행만 적격이다(`docs/CODEX_GATEB_C_FINAL.md:23-38`, `docs/CODEX_GATEB_C_FINAL.md:46-52`).

`25%`는 Gate B의 1% 분해능과 별개인 이 명세의 구조개선 최소효과 사전등록값이다. 1–24.999%는 “방향 일치/효과 부족”, 0 이하이면 “무개선/악화”로 기록하고 gate를 승격하지 않는다.

### 4.4 역방향 축 회복 — s8 뒤 필수

기존 조건부 지도에서 s0 Fe ratio는 `(7.87e4,114,0.962)`이고 세 성분 모두 역방향, s20 S ratio는 `(5.63e4,0.116,0.0147)`이고 세 성분 모두 역방향이다(`docs/CODEX_ABS_STATE_60_OVERLAY.md:9-22`). 이 지도는 앵커 비율 1과의 거리가 줄면 A, 늘면 R로 정의한 report-only 자료다(`docs/CODEX_ABS_STATE_60_OVERLAY.md:1-7`).

s8 구조 게이트 뒤 다음을 모두 만족해야 Wave 3를 “map recovery”로 부른다.

- **s0 Fe:** active II/III/IV 각각 `d_k(elem) < d_k(pair)`이고, `D`가 `>=25%` 감소한다. Fe IV처럼 이미 ratio 1에 가까운 성분도 악화 면제를 주지 않는다.
- **s20 S:** active II/III/IV 각각 `d_k(elem) < d_k(pair)`이고, `D`가 `>=25%` 감소한다.
- 둘 중 하나라도 실패하면 s8 파일럿의 구조 PASS는 유지할 수 있지만 production 확장과 “CMFGEN 방향 개선” 표현은 금지한다.
- s20은 Gate B 최종 셀 집합(s0/s8/s43)에 없으므로(`docs/CODEX_GATEB_PARITY59_AB_REPORT.md:44-60`), s20 결과를 Gate B oracle PASS라고 부르지 않는다. 동일 state-map comparator의 확장 진단으로만 기록한다.
- s0 Co/Ni는 이 S/Fe 파일럿의 acceptance가 아니다. 기존 지도에서 이들도 역방향이라는 사실만 보존하고(`docs/CODEX_ABS_STATE_60_OVERLAY.md:13-15`, `docs/CODEX_ABS_STATE_60_OVERLAY.md:32`), Co/Ni element-wide 확장 전에는 회복/악화 귀속을 하지 않는다.

### 4.5 최종 CMFGEN acceptance — released-T 앵커 뒤에만 판정

수렴 released-T CMFGEN steady self-run이 인증된 뒤에는 다음 절대 문턱을 적용한다.

- 지배 ion fraction: 절대차 `<=0.05`, ratio `<=0.10 dex`.
- 활성 SL/`b_k`: median `<=0.05 dex`, p95 `<=0.15 dex`.
- frozen-state 활성 rate: median `<=0.03 dex`, p95 `<=0.10 dex`, max `<=0.20 dex`.
- scaled residual-vector 상대차 `<=1e-6`; 향후 nonlinear 결합의 centered finite-difference Jv 상대차 `<=1e-4`.

이는 로드맵 공통 CMFGEN 문턱이다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:56-68`). fixed-T·미수렴 지도나 Gate B 17.01% coverage로 이 절의 PASS를 선언할 수 없다. 현 상태 지도도 사용 CMFGEN 값이 fixed-T이며 `MAXCH=3.46e3%`인 조건부 snapshot이라고 경고한다(`docs/CODEX_ABS_STATE_5154.md:12-24`).

## 5. 검증 프로토콜

### 5.1 산출물 계약

각 `(run_id,Z,s,assembler)`에 다음 파일을 만든다. 파일명은 구현자가 정하되 manifest에서 역할이 유일해야 한다.

1. `identity`: §3.1의 row/column identity와 checksum.
2. `matrix_raw`: conservation 전 channel별 matrix와 RHS.
3. `matrix_normalized`: conservation 뒤, scaling 전 matrix/RHS.
4. `matrix_equilibrated`: solve에 실제 투입한 matrix/RHS와 scale factors.
5. `solution`: raw solution, 복원 solution, ion totals, SL/full populations.
6. `diagnostics`: rank, singular values, `kappa_2`, rcond, pivots, pivot growth, refinement history, channel column sums, scaled residual, conservation residual, negative/nonfinite count, 모든 guard/fallback counter.
7. `provenance`: 각 nonzero의 channel, source/target identity, rate, units, field generation, target route, full→SL weight.

행렬 비교에 필요한 최소 목록은 dimension, row/column level identity, bf target mapping, rate coefficient/units, conservation row, residual, 최종 stage/level population이다(`docs/ARTIS_CMFGEN_PARITY_DIAGNOSIS_2026-07-30.md:299-309`).

### 5.2 네 갈래 대조

#### A. Gate B oracle 대조

- parity59 frozen s8의 동일 iteration/state를 사용한다.
- `strict-compared` 99행과 명시 context 9행만 수리효과 판정에 사용한다. collisional strict는 0행이고 전체 582행 중 비교 불가가 474행이므로 범위를 넓히지 않는다(`docs/CODEX_GATEB_C_FINAL.md:23-38`, `docs/CODEX_GATEB_C_FINAL.md:46-50`).
- Fe IV raw-J는 frozen topology에서 0행이므로 Jbar coverage PASS로 세지 않는다(`docs/CODEX_GATEB_C_FINAL.md:13-17`).
- 같은 입력 두 번의 dump와 comparator가 byte-identical이어야 한다. 기존 Gate B도 oracle 및 comparator 반복 재현성을 요구하고 달성했다(`docs/CODEX_GATEB_PARITY59_AB_REPORT.md:135-153`).

#### B. ARTIS 행렬 덤프 대조

- 같은 `Z,s8,stage-window,identity,atomic checksum,T_e,n_e,J`를 사용한다.
- ARTIS의 `rad_bb/coll_bb/ntcoll_bb/rad_bf/coll_bf/ntcoll_bf/autoion` plane과 Lumina plane을 identity permutation 뒤 비교한다. ARTIS plane 정의와 합산은 `../artis-ref/nltepop.cc:54-119`에 있다.
- 먼저 topology/support, 다음 rate coefficient, 마지막 solution 순서로 판정한다. target별 bf 배치는 `../artis-ref/nltepop.cc:563-618`, conservation은 `../artis-ref/nltepop.cc:1249-1260`을 정본으로 한다.
- ARTIS가 ion 범위를 줄여 재시도한 결과는 golden으로 쓰지 않는다. ARTIS에는 실패 시 top/bottom ion 제거 재시도가 있다(`../artis-ref/nltepop.cc:1291-1325`); 본 파일럿 golden은 동일 II–IV window를 유지한 성공 solve여야 한다.

#### C. pair-wise 전후 대조

- 같은 process gates와 같은 frozen inputs에서 legacy pair-wise와 element-wide를 각각 실행한다.
- pair-wise의 각 pair matrix/solution, save 전/restore 후 공유 stage population, element-wide solution을 함께 저장한다. save/restore 위치는 `src/lumina_plasma.c:16948-16993`이다.
- §4.3의 `D`, 25% 개선, stage별 악화, dominant fraction, `b_k`, eligible Gate B delta를 자동 표로 만든다.
- CE iteration 수나 damping 변화만으로 element-wide 개선을 대체할 수 없다. 현 코드도 강화된 CE 뒤 단일 element matrix를 잔여로 남긴다(`src/lumina_plasma.c:16909-16920`).

#### D. 정칙성·보존 진단

- 조립 직후 channel별 column sums와 graph connected components.
- conservation 전 rank `N-1`, conservation 뒤 rank `N` 기대를 기록하되, 실제 rank가 다르면 원인을 identity별로 출력한다.
- raw/equilibrated `kappa_2`, rcond, singular spectrum, pivot growth, refinement residual history.
- 원소보존과 각 SE row의 scaled residual. SE row scaling 분모는 `max(total inflow,total outflow,n_i/t_ref)`, 원소보존행은 원소 총밀도를 쓴다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:70-78`).
- stage/SL permutation 3종: canonical, stage-block reverse, deterministic checksum shuffle. 세 해가 §4.1 문턱 안에서 같아야 한다.
- hot/cold population seed로 rate 재구축이 달라지는 경로가 있으면 둘 다 실행해 같은 해를 요구한다. linear frozen-rate solve가 seed에 의존하면 lagged-population rate가 숨어 있다는 뜻이므로 실패다. 로드맵도 permutation 및 hot/cold 동일해를 요구한다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:190-197`).

## 6. 게이트 계획과 OFF 불변

### 6.1 제안 gate contract

```text
LUMINA_NLTE_ELEMENT_WIDE=0|1          # 기본 0
LUMINA_NLTE_ELEMENT_WIDE_Z=16,26      # ON일 때 허용 Z 목록
LUMINA_NLTE_ELEMENT_WIDE_SHELL=8      # 파일럿 기본 s8; 명시 없으면 ON 거부
LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0|1   # 기본 0: shadow solve/dump only
LUMINA_NLTE_ELEMENT_WIDE_DUMP=0|1     # 검증 빌드에서 1
```

실제 이름을 바꾸려면 A 차터 보고서에서 일대일 대응표를 남겨야 한다. 기본 OFF와 명시 OFF는 같은 분기를 타야 한다. Gate B의 신규 계기 계약도 default-OFF와 미설정 생산경로 byte identity를 요구한다(`docs/GATE_B_DUAL_ORACLE_SPEC.md:21-32`).

### 6.2 ON 동작

1. `ELEMENT_WIDE=1`, Z allow-list 일치, shell 일치일 때만 새 assembler를 호출한다.
2. `COMMIT=0`은 pair-wise 권위 결과를 유지하고 element-wide는 shadow buffer에만 쓴다.
3. `COMMIT=1`은 대상 `(Z,s)`에서 pair-wise solve/save-restore/per-ion pin/topstage reservoir/writeback을 호출하지 않고 element-wide 결과만 한 번 commit한다. 두 solver 결과를 평균·damp·혼합하지 않는다.
4. target coverage, rank, condition, residual, population validation 중 하나라도 실패하면 element-wide buffer를 commit하지 않는다. baseline pair-wise 결과는 보존하되 run verdict를 `EW_FAIL_FALLBACK_BASELINE`으로 강제하고 acceptance 표에서는 FAIL로 센다.
5. 대상 밖 원소·셸은 legacy 호출 순서와 산술을 그대로 유지한다.

### 6.3 OFF 불변 배터리

다음을 모두 만족해야 A 구현을 검토 대상으로 받을 수 있다.

- 환경변수 미설정과 `LUMINA_NLTE_ELEMENT_WIDE=0`의 표준 CPU 산출물 byte comparison `cmp=0`.
- 대상 밖 셸/원소의 표준 population/rate/tau CSV byte comparison `cmp=0`.
- OFF에서 element-wide allocation, target-map 강제 load, dump file, banner, counter, RNG 소비 0.
- 기존 pair count/order, CE iteration log, pair save/restore, legacy writeback 호출 횟수 불변.
- 같은 ON shadow run 2회의 모든 matrix/solution/diagnostic dump byte comparison `cmp=0`.

Gate B의 선례는 기본/명시 OFF object와 oracle 반복 산출의 byte identity를 검사했다(`docs/CODEX_GATEB_PARITY59_AB_REPORT.md:18-21`, `docs/CODEX_GATEB_PARITY59_AB_REPORT.md:135-153`). 본 runtime feature는 적어도 위 산출물 수준의 OFF 불변을 충족해야 한다.

## 7. A형 clamp 일괄제거 목록

“파일럿 ON lane에서 발화 금지”와 “저장소에서 삭제 가능”을 구분한다. 레지스트리는 개별 선행 패치를 금지한다(`docs/CLAMP_FIX_PRIORITY_REGISTRY.md:3-11`).

| 구분 | ID | Wave 3 처분 | 저장소 전역 제거 시점 |
|---|---|---|---|
| D-5 직접 군집 | **C64, C65** | element-wide 대상에서 Boltzmann anchor와 stage-IV `b_k` cap을 모두 금지하고 발화 0 요구 | Stage 2B 원소보존·global charge·동일해 관문 뒤 동시 제거 후보; relT 착지 후 확정 |
| singular/ill-conditioning 군집 | **C13, C14, C15, C16, C17, C19, C48** | element-wide PASS lane에서 floor/repair/cap/fallback 0 요구; 발화하면 FAIL | Stage 2B full-rank·conditioning·permutation·residual 뒤 후보화, Stage 4 clamp/floor/freeze=0 acceptance 뒤 일괄 제거 |
| 구조 잔여(비 clamp ID) | **D-5, G-1** | pair owner/save-restore와 per-ion pin을 element-wide 대상에서 소멸 | D-5/G-1 전역 퇴역은 Stage 2 확장과 global charge 뒤 |

C64/C65가 D-5 upper-stage-blind/continuum-drain 군집이라는 정본 매핑과 제거 관문은 `docs/CODEX_REGISTRY_BUILD_NOTE.md:44-54`에 있다. C13/C14/C15/C16/C17/C19/C48의 일괄 시점도 같은 표에 규정돼 있다(`docs/CODEX_REGISTRY_BUILD_NOTE.md:46-52`). 각 ID의 원장 위치와 Stage 매핑은 `docs/CLAMP_FIX_PRIORITY_REGISTRY.md:20-21`, `docs/CLAMP_FIX_PRIORITY_REGISTRY.md:31`, `docs/CLAMP_FIX_PRIORITY_REGISTRY.md:40-41`, `docs/CLAMP_FIX_PRIORITY_REGISTRY.md:51`, `docs/CLAMP_FIX_PRIORITY_REGISTRY.md:54`, `docs/CLAMP_FIX_PRIORITY_REGISTRY.md:71-73`에 있다. D-5와 G-1은 각각 Stage 2 element-wide/원소보존+global charge로 흡수된다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:296-315`).

C09/C22/C40/C44/C66은 이 수리 하나로 “소멸”했다고 세지 않는다. Stage 2 source coverage는 그 군집 수리의 시작일 뿐, state-side 제거에는 Stage 3 권위장과 Stage 4 global response가, C09/C22 formal legacy 제거에는 Stage 6 energy KA가 더 필요하다(`docs/CODEX_REGISTRY_BUILD_NOTE.md:46-49`).

## 8. 단계별 구현 순서

1. **계약 고정:** s8 frozen-input manifest, S/Fe model projection, internal↔spectroscopic stage map, target CSR, process inventory와 expected `N`을 생성한다. checksum mismatch가 있으면 중단한다. 근거 문턱은 `docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:56-78`이다.
2. **identity/indexer:** 원소별 II–IV canonical vector와 full→SL/target→SL map을 만든다. 아직 solve나 writeback을 연결하지 않는다.
3. **channel assembler:** bb, bf, thermal collision, 활성 NT/autoion/DR plane을 각각 조립하고 column-sum/topology self-test를 통과시킨다. ARTIS channel 구조는 `../artis-ref/nltepop.cc:54-119`와 `../artis-ref/nltepop.cc:478-710`을 따른다.
4. **보존·solver:** 단일 원소보존행, ARTIS식 equilibration, pivoted LU, iterative refinement, rank/condition/residual diagnostics를 붙인다. 정본은 `../artis-ref/nltepop.cc:713-765`, `../artis-ref/nltepop.cc:913-1096`, `../artis-ref/nltepop.cc:1249-1260`이다.
5. **s8 shadow:** `COMMIT=0`에서 S와 Fe dump를 만들고 ARTIS topology/matrix 대조와 pair-wise 상대 개선 표를 생성한다. 구조·수치 gate 실패 시 다음 단계 금지.
6. **s8 gated commit:** `COMMIT=1`에서 대상 element/shell의 pair/save-restore/pin을 완전 대체한다. 다른 셸/원소와 OFF 불변을 B가 검증한다.
7. **역방향 축:** s0 Fe와 s20 S shadow 비교를 수행해 §4.4를 판정한다. 실패하면 s8 파일럿으로 후퇴하고 production 확장을 금지한다.
8. **Stage 2B 준비:** 활성 인접 stage와 모든 원소/셸을 확장한 뒤 공유 charge equation 설계를 별도 발주한다. Stage 2B 전에는 `n_e` 동등이나 전 원소 clamp 제거를 주장하지 않는다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:185-198`).

## 9. Codex A/B/C 차터 초안

### Codex A — 구현 차터

목표: §1–§8의 Stage 2A CPU gated path만 구현한다.

- source 변경은 gate/parser, element identity/indexer, channel assembler, CPU solver/dump, 대상 호출 분기와 테스트 fixture에 한정한다.
- 기존 pair assembler를 복사해 두 벌의 rate 산술을 만들지 말고 공통 rate producer를 호출한다. 현행 재사용 자산은 §2.2다.
- S/Fe/s8 shadow가 먼저이며 `COMMIT=1`은 구조·수치 self-test 뒤에만 연결한다.
- target fallback, population floor, Boltzmann anchor, `b_k` cap, damping, empirical multiplier를 새로 만들지 않는다.
- 산출 보고서는 변경 file:line, gate 표, matrix identity/dimension, channel coverage, OFF 불변 예상, 미구현 process와 fail-closed 경로를 포함한다.
- A는 CMFGEN parity PASS나 spectrum 개선을 선언하지 않는다. released-T 앵커 전 금지 규약은 `docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:140-156`이다.

### Codex B — 검증 차터

목표: A의 설명이 아니라 source와 산출물을 기준으로 §4–§6을 재현한다.

- OFF 미설정/명시 0, 대상 밖 원소/셸, ON shadow 2회 결정론을 byte compare한다.
- S/Fe s8의 identity, topology, target coverage, channel column sums, conservation, rank/condition/pivot/refinement/negative/fallback counter를 독립 집계한다.
- 같은 frozen input의 pair-wise/element-wide/ARTIS 세 matrix를 permutation해 §4.2–§4.3 표를 다시 계산한다.
- Gate B는 strict/context eligible만 사용하고 Fe IV 0행과 비-compared 행을 PASS 분모에서 제외한다. 적격 제한은 `docs/CODEX_GATEB_C_FINAL.md:42-52`다.
- s0 Fe와 s20 S의 `D`와 stage별 방향을 독립 계산한다. s20을 Gate B oracle이라고 부르지 않는다.
- 실패 artifact도 삭제하지 않고 first offending row/column/transition identity를 남긴다.

### Codex C — 독립 리뷰 차터

목표: 구조 수리가 실제로 D-5를 제거했는지 적대적으로 판정한다.

- A/B 결론을 먼저 채택하지 않고 source에서 대상 `(Z,s)`가 pair/save-restore/pin/topstage reservoir를 전부 우회하는지 추적한다.
- III→IV의 모든 target route가 실제 IV unknown column/row에 연결되는지, RHS reservoir나 ground-only collapse가 남지 않았는지 검사한다.
- conservation row가 정확히 하나인지, 전하행이 원소마다 중복되지 않았는지 검사한다.
- condition/fallback gate가 실패를 숨겨 baseline이나 LTE population을 PASS로 세지 않는지 검사한다.
- ARTIS 비교가 같은 identity/atomic checksum/frozen state인지, CMFGEN final acceptance가 fixed-T/Gate B 제한을 넘지 않았는지 검사한다.
- 판정은 `PASS`, `PASS-WITH-SCOPE`(s8 구조만), `FAIL-TOPOLOGY`, `FAIL-NUMERICS`, `FAIL-ORACLE-SCOPE`, `FAIL-OFF-INVARIANCE` 중 하나와 first-failing file:line을 낸다.

## 10. 위험과 후퇴 기준

**[개정 2026-08-01, user 승인]**: s8 acceptance는 본 스펙의 EW 구조 판정에서 **분리·이관**한다. 근거: Γ 삼중대조(docs/CODEX_W3_GAMMA_TRIPLE_COMPARE_2026-07-31.md)가 s8 어긋남의 진범을 동결 MC 장 내용(Stage-3 생산자 결함)으로 확정 — 오염된 동결장 위에서의 acceptance는 원천 불성립. 처분: ①s8 어긋남은 "장 결함 원장 트랙"(Stage 3 폐합 시 재판정)에 등재 ②EW 구조 acceptance의 무게중심은 구조-지배 셀(s0 계열, M_V 창 포함; s20은 frozen 입력 확보 시) ③§4.3의 s8 25% improvement 사전등록은 그 암묵 가정(s8 지배결함=pair 구조)이 반증됐으므로 실효 — 기록은 보존.

| 위험 | 조기 신호 | 중단·후퇴 기준 |
|---|---|---|
| target graph 불완전 | CSR 없는 active lower level, 행렬 밖 upper target, support mismatch | coverage 100%가 아니면 solve 금지; stage/data projection으로 후퇴 |
| top/bottom boundary 절단 | 제외 I/V가 active-set 문턱 초과 | 해당 stage 포함 전 solution acceptance 금지 |
| rank/conditioning 실패 | zero component, `kappa_2>1e12`, pivot growth, refinement 정체 | clamp/anchor 금지; identity/rate/scaling 분석으로 후퇴. `>1e14` 또는 rank<N이면 hard stop |
| 음수/초열적 해 | `x_i/n_elem<-1e-14`, nonfinite, repair counter | commit 금지; raw artifact 보존 후 assembler/condition 원인 분석 |
| pair 산술 중복 | ON인데 pair/save-restore/topstage counter 발화 | FAIL-TOPOLOGY, 즉시 OFF 후 호출 그래프 수정 |
| frozen-`n_e` 국소 통과·global charge 실패 | s8 원소별 PASS지만 Stage 2B가 초기값 의존 | Stage 2A 결과만 유지, global charge 전면 재설계. 로드맵도 이 신호를 Stage 2B 위험으로 둔다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:360-375`) |
| 조건부 CMFGEN snapshot 과적합 | s8 ratio 개선이나 released-T에서 악화 | fixed-T 결과는 staging으로 강등하고 최종 PASS 철회(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:16-22`) |
| 역방향 축 악화 | s0 Fe 또는 s20 S의 active component 하나라도 `d_k` 증가 | s8 shadow 단계로 후퇴; production/전 셸 확장 금지 |
| OFF 오염 | 미설정 vs 0 또는 대상 밖 CSV `cmp!=0` | 변경 리젝; gate 경계 재설계 |
| 메모리 폭증 | dense `N^2` reference가 파일럿 자원 한계 초과 | identity/channel sparse dump를 유지하고 solver backend를 block/sparse로 전환; 물리 절단·SL cap으로 우회 금지. 로드맵은 global 단계의 dense 대형 할당을 금지한다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:224-235`) |

운영상 후퇴는 환경변수를 제거해 legacy pair-wise 경로로 돌아가는 것이다. 실패한 element-wide 해를 부분 commit하거나 legacy 해와 혼합하지 않는다. OFF에서 기존 결함이 의도대로 남더라도 그것은 회귀가 아니라 gate 계약이며, Wave 3 승격 여부는 ON의 명시된 범위에서만 판정한다. Gate B도 default-OFF 생산 불변을 필수 처분 규약으로 둔다(`docs/GATE_B_DUAL_ORACLE_SPEC.md:21-32`).

## 11. 최종 완료 체크리스트

Wave 3 Stage 2A는 아래가 전부 확인될 때만 완료다.

- [ ] s8 S II–IV 단일 matrix, s8 Fe II–IV 단일 matrix.
- [ ] 각 matrix에 원소보존행 정확히 1개, 전하행 0개.
- [ ] pair owner/save-restore/pin/topstage reservoir 중복 호출 0.
- [ ] full-level bb/bf/collision의 SL 투영과 bf upper-target coverage 100%.
- [ ] §4.1 구조·수치 문턱 전부 PASS, repair/fallback 0.
- [ ] §4.2 ARTIS topology/rate/solution 문턱 PASS.
- [ ] §4.3 pair-wise 대비 S/Fe 각각 25% 이상 개선.
- [ ] s0 Fe와 s20 S가 §4.4의 stage별 방향 및 25% aggregate recovery PASS.
- [ ] Gate B 결과는 strict/context eligible과 1%급 효과로만 한정.
- [ ] CMFGEN 최종 PASS는 released-T 앵커 전 `NOT-YET-ELIGIBLE`로 표기.
- [ ] C64/C65와 solver-failure A형 군집은 ON lane 발화 0; 저장소 전역 삭제는 §7 시점까지 보류.
- [ ] default OFF, explicit OFF, 대상 밖 원소/셸 불변 PASS.
- [ ] A 구현 보고, B 독립 수치 검증, C 적대 리뷰가 모두 같은 manifest/checksum을 인용.

이 체크리스트는 D-5를 “공유 lo-ion restore 한 줄 삭제”가 아니라, 원소-wide 방정식·target topology·보존·정칙성·oracle 범위를 함께 닫는 구조 수리로 정의한다. 로드맵도 반드시 0으로 수렴할 구조 어긋남에 pair-wise owner/save-restore/pin, continuum upper-target collapse, full-rate↔SL 불일치를 함께 올려둔다(`docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:323-355`).
