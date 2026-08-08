## Q1 관대한 오판

있다. 네 PASS 중 SH-PUB만 그대로 유지할 수 있고, 나머지 셋은 적어도 “사전등록 게이트 전항 PASS”로는 증거가 부족하다.

- SH-GAMMA — PASS 보류.

  - [실측] Γ2는 배열의 “byte-동일”을 요구한다([RUNG_GAMMA_DEPOSITION_OWNER.md:131](</tmp/claude-10396/codex_audit/rungs/RUNG_GAMMA_DEPOSITION_OWNER.md:131>), [136](</tmp/claude-10396/codex_audit/rungs/RUNG_GAMMA_DEPOSITION_OWNER.md:136>)). 그런데 판정은 순서와 셸 위치를 버린 514개 수치 다중집합만 비교했다([GAMMA_VERDICT.md:96](</tmp/claude-10396/codex_audit/verdicts/GAMMA_VERDICT.md:96>)). 값 교환을 잡지 못하므로 Γ2-b 판정은 무효다.
  - [실측] Γ1이 요구한 NC1~NC4 중 NC2·NC4가 미실행이고, Γ4도 미실행이다([GAMMA_VERDICT.md:124](</tmp/claude-10396/codex_audit/verdicts/GAMMA_VERDICT.md:124>)). 따라서 전체 단 PASS도 사전등록보다 관대하다.
  - [실측] 다만 코드에는 epoch 검사와 매니페스트 재검산, 이중발행 거부가 실제로 존재한다([lumina_plasma.c:16418](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:16418>), [16461](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:16461>)). 즉 소유권 설계가 틀렸다는 판정은 아니고, 폐합 증거가 부족하다는 뜻이다.

- DET-R6 — 계약 구현은 부정되지 않지만 전체 PASS는 보류.

  - [실측] “물리 줄 157/157 동일”은 Γ2의 다중집합과 동일한 결함이라고 단정할 수 없다. 판정문은 전체 로그 diff에서 14개 비물리적 혼입만 분리했다고 하므로, 필터 후 순서를 보존했다면 위치 교환을 잡을 수 있다([R6_VERDICT.md:49](</tmp/claude-10396/codex_audit/verdicts/R6_VERDICT.md:49>)).
  - 그러나 사전등록은 MC “byte-parity”다([RUNG_R6_DETERMINISTIC_LINE_JBAR.md:75](</tmp/claude-10396/codex_audit/rungs/RUNG_R6_DETERMINISTIC_LINE_JBAR.md:75>)). 필터된 157줄 동일은 그보다 약하다. stdout/stderr를 분리해 결정적 산출물을 순서·바이트로 비교했어야 한다.
  - 더 직접적으로 R6-3은 N6-1~N6-4 전항을 요구하지만 N6-2·N6-3이 미실행이다([RUNG_R6_DETERMINISTIC_LINE_JBAR.md:74](</tmp/claude-10396/codex_audit/rungs/RUNG_R6_DETERMINISTIC_LINE_JBAR.md:74>), [R6_VERDICT.md:65](</tmp/claude-10396/codex_audit/verdicts/R6_VERDICT.md:65>)).

- SH-GRID — 핵심 격자 계약은 PASS 가능하지만 B-6 폐합은 아니다.

  - [실측] B-6은 격자 변경이 실제 값을 “어디서 얼마나” 움직였는지 재는 항목이다([RUNG_GRID_CONTAINMENT_CONTRACT.md:209](</tmp/claude-10396/codex_audit/rungs/RUNG_GRID_CONTAINMENT_CONTRACT.md:209>)).
  - [실측] 판정은 canonical 빈별 값이 표면화되지 않았음을 인정하고 측정을 유보했다([GRIDB_VERDICT.md:33](</tmp/claude-10396/codex_audit/verdicts/GRIDB_VERDICT.md:33>)). 이는 정당한 유보가 아니라 미측정이다. 차원이 4000→3866으로 바뀌므로 공통 BF 격자에 양쪽을 투영한 실제 장, 적분 모멘트 또는 위치 결박 매니페스트를 비교해야 했다.
  - B-2 왕복 자가검사는 구조를 검증하지만 실제 생산 런의 장 이동량을 대신하지 않는다. 따라서 “핵심 포함 계약 PASS, B-6 미폐합”으로 고쳐 적는 것이 맞다.

- SH-PUB — PASS 유지.

  - [실측] 실제 실패에서 `(T_e, generation)` 보존을 보았으므로 같은 결함을 다시 주입할 필요는 없다([R7_VERDICT.md:17](</tmp/claude-10396/codex_audit/verdicts/R7_VERDICT.md:17>)).
  - [실측] 구현도 후보 publication에만 쓰다가 전 셸 성공 후에야 공개 `T_e`를 복사한다([radeq_publication.c:19](</tmp/claude-10396/codex_audit/src/radeq_publication.c:19>), [23](</tmp/claude-10396/codex_audit/src/radeq_publication.c:23>)); 호출부 역시 실패 시 세대를 올리지 않는다([lumina_plasma.c:12443](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:12443>), [12461](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:12461>)).
  - 이후 실제 `RADEQ_NO_BRACKET`에서도 보존됐다는 추가 관측도 있다([GRIDB_VERDICT.md:60](</tmp/claude-10396/codex_audit/verdicts/GRIDB_VERDICT.md:60>)).

## Q2 SH-RADEQ 진단

### 맞는 것

- [실측] 현 솔버는 모든 셸에 `[10,10^7] K`를 고정하고([lumina_plasma.c:12438](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:12438>)), 양 끝 부호가 같으면 내부를 한 번도 탐색하지 않고 `NO_BRACKET`으로 끝낸다([radeq_publication.c:20](</tmp/claude-10396/codex_audit/src/radeq_publication.c:20>)).
- [실측] 현 잔차에서 bf 방출은 `T^{-1/2}`, ff 방출은 `T^{1/2}`로 기준 발행값을 외삽한다([lumina_plasma.c:12329](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:12329>)).
- [실측] adiabatic 냉각은 `T`에 비례하고([lumina_plasma.c:12358](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:12358>)), Compton도 `T_rad-T`이므로 고온에서 냉각 쪽 선형항이 된다([lumina_plasma.c:12352](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:12352>)).
- [실측] CMFGEN BFCR은 명시적으로 `(ν-edge)/ν` 가중량이다([prrr_sl_v6.f:102](</tmp/claude-10396/codex_audit/cmfgen/prrr_sl_v6.f:102>), [107](</tmp/claude-10396/codex_audit/cmfgen/prrr_sl_v6.f:107>)). Lumina는 `eta_bf` 전체 에너지를 적분한다([lumina_plasma.c:12329](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:12329>)). 두 정의는 같지 않다.
- [판정] 따라서 세 표적 중 “T 외삽”과 “BF 냉각 정의”는 실제 원인 후보로 타당하다. 전역 끝점 이분법 또한 현재 실패의 직접 원인이다.

### 틀린 것

- [판정] “양 끝이 음수이므로 구간 어디에도 근이 없다”는 수학적으로 틀리다. 양 끝이 음수여도 중간에서 양수가 되어 두 근을 가질 수 있다. 증명되는 것은 “현재 이분법이 이 구간으로 bracket을 만들 수 없다”뿐이다.
- [판정] `T→0` 음의 발산은 셸별 `∑eta_bf Δν > 0`일 때만 성립한다. a209는 `eta_bf=0`을 합법적 exact-zero로 허용한다([lumina_plasma.c:8251](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:8251>)). 따라서 모든 셸에 대한 구조적 명제는 아니다.
- [판정] `T→∞` 음의 발산도 적어도 `n_e>0`, 유한 양의 explosion time 등의 조건이 필요하다. 솔버 자체는 `n_e=0`을 허용한다([radeq_publication.c:20](</tmp/claude-10396/codex_audit/src/radeq_publication.c:20>)).
- [판정] 그러므로 GRID 판정의 “그 범위 어디에도 에너지 균형이 없다”와 “첫 정직한 물리 실패”는 과장이다([GRIDB_VERDICT.md:54](</tmp/claude-10396/codex_audit/verdicts/GRIDB_VERDICT.md:54>)). 현재 잔차 정의 자체가 아직 CMFGEN과 다르다.

### 빠진 표적

- [실측] photo 항이 비음수 사건 흡수율이 아니라 음수가 될 수 있는 signed net `chi_bf`를 사용한다([lumina_plasma.c:12325](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:12325>), [8221](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:8221>)). 그 결과 `photoionization_rate`조차 음수가 될 수 있다. CMFGEN PR은 양의 흡수 가중 `WSE·HN·JPHOT`으로 구성된다([prrr_sl_v6.f:124](</tmp/claude-10396/codex_audit/cmfgen/prrr_sl_v6.f:124>)). BF 문턱 초과 가중과 함께 우선 감사할 부호·정의 문제다.
- [실측] trial T에서 바뀌는 것은 두 제곱근과 Compton·adiabatic뿐이다. population, `n_e`, opacity, line emissivity는 발행 당시 값으로 고정된다([lumina_plasma.c:12301](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:12301>), [12325](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:12325>)). “단면적·LTE population 재평가” 문제는 bf만이 아니라 ff 스펙트럼과 line source까지 포함해 점검해야 한다.
- [실측] `NONTHERMAL`은 항상 exact-zero이고, collisional-line은 항상 not-applicable이다([lumina_plasma.c:12357](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:12357>)). 방사선 line owner가 이를 정당하게 대체하는지는 별도 항별 CMFGEN 대조가 필요하다.
- [추정] adiabatic 항이 총 입자압이 아니라 `n_e`만 사용한다([lumina_plasma.c:12358](</tmp/claude-10396/codex_audit/src/lumina_plasma.c:12358>)). 단위는 맞지만 입자 census가 맞는지는 제공된 CMFGEN 두 파일로 판정할 수 없다.

주의: `prrr_sl_v6.f:126-129`는 BFCR 조립은 보여 주지만 그 입력 배열이 trial T마다 어디서 재계산되는지는 보여 주지 않는다. “CMFGEN이 단면적·LTE 인구를 매 trial 재적분한다”는 강한 설명은 이 파일만으로는 확인 불가다.

## Q3 SH-R1 잣대

결론 “이 모델에는 CMFGEN 회색 해가 존재하지 않는다”는 타당하지 않다.

- [실측] `COMP_GREY_V4`에는 `FIX_T` 검사가 전혀 없다. 호출되면 즉시 `COMPUTED=.TRUE.`로 두고 시작 표지를 출력한다([comp_grey_v4.f:88](</tmp/claude-10396/codex_audit/cmfgen/comp_grey_v4.f:88>)).
- [실측] 성공 경로는 moment 해에서 `TGREY`를 계산한다([comp_grey_v4.f:270](</tmp/claude-10396/codex_audit/cmfgen/comp_grey_v4.f:270>)). 일부 분기만 하위 solver가 `COMPUTED=.FALSE.`를 반환하면 중단한다([comp_grey_v4.f:200](</tmp/claude-10396/codex_audit/cmfgen/comp_grey_v4.f:200>), [238](</tmp/claude-10396/codex_audit/cmfgen/comp_grey_v4.f:238>)).
- [실측] 제공된 허용 파일에는 `FIX_T`나 `COMP_GREY_V4` 호출자가 없다. 따라서 `FIX_T=T`가 원래 호출을 생략시키는지는 이 자료로 판정할 수 없다.
- [판정] 기준 런의 호출 0회는 “그 런에서 계산하지 않았다”는 증거일 뿐, 회색 해 부존재의 증거가 아니다. 나머지 네 런도 NaN·negative opacity 이후 실패했으므로 유효한 물리 입력에서의 부존재를 증명하지 못한다.
- [판정] `FIX_T=F` 재실행은 호출 조건을 밝히는 유용한 실험이지만, 먼저 caller가 `FIX_T`에 따라 회색 solver를 우회하는지 확인해야 한다.

현재 증거로는 “CMFGEN도 T가 주어지면 회색 IC를 만들지 않는다”가 유력한 가설일 뿐이며, 확정은 불가다. 만약 caller 확인으로 사실이라면 SH-R1의 CMFGEN 동종 잣대는 “회색 해 생성”이 아니라 “주어진 T 프로파일 소비”가 된다.

## Q4 발자국 규약 적용

기본 경계는 적절하다. 명명 규약도 DET에는 적용, MC에는 비적용, SH에는 대응물이 있을 때 적용한다고 정확히 적고 있다([RUNG_NAMING.md:7](</tmp/claude-10396/codex_audit/rungs/RUNG_NAMING.md:7>)).

과소적용:

- SH-RADEQ는 CMFGEN 대응물이 명백한 공유 기계인데, BF 냉각 정의·trial-T 처리·선형화 순서가 다르다. 이 셋은 발자국 규약의 정당한 적용 대상이다.
- 특히 BFCR의 문턱 초과 에너지 정의는 CMFGEN 코드에서 직접 확인되므로 고치는 것이 정당하다.

과적용 또는 미확인 적용:

- `FIX_T=T`에서도 Lumina가 반드시 CMFGEN 회색 IC를 만들어야 한다는 주장은 과적용일 수 있다. CMFGEN caller가 그 조건에서 회색 solver를 원래 건너뛴다면 대응물 자체가 없다.
- “CMFGEN이 Newton을 쓴다”는 설명은 제공된 두 CMFGEN 파일에서는 확인되지 않는다. `comp_grey_v4.f`는 회색 moment solver이지 복사평형 T 보정 solver가 아니다.
- 단순히 이분법을 Newton으로 바꾸는 것만으로는 발자국 복제가 아니다. CMFGEN의 residual, 선형화 변수, population·charge coupling, 갱신 순서를 함께 맞춰야 한다.
- MC 사건 측도·패킷·추정자를 CMFGEN 비적용으로 둔 것은 적절하다.
- “동종 대 동종” 잣대도 적절하지만, 같은 이름의 출력이 아니라 같은 입력 상태·물리 정의·호출 조건을 먼저 확인해야 한다.

## Q5 순서

`SH-R1 전체 → SH-RADEQ 전체` 순서는 권하지 않는다. 현 전역 bracket 실패는 seed를 개선해도 구조적으로 해소되지 않을 가능성이 높고, SH-R1의 CMFGEN 잣대도 아직 자격이 없다.

권고 순서는 다음과 같다.

1. SH-RADEQ 정의 감사: 동일 셸·동일 현재 T에서 CMFGEN과 항별로 BF/FF/photo/line/adiabatic residual을 맞춘다.
2. SH-R1 잣대 자격 확인: CMFGEN caller의 `FIX_T` 분기와 유효 입력에서의 `COMP_GREY_V4` 호출을 확인한다.
3. 자격이 확인되면 SH-R1로 local-Newton에 쓸 물리적 초기 상태를 마련한다.
4. SH-RADEQ solver·선형화 폐합: 현재 T 주변의 CMFGEN식 갱신으로 끝낸다.

두 단을 반드시 통째로 순서화해야 한다면 SH-RADEQ가 먼저다. 다만 solver 활성화는 SH-R1 확인 뒤로 두는 것이 안전하다.

## 총평

가장 위험한 것은 `NO_BRACKET`을 “물리적 근 부재”로 읽는 것이다. 현재는 CMFGEN과 다른 residual을, 양끝 같은 전역 구간만 보고 거부한 상태다. 여기서 solver만 바꾸면 그럴듯하지만 잘못된 `T_e`를 발행할 수 있다. 먼저 BF의 문턱 초과 정의와 signed `chi_bf` 사용부터 폐합해야 한다.