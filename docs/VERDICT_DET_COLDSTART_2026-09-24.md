# VERDICT — DET-COLDSTART (2026-09-24) · 분기 **PIN-FAIL**

- 계약: `docs/PREREG_DET_COLDSTART_2026-09-24.md` (개정 8, 97e938b) · 판독기 동결 33b3635 (sha 5b641554…, referents 5910685e…; `/gpfs/kjhan/lumina/cmfgen_map/coldstart_20260924/JUDGE_FROZEN.txt`)
- 런: job **402180** (h100 syn09, 2 GPU), run root `/gpfs/kjhan/lumina/cmfgen_map/coldstart_20260924/run1`, 12:33 시작 · 모델 경과 2931 s = 48:51(RUN_FOOTER elapsed_seconds; sacct 잡 경과 48:57) · model_rc=1 · terminal_state=MODEL_FAILED. 바이너리 f585e30a…
- 판독기 산출: `…/coldstart_20260924/judge_run1/coldstart_result.json` — **branch = PIN-FAIL**, G-RUN code=PIN-FAIL path=SEED_FATAL k=0 G=∅ problems=[] · **G-ID PASS**(봉인 deck 55·topion 3·sigma·fixed_te, source diff = 변경집합 4 C파일, exports 델타 2키, 공개 사본 6, footer 바이너리 sha 일치).
- 판정자: 운전자(개정25). 분장 장부 = 사전등록 §6 집행 기록.

## 1. 판정

§5 분기표 PIN-FAIL 행의 조건 = G-ID PASS ∧ §4 G-RUN 고정 창 실패 식별. 실측:
- stderr `[COLDSTART] current_iter=0 armed=1 n_pin=2`(SEED, 1회) → `[A2-INIT][SEED-BUNDLE-BLOCKED] … solve_stage=PAIR_SOLVE ion_status=NEGATIVE_POPULATION` → `[A2-INIT][FATAL] lane=DET event=SEED_MATERIAL_PREDICTOR rc=4`.
- 개정 5 등식: SEED FATAL ⟹ G=∅ — ion.csv 0개·준위 증인 파일 0개·LEVELS 태그 0개·applied.csv 없음(SEED 커밋 전 반환). 초과·누락 없음.
- G-INJ(a)/(b)·G-COLD 는 **미평가 — SEED 커밋 전 종료**(개정 5 가 SEED FATAL 에 G=∅ 를 예정; 입력 파일 부재). 독립 위반 관측 아님. 종료감사 두 겹 모두 이 해석을 §4·§5 와 정합으로 판정; 단 판독기 :832–834 의 PIN-FAIL 시 G-INJ·G-COLD 일괄 면제는 선행 커밋이 있는 다른 실패 경로에서 독립 위반까지 면제할 수 있어 일반 규칙으로 쓰지 않는다(기재). X 행 「독립 게이트」는 「독립 신원 게이트(G-ID)」로 좁히는 문면 개정 권고(Fable F7).
- 주입값 독립 재현(Fable 종료감사): 공개 Si 19.480 블록 ln v 보간·재규격화 → sh4 n_II 23.1557999846, n_III 1.001557744e7 = 로그 lo/hi_target(상대 1e-14) ⟹ 실패 셀에서 주입은 계약대로 실행됐다(applied.csv 부재 공백을 메움).

**⟹ 분기 PIN-FAIL.** 뜻(계약 문면): 고정 경로에서 솔버가 수용 가능한 해를 내지 못했다 — 비음수 해 부재의 증명이 아니고 「차가운 상태 불가」 물리 주장도 아니다. **S II/III 쌍안정 가설에 대해 이 런은 아무것도 말하지 않는다**(r·분류·소스 증인 모두 미정의).

## 2. 실패 진단 (계약 「다음」 열: 셸·이온·lo/hi 목표 대 solved 합)

`[A2-INIT][SEED-BUNDLE-BLOCKED]` 행(stderr.log:120) 실측:

| 항목 | 값 |
|---|---|
| 실패 쌍 | **Si II–III (Z=14, ions 1:2), shell 4** (v_mid ≈ 7,180 km/s — IGE 코어, J 밖, S 아님) |
| 경로 | solve_stage=PAIR_SOLVE · ion_lock=1(armed OR) · ce_iteration=1/5 |
| 목표 대 해 | lo_target 23.1558 = solved_lo_sum 23.1558 · hi_target 1.00156e7 = solved_hi_sum (두 보존 합 부동소수 정밀도 내 충족, 상대 3e-16·2e-16) |
| 음수 | negative_count=62, 최소 −0.012160 (super-level 성분 100 = Si II 25 준위 묶음, 대표 E=15.671 eV, g=8), negative_relative = 1.22e-9 (= |min|/max 1.0e7) |
| 선형계 | rank 202, backward error 6.9e-17, pivot growth 1.0, 반복 개선 2회 |

**기전 — 두 가설 병립(판별 미완; 종료감사 정정)**: 고정 창은 준위 해를 legacy 두 보존 행 경로로 보낸다 — 마지막 bb-연결 lo·hi 행 두 율 방정식을 Σn_II = n_lo·Σn_III = n_hi 로 교체(lumina_plasma.c:20594–20610), generator(GTH) 경로 OFF, dense LU(평형화·반복개선), `b<0` 무허용 fail-closed.
- **가설 N(수치 오차)**: 미소 성분이 반올림으로 부호 반전. ⚠초판의 「음수 크기 1.2e-9 = 오차 규모」는 철회 — `negative_relative` 는 |min|/max 정의일 뿐 측정 오차가 아니다(순환). backward error 6.9e-17 은 부호 있는 해의 잔차이지 정확해와의 거리·양수성 증명이 아니다. 최소 음수는 lo 총량 대비 5.25e-4.
- **가설 S(정확해 음수)**: 두 율 방정식을 총량 제약으로 교체한 정방계는 비음수 보장이 없다(astra 3×3 반례: 총량비 1 에서도 음수). 해공간 = 자연 정상 분포 π + 교체 행 소스/싱크 응답 ψ; 강제 분할이 자연 분할과 다르면 ψ 성분이 교체 행 인접 고준위를 음수로 민다. 관측 서명과 합치: 음수 62개(super-level 성분)가 Si II 고준위 쪽(최악 super-level 100 = 25 준위 묶음, 대표 15.67 eV, IP 16.35), 크기 ~1e-2.
- **강제 분할 대 코드 자연 분할(Fable 실측)**: run1 `lumina_pop_djdn.csv` epoch 1(주입 직전 SEED 율 해 분할) sh4 Si x_star = 901.7 대 주입 x_pub = 4.3253e5 — **480×**. base 398939 sh4 준위합 x(gen 3) = 675(단계 배열 3.5e4 와 52× 불일치).
- ⚠**초판 철회**: 「같은 lo/hi 비를 base 는 통과했다 ⟹ 실패는 차가운 값 때문이 아니다」 — base 는 그 단계비를 준위 해에 강제한 적이 없고(보존 행 1 + GTH 로 자기 분할 선택), 인용한 ≈4e-6 은 iter 5 값이다(SEED 2.86e-5). 다른 알고리듬·다른 문제를 같은 문제로 읽었다. 목표 총량과 남은 율 방정식의 조합 자체가 원인일 가능성이 열려 있다.
- 셸 커버리지: sh0–3 은 Si 총량 0 ⟹ sh4 가 첫 살아있는 Si 셸이고 OpenMP 셸 루프는 최저 실패 셸만 기록 ⟹ 잠금 경로가 어느 셸·쌍에서든 통과했다는 증거 0. 쌍 루프는 순차(Si 쌍 실패 시 S 쌍 미도달).
- 비용: GPU 49 분(2×H100). 증거 위태 없음(run root 신규, 봉인 입력 무접촉 — G-ID PASS).

## 3. 주장 상한

- 말하는 것: 등록된 고정 경로가 SEED 에서 수용 가능한 해를 내지 못했다(첫 실패 Si II–III sh4, super-level 음수 62, 최소 −0.0122); S 유지·복귀 통계는 생성되지 않았다.
- 말하지 않는 것: S II/III 차가운 가지의 유지/복귀 · 쌍안정 유무 · 음수가 반올림(N)인지 정확해(S)인지 · 「차가운 값 때문이 아니다」 · 다른 셸·쌍의 고정 경로 거동.

## 4. 다음 (user 결정)

**user 결정(2026-09-24)**: 안 A(pin 을 S 쌍·J 셸로 좁힘) + **오프라인 확인 먼저**, 캡처런 (i) 승인.
⚠**A 의 위험 평가 정정(초판 「재현될 이유는 약하다」 철회; §6 에서 확정)**: 가설 S 아래에서는 S 쌍도 같은 실패를 낸다 — sh24 주입 x_pub 0.744 대 코드 SEED 자연 분할 x_star 81.4(base djdn epoch 1) = **110× 어긋남**(반대 방향, 결손은 S III 고준위 쪽). 총량비 O(1) 은 가설 N 에서만 뜻이 있다. 가설 S 가 맞으면 **두 보존 행 pin 은 원소·셸을 좁혀도 성립하지 않는 설계**이고 B(스케일링)도 무효다(양의 가역 스케일링은 정확해 부호를 못 바꾼다). A 는 lock OR 만이 아니라 단계 배열 주입 범위도 명시해야 한다(비-S 의 솔버 경로와 인구 상태를 구분). C(pin 없는 주입)는 준위 합의 차가운 출발을 보장하지 못해 현 계약 밖.
**오프라인 판별(발주 전건 조건)**: base 모드(pin OFF, run1 바이너리) 캡처런 3개 — `LUMINA_NLTE_MATDUMP` prelock 행렬(보존 행 교체 전 율 행렬, lumina_plasma.c:20523) Si II sh4(음성대조: 오프라인으로 두 행 교체+주입 목표를 풀어 run1 의 음수 62·최소 −0.01216 재현해야 함)·S II sh10·S II sh24 (jobs 402464–402466, SEED 후 중단). 오프라인: 같은 행렬에 두 보존 행(주입 목표)을 넣어 고정밀(mpmath)과 double 로 풀어 음수의 정확해 여부·분포, 제거한 율 방정식의 불균형을 본다. 판별 결과 전 A 구현 발주 없음.

## 5. 종료감사

Fable fresh + astra 독립 2겹(개정25-m; ① 물리적 확고성 ② 알고리듬–물리 정합).
- astra(`/gpfs/kjhan/lumina/cmfgen_map/coldstart_20260924/closeout_astra_out.txt`): «폐합 차단» 3(수치 오차 해석 무근거·base 대조 인과 과대·A 위험 과소) · 기재 2(PIN-FAIL/X 정합, 진단 표현 정밀화). PIN-FAIL 분류 지지.
- Fable(`/gpfs/kjhan/lumina/cmfgen_map/coldstart_20260924/closeout_fable_out.md`, 자기 계약 편향 기재): «폐합 차단» 1(§2 기전 단정 + §4 A 위험 — astra 차단과 같은 내용) · 기재 6(F1 분기 지지·F3 셸 커버리지·F4 rev 7 메타·F5 경과시간·F7 X 행 문면·F8 오프라인 실행 경로 명시).
- 반영(이 판본): §1 게이트 「미평가」·주입 재현, §2 기전 두 가설 병립·초판 두 단정 철회·자연 분할 실측, §3 주장 상한, §4 A 위험 정정·오프라인 판별 경로, 경과시간. 판독기 `rev: 7` 대 계약 개정 8 은 메타 불일치(개정 8 은 문면만) — 기재.
- 정정 회차: 1회차 전건 반영(개정18 ③). 재감사는 오프라인 판별 결과와 함께 받는다.

## 6. 오프라인 판별 결과 (2026-09-24, 캡처런 402464–66 + 재생)

캡처: base 모드(pin OFF, run1 바이너리 f585e30a), `LUMINA_NLTE_MATDUMP` SEED 최종 CE 반복 prelock 행렬(각 12회 기록 중 마지막), N=202·n_lo=101, T=10020 K. 잡은 SEED 덤프 후 운전자가 소유 확인(sacct WorkDir) 뒤 취소(각 ~60 분, 2×H100). 재생: `/gpfs/kjhan/lumina/cmfgen_map/coldstart_20260924/offline/replay_lock.py`(Opus 5.5; 코드 해법 이식·LAPACK·mpmath dps≥100, 단위 시험 전건 PASS), 산출 `replay_{si4,s10,s24}.json`·`replay_summary.txt`.

| 셸 | 음수 개수 double / 이식 / mpmath / mp(외유출 대각) | 최소(성분) | 강제/자연 hi/lo | 제거 두 행 불균형(상대) |
|---|---|---|---|---|
| Si sh4 (음성대조) | 62/62/62/62 | −0.012336 (100) | ×503 | lo +1.25e5 (0.79) · hi −1.25e5 (0.95) |
| S sh10 | 56/56/56/56 | −1.7617 (116, S III 쪽) | ×1/77 | ±1.67e6 (hi 0.998) |
| S sh24 | 42/42/42/42 | −0.046290 (116, S III 쪽) | ×1/503 | ±2.19e4 (hi 0.999) |

- **음성대조 통과**: Si sh4 재생이 run1 의 음수 개수 62·최악 성분 100·평형화 10·개선 2·pivot growth 0.99997198 를 재현; 최소값 −0.012336 대 run1 −0.012160(상대 1.4e-2 — base/pin 행렬 차이·CE 반복 차이, 캐비앗).
- **판별**: 고정밀 정확해가 double 과 같은 성분에서 같은 수의 음수 ⟹ **가설 S(정확해 음수) 확정, 가설 N(반올림) 기각.**
- **안 A 판정**: S 쌍(J 셸 sh10·sh24)도 정확해 음수 — 결손은 S III 고준위(교체 행 인접), 방향은 Si 와 반대(강제가 S II 쪽). ⟹ **두 보존 행 pin 은 원소·셸을 좁혀도 성립하지 않는다**; 안 A·B 는 현 설계로 발주하지 않는다.
- 기전: 강제 분할이 SEED 율 행렬의 자연 분할과 77–503× 어긋나, 교체로 제거된 두 율 방정식이 총 흐름과 같은 크기의 불균형(상대 ~1)을 떠안고 그 불균형이 교체 행 인접 고준위로 떨어진다.
- 다음: pin 재설계(총량 고정과 율 방정식의 불일치를 어디로 보낼지 명시하는 물리 결정) — 갈림길 평가 Fable → user 결정.
