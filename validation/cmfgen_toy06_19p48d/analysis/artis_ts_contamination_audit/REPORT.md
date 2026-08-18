# ARTIS-timestep 잣대 사고 (사례 18 후보) — 오염 범위 감사 REPORT

감사 주체: Fable 오프라인 감사 에이전트, 2026-07-31.
제약 준수: 런 0회, 소스/스크립트 수정 0건, git 조작 0건. 유일한 쓰기 = 본 디렉토리.
재계산: nlte_000*.out 직독, b_k=n_NLTE/n_LTE=col7/col6 — compare_bk_artis.py:33 / artis_baseline_bk.py:19-20 관례 실독 후 동일 적용 (§4).

---

## §0. 사고 재확인 (전 항목 본 감사에서 독립 재실측)

../artis-ref/tests/toy06_nlte_bk/timesteps.out 실독:

| ts | tstart [d] | tmid [d] |
|----|-----------|----------|
| 20 | 10.7722 | 11.2353 |
| 26 | 17.8519 | 18.6195 |
| **27** | **19.42** | **20.2549** |

- 19.48d를 포함하는 bin = **ts27** (19.42 <= 19.48 < 21.1258). ts20은 10.77-11.72d. 확증.
- docs/ARTIS_PARITY_GAP_AUDIT.md:5 "timestep 20 = 19.4945 d" = 오기, 같은 줄의 자체 경고 "nlte-output parse/timestep-extraction still to be verified" 미상환 박제. 확증.
- scripts/artis_baseline_bk.py:7 기본 TS=20, :25에 "(19.49d)" 하드코드. 확증.
- scripts/compare_bk_artis.py:37 ts=max(alltss, key=...S II 저준위 평균...) = 데이터-의존 epoch 자동선택. 확증. **현행 코드의 실제 선택 = ts8** (S II L1-4 mean 1.60이 최대; §5).
- scripts/radfield_3way.py:13 TS=27 기본 + :53 라벨 "19.42-21.13d bracket of 19.48d" — **정확. 무오염.**
- toy06_nlte_bk / toy06_nlte_run / toy06_whitebox_run 세 시험 디렉토리의 timesteps.out **완전 동일 그리드**. nlte 출력에는 ts5-29 전부 존재 (10638행/ts/rank) — "20"이 "마지막 가용 ts"였다는 가설은 성립 불가.

---

## §1. 기원 (Q1)

### 1-a. 교차-시험(grid transplant) 가설 — 기각
- toy06_whitebox_run/timesteps.out = toy06_nlte_bk와 **동일 그리드** (ts20 mid=11.2353d). 다른 grid에서 옮겨온 것 아님.
- 전 시험 디렉토리 전수: ddc10_nlte_run / classicmode_1d_3dgrid_testrun은 ts20=5.769/5.864d, ts27=7.25/7.37d — 어디에도 ts~20이 19.49d인 그리드 없음.
- StaNdaRT 공개 시간축(data/standart_data1/toy06/spectra_toy06_artis.txt 헤더, 145 epochs): 19.48d 근방은 **19.21 / 19.61** — 19.49d 라벨 부재. StaNdaRT 기원도 기각.

### 1-b. 확정적 기원 후보 (신규 발견): adata.txt 준위-에너지 행 이식
artis-ref/tests/toy06_nlte_bk/adata.txt:1479 (C II 블록, Z=6 ion 2, 207준위, 이온화 24.38 eV — 헤더 실독 확인):

```
          20      19.4945489823976281       2.0000000000000000          14
```

= **C II 준위 20, E=19.4945 eV, g=2, 전이 14개**. artis-ref 트리 전체에서 "19.4945"가 등장하는 곳은 adata.txt의 이 행(및 :10988의 준위 33 행)이 **유일**하다. 오기된 문구 "timestep 20 = 19.4945 d"와 첫 컬럼 "20"·수치 "19.4945"가 **자리수까지 일치**.

재구성된 사고 경로(정황): 07-21 비교군 구축 시 nlte 첫 쿼리가 빈값(rank0 nlte_0000.out=헤더 전용; memory reference_cmfgen_published_toy06_19p48d.md:208에 "첫 쿼리 빈값" 기록) → 19.48/19.49d에 해당하는 timestep을 찾으려 시험 디렉토리를 "19.49"류 문자열로 검색 → timesteps.out에는 19.49가 없고(브래킷 시작=19.42) adata.txt의 준위-에너지 행이 걸림 → 첫 컬럼 20을 timestep으로 오독·이식. **분류: 교차-시험 이식이 아니라 교차-파일(원자데이터→시간축) 이식.** 전사 기록이 없으므로 확률적 판정이지만, (i) 자리수 완전일치 (ii) 대안 전무(전 그리드·StaNdaRT 부재) (iii) "검증 필요" 자체 경고와 동시 기재 — 세 정황이 합치한다.

사고의 뿌리(운전석 진단 재확인): 경고를 달아 놓고 상환하지 않은 것. 07-23 Dig B가 "로컬 ARTIS ts20=11.24d(19.48d 아님! 옳은 비교=ts26/27)"를 이미 발견·기록했으나(project_artis_parity_campaign.md:52 (2)) **docs·스크립트·인덱스로 전파되지 않아** 오기가 7일간 추가 생존했다.

---

## §2. 사용처 전수 (usage census)

분류: **A**=치명(등록 판정/타깃이 오염 — 삭선 또는 ts27 재기저 필수) / **B**=경미(수치·서사는 오염이나 결론은 독립 앵커 보유) / **C**=무관(언급만, 소비 없음 / 무오염 확인).

| # | file:line | 무엇을 먹였나 | 분류 | 근거 |
|---|-----------|--------------|------|------|
| 1 | docs/ARTIS_PARITY_GAP_AUDIT.md:5 | ARTIS-parity 캠페인 전체의 비교군 선언 "ts20=19.4945d" + 미상환 자체 경고 | **A** | 캠페인 정본 문서의 기저 선언 자체가 오기. ts27 재기저 + 경고 상환 필수 |
| 2 | scripts/artis_baseline_bk.py:7,25 | Fe III 트랩 b_k 앵커·f(FeIV) 추출기 — TS=20 기본 + "(19.49d)" 하드코드 | **A** | 이 계기가 #4의 오염 앵커를 생산. 기본값·라벨 수정 필수 (§7) |
| 3 | scripts/compare_bk_artis.py:3-5,26-37 | 데이터-의존 epoch 자동선택(현행 코드 실측: **ts8** 선택) + 무단서 서사 "ARTIS runs its line-formers super-thermal (Si II ~18, S II ~48, Ca II ~10)" | **A**(계기)/**B**(서사) | 자동선택은 max-picker=취약 계기, 제거 필수. 단 :28-30 주석은 "ts27(19.48d)은 faded"를 **이미 알고** 형광-활성기를 의도 선택(caveat이 project_macroatom_artis_diff_breakthrough.md:55,82에 등록) — 순수 오독이 아니라 방법론적 선택+계기 불량의 혼합. :78이 x18-48을 "Rydberg-inflated"로 자체 강등 완료 |
| 4 | memory reference_cmfgen_published_toy06_19p48d.md:208,210 | "ts20=19.4945d" 등록 x2 + **"ts20 광구(cell10-11): Fe III lev17 b_k=0.60/1.02, lev25=0.86/1.23 (~1)" → Group A 타깃 "Lumina 100-180→~1"** + awk 추출법 $1==20 | **A** | 등록 앵커 수치가 11.2d 상태. 진짜 19.48d의 ARTIS는 lev17=1.8-3.1, lev25=1.7-2.6 (**~1 아님**, §5). 단 Group A "과충전은 병리" 결론 자체는 dig_B2 CMFGEN 심판(lev17 41x vs CMFGEN 0.94)으로 독립 생존 — 삭선 대상은 "ARTIS@19.48d ~1" 앵커 수치 |
| 5 | memory reference_artis_whitebox_event_ledger.md:17 | "ts20=19.4945d" + "cell10 f(FeIV)=0.34 vs StaNdaRT 0.000" → 판정규칙 "로컬 nlte_bk=b_k구조(lev17/25~1)만 유효" | **B** | 수치 오염(정 epoch ts27: f(FeIV)=**0.455**) — 그러나 규칙의 결론(로컬 절대 이온화 != 공개 StaNdaRT, 절대값 사용 금지)은 0.455에서 **오히려 강화**. 괄호 "(lev17/25~1)"만 ts27 값(~1.7-3.1)으로 정정 필요 |
| 6 | memory project_artis_parity_campaign.md:23,32 | parity1/2 잔차 정량 "(ARTIS ~1까지 잔여 2-20x)" → MEMORY.md 실미결 항목 "b_k 2-20x" | **B** | 같은 파일 :52 (2)에서 Dig B가 07-23에 자체 강등 완료("비교자 무효, ts20=11.24d, 옳은 비교=ts26/27; 잣대보정 후 lev17 4-5x만 실재"). 원장 내 자기교정 존재 — 인덱스 전파만 누락 |
| 7 | memory MEMORY.md 캠페인 훅 "실미결=...b_k 2-20x..." | 신규 세션이 첫 로드하는 인덱스 | **B** | 정본 토픽파일이 이미 강등을 담고 있으나 인덱스 행이 구 수치를 무주석 반송 — 재주석 권고 |
| 8 | memory project_macroatom_artis_diff_breakthrough.md:53-56,60-63,82 | "ARTIS 실측 b_k (수렴 ts11, cell24): Si II median=17.9, S II=48.1, Ca II=9.9" → 서사 탄생지; G1 liftoff 타깃(S II L1-4>=2); 도구 등록 | **B** | epoch가 **명시적으로 ts11(5.27d)로 라벨됨**(오독 아님) + caveat 3중 등록(:55 decay-powered 절대값 무효 / :60 median=Rydberg-inflated / :82 "ts27(19.48d)은 sub-thermal"). 07-08~14 co-evolve 캠페인의 "Lumina b_k==1 동결=배선결함" 결론은 ARTIS 소스 구조 차분+이후 CMFGEN 심판으로 독립 생존. 단 이 시대의 절대 liftoff 타깃(x2-3)은 5.3d 상태 기준 — 19.48d ARTIS로는 정당화 불가(§5), 현행 CMFGEN-심판 체제로 대체 완료 |
| 9 | scripts/radfield_3way.py:13,53 | J_nu 3-way 비교 | **C** | TS=27 + 정확한 브래킷 라벨. 실측 확인, 무오염 |
| 10 | scripts/jbl_verdict.py:27-29 | corr(MC,ARTIS) — StaNdaRT 스펙트럼 열 p[77] | **C** | p[77]=flux_t76=TIMES[76]=**19.61d** (19.48d 최근접 격자점; 19.21/19.61 중 19.61이 더 가까움). nlte_bk/ts 무관. 무오염 |
| 11 | docs/ARTIS_COMPARISON_LADDER.md | 구 사다리 (2026-06-28) | **C** | 비교군=ddc10@7.87d — 별개 그리드·별개 시대. ts20 소비 없음 |
| 12 | docs/FLUOR_ATTACK_DESIGN_2026-07-06.md:13,24 / FLUORESCENCE_DESIGN_AB.md / LADDER_V2.md / N3_JBAR_DAMP_UNIFY_DESIGN.md | ARTIS 언급 = UV분율(스펙트럼 수준)·macroatom/radfield **소스 구조** 참조 | **C** | nlte_bk timestep 데이터 비소비. LADDER_V2:21 및 구 super-thermal S_l 사가(CONTROLLED_TEST_SUITE 등)는 **LUMINA 내부 S_l** 이야기로 본 사고와 별개 계보 |
| 13 | logs/coevolve_consume_parity43,51,54/VERDICT_*.md | 최근 판정의 ARTIS 참조 | **C** | 전부 소스-구조(vR+Axelrod 처방, N3 raw-통일, 문턱 3) 또는 corr(ARTIS)=StaNdaRT@19.61d. ts20 b_k 소비 없음 |
| 14 | docs/VERIFICATION_REGISTERS.md | 3대 대장의 ARTIS 항목 | **C** | N3 결착=radfield.cc 실독(구조), "ARTIS Fe-피크=Verner 비독립" 노트 등 — 시간축 데이터 무관 |
| 15 | docs/ARTIS_CMFGEN_PARITY_DIAGNOSIS_2026-07-30.md / RE_...2026-07-30.md | 사고를 **기술**하는 문서 | **C** | 소비자 아님(사고 발각 문서) |
| 16 | logs/coevolve_consume_parity8/analysis/dig_B_bk/ | ts26/27 재기저 교정 분석 (07-23) | **C** | 해독제(본 감사와 동일 방향의 선행 교정) |

**집계: A=4** (#1 문서, #4 원장, #2·#3 계기) / **B=4** (#5,6,7,8) / **C=8** (#9-16).

---

## §3. b4 문턱 2.5의 provenance (Q3) — **무오염 (CMFGEN/내부 유래) 판정**

유래 사슬 (실독):
1. logs/coevolve_consume_parity8/analysis/dig_F12_siiii_pops/VERDICT.md:150 — "R2 ... 예측 **b4=1.04+-0.15** (t3e 실측장 평형 표), **b4가 재솔브 후에도 2.8이면 R2 기각·미지 채널 수색**". 입력 전부 LUMINA 내부: P-dump b4=2.79(솔버 실산출, cuda.cu:8041-8084 writer 직독 인증), t3e 실측장 재솔브 평형 1.04, 이벤트로그 J 79선.
2. 기준 잣대 = **CMFGEN 진리** (1P-deg 0.9753/1.0055 — dig_F12 T4에서 진리 J 대입 비교).
3. 캠페인 원장(project_artis_parity_campaign.md:135)이 이를 "예측 b4=1.04+-0.15, **>=2.5면 기각**"으로 등록 — 2.5는 평형 예측(1.04+0.15)과 stale dump(2.79) 사이의 판별 경계.
4. parity26-diag: 재솔브 b4=3.08 >= 2.5 → R2(이월) 기각 — 문턱 소비처.

**ARTIS timestep 데이터는 이 사슬 어디에도 등장하지 않는다.** b4 잣대(Si III 1P-deg 대 CMFGEN)와 문턱 2.5는 본 사고와 무관 — 삭선·재기저 불요. (유일 캐비앗: VERDICT 원문은 "2.8이면 기각", 원장 등록은 ">=2.5" — 전사 중 0.3 낮춰졌으나 parity26 실측 3.08은 어느 쪽으로도 기각이라 판정 불변.)

---

## §4. 사용 규약 검증 (Q5 전제)

compare_bk_artis.py 관례 실독: b_k = float(r[6])/float(r[5]) = **n_NLTE/n_LTE** (스키마: timestep mgi Z ionstage level n_LTE n_NLTE ion_popfrac; ionstage 1=중성). 저준위 통계 = 준위 1-4, 필터 0<b<1e3, **평균** (단 :77 출력문은 "MEDIAN L1-7"이라 주장 — 코드와 불일치, 부수 결함으로 기재). 셀 필터 **없음** (:17 주석 "line-forming cells = inner-mid third"도 코드 미구현 — 부수 결함). 서사 수치 ~18/~48의 원 관례 = **전준위 median, cell24, ts11** (project_macroatom_artis_diff_breakthrough.md:53).

---

## §5. ts27 재기저 실측 (Q5) — 생존 시험

### 5-a. 서사 수치의 재현 (잣대 인증)
전준위 median @ cell24, **ts11**: Si II **16.57** (max 192.8), S II **48.13** (max 128.0), Ca II **9.81** (max 140.8); b>3 비율 94/90/73%.
→ 원장 등록치 "17.9(max193) / 48.1(max128) / 9.9(max141), 94/90/73%"와 **일치** (S II·Ca II·max·비율 완전 재현; Si II median 16.57 vs 17.9는 경미한 필터 차이 소산). **서사의 출생지=ts11(~5.27d) 확증.**

### 5-b. 동일 통계량 @ ts27 (19.48d 브래킷) — **서사 비생존**
| 관례 | Si II | S II | Ca II |
|------|-------|------|-------|
| 전준위 median cell24, ts11 (서사 원본) | 16.57 | 48.13 | 9.81 |
| **전준위 median cell24, ts27** | **0.00** | **0.01** | **0.00** |
| ts27 max / frac(b>3) | 1.0 / 0% | 1.3 / 0% | 1.0 / 0% |

현행 코드 관례(L1-4 mean, 전셀):
| ts | Si II | S II | Ca II | Fe II | Fe III |
|----|-------|------|-------|-------|--------|
| auto(=ts8) | 149.75+ | 1.60 | 5.92 | — | — |
| ts20 | 0.48 | 1.09 | 0.67 | 0.97 | 1.00 |
| **ts27** | **0.27** | **0.80** | **0.39** | **1.01** | **0.98** |

(+) ts8 Si II 평균 149.75는 max 870 아웃라이어 지배 (BMAX=1e3 필터 통과) — max-picker 계기의 취약성 부수 실증.

**판정: "ARTIS는 line-former를 super-thermal로 돌린다"는 서사는 매칭 epoch(19.48d)에서 죽는다.** ts27의 ARTIS line-former는 전면 **sub-thermal**(고준위 붕괴, L1-4도 <=1). 초열화는 decay-powered 초기(ts8-11, ~5-5.3d) 한정 현상 — 원장 :82의 caveat 그대로. 따라서 "Lumina도 19.48d에서 b_k를 ~2-48로 들어올려야 한다"는 형태의 잔존 타깃은 어느 것도 ARTIS 근거로 정당화 불가 (현행 CMFGEN-심판 체제와도 합치).

### 5-c. Fe III 트랩 앵커·f(FeIV) 재기저 (artis_baseline_bk 관례)
| epoch | cell | f(FeIV) | lev17 | lev25 |
|-------|------|---------|-------|-------|
| ts20 (등록 원본) | 10 / 11 | 0.338 / 0.216 | 0.60 / 1.02 | 0.86 / 1.23 |
| ts26 | 10 / 11 | 0.450 / 0.453 | 1.92 / 3.26 | 1.89 / 2.93 |
| **ts27 (19.48d)** | 10 / 11 | **0.455 / 0.457** | **1.83 / 3.12** | **1.71 / 2.64** |

- 등록 원본(0.60/1.02, 0.86/1.23, f=0.34) 완전 재현 → 원장 수치의 출생지=ts20 확증.
- **정 epoch의 ARTIS 트랩 준위는 ~1이 아니라 ~1.7-3.1** — "ARTIS는 트랩을 b_k~1로 열화한다 @19.48d" 명제는 수치적으로 틀림(ARTIS도 19.48d에는 트랩을 2-3x 부양). Dig B의 "잣대보정 후 lev17 4-5x 과잉"과 정합 (LUMINA ~8-20 / ARTIS ~2-3).
- Group A의 물리 결론(배수망 부재로 인한 100-180x 과충전=병리)은 **CMFGEN 심판으로 생존** (dig_B2: LUMINA lev17 41x vs CMFGEN **0.94** sub-thermal). 아이러니: "→~1" 타깃은 ARTIS 근거로는 틀렸으나 CMFGEN 근거로는 대략 맞는 값이었음.
- f(FeIV) 로컬런 0.455 vs 공개 StaNdaRT 0.000 — 절대 이온화 불신 규칙은 **강화**.

---

## §6. 삭선/재기저 목록 (원장 처분 — 기재 권고, 본 감사는 수정 미적용)

1. **docs/ARTIS_PARITY_GAP_AUDIT.md:5** — "timestep 20 = 19.4945 d" 삭선 → "timestep 27 (19.42-21.13d bracket of 19.48d, mid 20.2549d)" + 경고 상환 문구(본 감사 참조) 기재.
2. **memory reference_cmfgen_published_toy06_19p48d.md:208,210** — "ts20=19.4945d" 2건 정정; **앵커 교체**: "ARTIS 광구 트랩 b_k~1" → "ts27 cell10-11: lev17 1.83/3.12, lev25 1.71/2.64 (ARTIS도 2-3x 부양; ~1은 CMFGEN 쪽 값 0.94)"; awk 추출법 $1==20 → $1==27. Group A 결론은 dig_B2 CMFGEN 앵커 명기로 존치.
3. **memory reference_artis_whitebox_event_ledger.md:17** — "ts20=19.4945d"→ts27; "cell10 f(FeIV)=0.34"→0.455 (규칙 자체는 존치·강화 명기); "(lev17/25~1)" 괄호 → "(lev17/25~1.7-3.1 @ts27)".
4. **memory MEMORY.md 캠페인 훅** — "실미결 ... b_k 2-20x"에 Dig B 강등 주석 반영(비교자 무효; 잣대보정 후 lev17 4-5x만 실재).
5. **서사 은퇴 공지**: "ARTIS line-formers super-thermal (Si II ~18/S II ~48/Ca II ~10)"는 "ARTIS **초기 형광기(~5.3d)** 상태, 19.48d에는 sub-thermal(median~0)"로만 인용 가능 — 매칭-epoch 타깃으로 재사용 금지.
6. **사례 등록**: feedback_audit_the_yardstick_first.md에 사례 18 (adata.txt 준위-에너지 행 → 시간축 이식 + 자체 경고 미상환 + Dig B 발견의 전파 실패) — 운전석 결정 사항.
7. 삭선 불요 확인: **b4 문턱 2.5** (§3, CMFGEN/내부 유래), radfield_3way.py, jbl_verdict.py(19.61d), 최근 parity 판정(43/51/54), N3/Y6 결착 — 전부 무오염.

---

## §7. 스크립트 수리 사양 (미적용 — 스펙만)

### scripts/artis_baseline_bk.py
1. :7 — `TS=int(sys.argv[1]) if len(sys.argv)>1 else 20` → 기본값 `27`.
2. :4 — Usage 문자열 `[timestep=20]` → `[timestep=27]`.
3. :25 — 하드코드 라벨 `(19.49d)` 제거; timesteps.out을 직독해 `f"(t={tstart:.2f}-{tend:.2f}d)"` 출력 또는 최소한 `(ts27=19.42-21.13d)` 정적 라벨로 교체. **시간 라벨을 다시 하드코드하지 말고 파일에서 읽는 쪽 권장** (동일 사고 재발 차단).

### scripts/compare_bk_artis.py
1. :36-37 — 데이터-의존 자동선택 `alltss...; ts=max(alltss, key=...)` **제거** → 명시 파라미터 `ts=int(sys.argv[3]) if len(sys.argv)>3 else 27` (형광-활성기 비교가 필요하면 호출자가 ts=8~11을 명시적으로 넘기게 강제; max-picker 재도입 금지 — §5-b (+)의 아웃라이어 지배 실증).
2. :3-5 docstring — "ARTIS runs its line-formers super-thermal (Si II ~18, S II ~48, Ca II ~10)" → epoch 단서 필수 명기: "ARTIS(decay-powered)는 초기(~5.3d, ts8-11)에만 super-thermal; 매칭 epoch ts27(19.48d)은 sub-thermal (L1-4 mean: Si II 0.27 / S II 0.80 / Ca II 0.39; 전준위 median~0)".
3. :26-30 주석 — "Pick ts by the highest low-level S II median" 문장 삭제(1의 코드 변경과 정합화). "faded by the matched epoch (ts27, 19.48d)" 문구는 정확하므로 존치.
4. 부수 결함(같은 김에, 별도 헝크): :17 주석 "line-forming cells = inner-mid third"는 코드 미구현(전셀 사용) — 주석 삭제 또는 셀 필터 구현 중 택일; :77 출력문 "MEDIAN L1-7" vs 실제 계산(mean L1-4) 불일치 정정.
5. 수정 후 §5-a 재현 체크(ts11 cell24 median 48.1) 1회로 잣대 연속성 인증 권고.

### docs/ARTIS_PARITY_GAP_AUDIT.md:5
§6-1의 문구 교체 (문서 수정 — 운전석 재량).

---

## 부록: 감사 커버리지
- grep 리드 전수 실행: ts20/TS=20/timestep 20/19.49/19.4945/super-thermal/Si II ~18/S II ~48/artis_baseline_bk/compare_bk_artis/toy06_nlte_bk x {docs, scripts, logs, validation, memory(읽기전용), artis-ref}.
- toy06_nlte_bk를 참조하는 스크립트는 정확히 2개(둘 다 §7에서 수리 지정); radfield_3way는 whitebox_run 경로·ts27로 무오염.
- 구 "super-thermal S_l" 문서군(CONTROLLED_TEST_SUITE/TOPSTAGE/FREQ_RESOLVED 등)은 LUMINA 내부 S_l 사가로 본 사고와 계보 분리 — 오염 아님.
