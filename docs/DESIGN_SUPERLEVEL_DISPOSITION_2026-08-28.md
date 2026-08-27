# 설계 연구 — 반상태(super-level collapse)의 처분 (2026-08-28)

**성격: 사전등록이 아니다.** 계약·게이트·NC 를 정의하지 않고, 수리를 집행하지 않는다.
BITLOCK 폐합(`4bf4a5b`)이 연 미결 #4(«반상태 수리 단 개설 여부 — user 결정 대기»)와
#3(«반상태 전제 재감사 — 11이온»)에 대해, **user 가 갈림길을 고를 수 있게 증거를 정리한
문서**다(`feedback_fable_at_forks` 규약). 두 물음 — ① 사정거리(무엇이 이 반상태 위에 서
있는가) ② 수리 여부 — 을 한 문서에서 다루고, ③ 갈림길 자체의 타당성을 검사한다.

- 저자 = Fable (분담 개정14 ①). 작성 시점 HEAD `af7441d` [저자 실행: `git log -1`].
  `git diff --stat a849306 af7441d -- src/` **공백** [저자 실행] ⟹ 본 문서의 소스 행 번호 =
  BITLOCK 판정문의 행 번호 = 현 HEAD, 전 행 저자 직접 재확인.
- 봉인 4곳(census·R-II·L6C·L6) 접근 전부 read-only(grep·스트리밍 파스). 무거운 파스 1건은
  grammar-debug(nested ssh, stdin 파이프) [부록 A].
- **오라클(CMFGEN 자체런) 정량 인용 0.** §5-3 의 CMFGEN 대조는 **소스 관례의 정성 직독**이다
  (수렴 런 산출물 인용 아님).
- 표기: `[저자 실행]` = 이 문서를 위해 직접 실행·실측 / `[판단]` = 증거에서의 저자 판단 /
  `[추정]` = 추측 / `[미확인]` = 확인하지 못함. 명령 전량은 부록 A.

---

## §0. 대상의 정확한 특정 — 반상태란 무엇인가 (전건 [저자 실행] 재확인)

BITLOCK 이 귀속한 사슬을 전 행 재확인했고, **판정문에 없던 실측 2건**을 더한다.

**사슬(재확인)**: ① `src/lumina_atomic.c:1468` `getenv("LUMINA_SUPER_CUTOFF")` · `:1470`
`if (K > 0)` — **무조건** 덮어쓰기, 배너 `:1476-1477` 에 "set LUMINA_SUPER_LEVELS=1 to
activate" **권고 문구만** ② `lumina_plasma.c:16350-16351` `super_mode = super_requested && …`,
`:16456-16457` `super_requested = getenv("LUMINA_SUPER_LEVELS")…` — 레이아웃(`fl_to_super`,
`:16384-16387`)은 super_mode 와 **무관하게** `level_super[]` 에서 온다 ③
`population_contract.c:199-200` `if (!super_mode) work[i]=1.0` ⟹ `within_sl_frac ≡ 1.0`
④ 재분배 `:19396-19399` `xfl[i] = b[sl] * f` — 이온당 lumped 전원 한 값 ⑤ 배너
`:16482-16484` 가 `off (identity)` 를 인쇄하며 자기모순.

**신규 실측 1 — f 는 재분배만이 아니라 solve 행렬 조립에도 들어간다.**
`lumina_plasma.c:16991-16994` [저자 실행 직독]:

```
#define SOLVE_OF(fl_g) (nlte->fl_to_super[(fl_g)] - super_start)
#define FRAC_OF(fl_g)  (nlte->within_sl_frac[…])
```

주석 문면: *"In super mode the matrix dim N is the super-level count and FL contributions
aggregate into their SL row/col, **weighted by the within-SL Boltzmann fraction**."*
⟹ 반상태(f≡1)에서 SL 방정식의 계수는 구성원 FL 율의 **무가중 합**이다. 어느 SL 정식화
(CMFGEN·ARTIS, §5-3)도 무가중 합을 쓰지 않는다. **반상태는 재분배 출력만이 아니라 SL
방정식 자체를 오조립한다** — 즉 현재의 1,173-SL 해 `b[sl]` 는 «올바른 SL 해» 도 아니다.

**신규 실측 2 — 항등 복원은 env 제거로 되지 않는다.**
`LUMINA_SUPER_CUTOFF` 를 **빼면** `level_super[]` = 덱 `super_level` 열이 되는데, 이 열은
CMFGEN `f_to_s` 파일의 직수입이라(`scripts/expand_atomic_data_cmfgen.py:159-190` [저자 실행]
— 예: `(26,2): 'f_to_s_342'  # Fe II 2698 FL -> 342 SL`) **원생부터 비항등**이다(BITLOCK A3:
커버 이온 6,154/6,157행 비항등). ⟹ 노브 제거 = 덱-원생 층위의 **같은 계급 반상태**(BITLOCK
§4-3 좌상단 칸). 항등을 만드는 경로는 ⓐ src 수리(map 을 super_mode 에 종속) ⓑ 덱 재생성
(열 제거→로더 항등 기본, `lumina_atomic.c:1444-1446`) ⓒ **env 트릭: `K ≥ 전 이온 최대
준위수`**(예: 99999 — `:1473-1474` 산법상 `level_super = level_num` 항등이 된다) 셋뿐이다.
ⓒ의 실행에는 DET 런처의 fail-closed 검사(`LUMINA_SUPER_CUTOFF == 100` 요구, §1) 수정이 동반된다.

**死블록 재확인** [저자 실행 — 자체 awk 깊이 오토마톤]: `lumina_plasma.c` 死블록 = 정확히
**1324–1693**(`#if 0` A2-17 retired), `lumina_cuda.cu` = 정확히 **1083–2063**. 본 문서의
인용 행 전건이 두 구간 밖이다.

---

## §1. 계보 — 반상태는 어디서 왔고, 무엇이 그 위에서 돌았나 (전건 [저자 실행])

| 시점 | 사건 | 실측 근거 |
|---|---|---|
| 06-08 | SL 기계 도입(`9a4e91d`) — 헤더 계약 `lumina.h:723-724`: *"super_mode==0 이면 전 map 항등·FL solve 와 byte-동일"* | `git log -S within_sl_frac` |
| 07-03 | ARTIS 컷오프 무조건 덮어쓰기 도입(`66586d6`) — 도입 사유 = **full 준위 행렬의 ~250-order 스팬 ill-conditioning**(로더 주석 `:1449-1460` + `docs/CLAMP_INTRO_REASON_ARCHAEOLOGY.md` C48) | git show + 주석 직독 |
| 7월 | **두 계보 병존**: ⓐ parity 계보 = `launch_parity*` 17/18본이 `LUMINA_SUPER_LEVELS=1` 명시 + PARITY59 env 5본 양 노브 ⟹ **ACTIVE** — 배너 실측 4런(`parity45/47/49/57`): `Super-levels: ACTIVE (21038 FL -> 2828 SL)` ⓑ 애드혹 `run_*.sh` 77본 = **접두 없는 `SUPER_LEVELS=1`**(소스 독자 **0** — `getenv("SUPER_LEVELS")` grep 0건) + `LUMINA_SUPER_CUTOFF=100` ⟹ **반상태** (활성화 **의도**의 오철자 흔적) | 런처 grep 전수 + 봉인 로그 배너 |
| 08-05~08 | `a97d0e1`(층 0) → `run_det_convergence_2026-08-08.slurm`: 수리-노브 전건 0 강제 + **`LUMINA_SUPER_CUTOFF == 100` fail-closed 요구**(주석: *"SUPER_CUTOFF is a **declared super-level topology** (model resolution), not a post-solve floor/cap"*) · `LUMINA_SUPER_LEVELS` **미설정** ⟹ K36/DET/A210 계보 전건 반상태 | 런처 직독(`:278-282`) |
| 08-15~현재 | 반상태 배너 실측: `a210_cancellation_census_mgpu`(08-15)·L6·L6C·census·R-II 전건 `off (identity) (13015 FL -> 1173 SL)` · 봉인 4곳 exports 전건 `LUMINA_SUPER_CUTOFF="100"` 단독 | 봉인 stdout·exports grep |

**SUPER_LEVELS 가 계보에서 언제·왜 탈락했는가**: parity59(양 노브)와 08-08 DET 런처(컷오프만)
사이. 정확한 탈락 커밋·의도는 **[미확인]** — 그 창은 Codex 자율 진행기(08-05~08)다.

**의도 판독 [판단]**: 서로 독립인 세 흔적 — ⓐ 77본 스크립트의 (오철자) 활성화 시도
ⓑ DET 런처의 "declared super-level topology" 선언 ⓒ 코드 자신의 활성화 지시(배너·주석)
— 이 전부 «**럼핑은 의도된 형상, f≡1 은 비의도**»를 가리킨다. 반상태를 "의도된 identity
모드"로 읽을 흔적은 찾지 못했다(항등이 의도였다면 컷오프 요구·f_to_s 임포트·활성화 문구가
전부 무의미해진다).

---

## §2. 물음 1 — 사정거리: 대장 A·C 전수 판정

### 2-0. 판정 틀

노출 계급 셋으로 가른다 (§1 실측 기반):

- **반상태 노출**: 08-08 이후 K36/DET/A210 계보 + 7월 애드혹 `run_*.sh` 계열.
- **의도-SL 노출**: 7월 parity 계보(ACTIVE 배너·Boltzmann f) — **반상태가 아니다.**
- **무노출**: 오프라인·데이터·소스·합성 픽스처.

★**구조적 사실 하나가 판 전체를 지배한다**: 06-08 이후 **full-FL solve 로 잰 결론은 대장에
하나도 없다.** 7월 parity 계보도 SL-차원 solve(2,828 SL)였고, 현 계보는 1,173 SL 이다.
대장 항목의 차이는 «FL 이냐 SL 이냐»가 아니라 «f 가 Boltzmann(의도-SL)이냐 1(반상태)이냐»다.

### 2-1. A 대장 12건 — 전수 판정

| # | 항목 | 반상태 의존 | 근거 |
|---|---|---|---|
| 1 | verdict_gate.py | **무관** | 판정 도구·합성 픽스처 — 인구 비소비 |
| 2 | Saha 하니스+RATES_FIX | **무관** [판단] | 이온화 밸런스 계약. partition 은 전 준위 직산(`population_partition_ion`) — `fl_to_super` 비경유 |
| 3 | 율 기계 PRRR 인증(4이온) | **무관** [판단] | 원 판정문이 자체 population 을 인증 범위 **밖**에 명시(`rates_certification/VERDICT.md:246`, 0g 재확인 대장). 실행 시대 = 7월 트윈 — 모드 [미확인]이나 어느 모드든 반상태 축 무관 |
| 4 | Γ(Co III) 34-57× | **무관(반상태 축)** | 7월 시대. Γ 분해가 인구-가중이면 **SL-럼핑(의도) 민감성은 [모름]** — 별개 축(§2-4) |
| 5 | POPCOB 규약 | **무관** | 규약·왕복 항등 |
| 6 | GPH 3게이트 회귀 귀속 | **무관** [판단] | 45런 차동 귀속 — 럼핑은 양팔 공통(차동 면역). 45런의 시대 혼합 여부 [미확인]이나 귀속 논리는 차동 |
| 7 | parity42 판정 | **무관** | 차동(V0-V5), 양팔 공통 구성. parity 계보 = ACTIVE 모드 [실측] |
| 8 | S IV/V·Ca IV/V 임포트 | **무관** | 데이터 자체검증 |
| 9 | 파이프라인 결정론 | **무관(성질)** | 결정론이라는 성질 판정은 유효. ★단 그 **앵커 값**은 구성-한정 — 수리 시 재-bake 대상(§6) |
| 10 | FORMAL_FIX 수리(R1-R3) | **무관** [판단] | formal 적분기 기계 수리 — 영점·OFF-실증·차동 잣대 |
| 11 | 실Υ 40이온 추출 | **무관** | 데이터 끝-끝 사슬 |
| 12 | 클램프 무죄 8건 | **무관(현 구성 사실)** | "발화 0" 은 측정 런들의 구성에서의 사실. ★수리로 인구가 바뀌면 발화 여부는 **재확인 대상**(§6 큐) |

**A 소계: 의존 0 · 무관 12** (그 중 #9·#12 는 수리 시 재-bake/재확인 큐).

### 2-2. C 대장 본표 — 전수 판정

본표 실측 **25행**(헤더 "21건"은 구계수로 보임 — 관측 기재만 [저자 실행: 행 계수]).
전건 7월 시대의 산출이고, 증거 계급은 ⓐ 소스/데이터 직독 ⓑ 산술·항등식 ⓒ 차동(A/B·쌍둥이·
byte-parity) ⓓ 절대 정량 넷으로 갈린다.

| 행(대장 표기) | 판정 | 계급·근거 |
|---|---|---|
| 게이트 격자에 해 없음 / SKIP_Z 화석 / LTE floor 병리 / 장 되먹임 교락 / B-C Saha 비대칭 / CAP_TOPION / OMEGA_FLOOR / OMEGA_GEN_V3 / formal 구조 결함 / Co III 105=full / 조기정지 오판 / S_l 3원화 / K6 실체 / K6 결착 / [20호] Y6-N2 / [21호] Y4-N7 / 이산화 2후보 / parity43 판정 | **무관** | 소스·데이터 직독 또는 차동(양팔 공통 구성 — 럼핑 면역). parity44/45/47/49 배너 **ACTIVE 실측** — 반상태 아님 |
| mode-2 상쇄 기각 / 상쇄 항등식 S·τ ∝ n_u | **무관** | 산술·코드 항등식(인구 값과 무관하게 성립 검증) |
| S IV/V·Ca IV/V… 임포트·Ni II 임포트·인증 커버리지 지도 | **무관** | 데이터 축 |
| 2-20× 이온화 오차=장 귀속 / Γ(Co III) FUV 지배 / det-jbar 오염 | **무관(반상태 축)** · **[모름](SL 축)** | 절대 정량 포함 — 시행 시대는 의도-SL. 반상태 의존은 없으나, 정량이 **의도-SL 럼핑에 얼마나 민감한지는 미측**(§2-4) |

**C 본표 소계: 반상태-의존 0.** (의도-SL 민감성 [모름] 3행 — 이것은 반상태 수리와 무관하게
존재해 온 축으로, D-이월 «H2 SUPER_CUTOFF 레시피» 계급이다.)

### 2-3. C 신규(4~8회차 승격) — 전수 판정

| 회차·항목 | 판정 | 근거 |
|---|---|---|
| 4회차: NO_BRACKET 5건(순-흡수 0/1282 · S_p/B=0.999993 · J_cont≤B·J̄≤S_p · NO_BRACKET=LTE 시드 필연 · 생산/소비 불일치) | **무관** [판단-강] | 대상 1,282선 = **ion IV = NLTE 인벤토리 밖**(A207 W-A 실측: 594행 전건 base 0/594 사상) — 그 행들의 인구는 LTE-at-핀-T 경로가 공급하며 반상태 재분배를 지나지 않는다. ★단 J_cont·J̄ **값 자체**는 반상태 물질이 만든 장의 값 — 정성 결론(희석 부등식·근 부재의 대수)은 강건, 정량은 구성-한정 |
| 5회차: Z-O census (hazard 221,382 · fa=1.0 · 등g 전건 · bit-동일 0/221,382) | **의존-구성** | hazard 집합 **자체가 반상태의 산물**(BITLOCK 귀속). 그 구성의 관측으로서는 영구 유효 — 물리 일반화 불가는 이미 자기 명기 |
| 6회차: BITLOCK (H-FLAT 귀속) | **반상태의 서술 그 자체** | 재감사 무의미 — 귀속이 곧 내용. 소스 사슬은 저자가 전 행 재확인(§0) |
| 7회차: Z-2A (격리 Class-1 in-vivo) | **무관(기전)** | 격리 계약(진단 실패≠물리 중단)의 검증 — 수리와 독립으로 존속. 단 그 in-vivo 스트레서(hazard 442,764회)가 반상태 산물이므로, 수리 후에는 같은 스트레스 재현이 불가하다는 **주석**만 남는다 |
| 8회차: F1b (ff=95/885 · C-M-S · ff_resp=0.0 · 시드≠LTE) | **의존-명시** | 판정문 스스로 «SL-단위» 한정·반상태 구성 명기. **수리 후 재측정의 前항으로 설계됨**(F1b §5-1 ④·§9 #1) |

**C 신규 소계: 의존 3(Z-O·BITLOCK·F1b) · 무관 2(NO_BRACKET·Z-2A).**

### 2-4. 종합 — 사정거리의 정확한 모양

1. **반상태는 대장의 어떤 «확정»도 뒤집지 않는다.** 의존 3건은 전부 8월 사슬이고, 셋 다
   주장 상한 규율로 스스로 «이 구성의 관측»으로 한정해 두었다(감리 두 겹이 어휘 스캔까지
   했다). 사정거리는 **결론이 아니라 (i) bit-앵커 계급(§6) 과 (ii) 미래 측정의 잣대**에
   미친다.
2. **11이온 사정거리의 신규 정량** [저자 실행 — 덱 전수, grammar-debug]: NLTE 인벤토리
   12슬롯(11절단+Ca II) 기준, **양쪽 준위가 모두 lumped(≥100)인 선 = 1,537,729행 = 덱 전체
   2,588,798행의 59.4%**. 그 중 등g = **539,149행**(반상태에서 τ 정확-0 계급), 나머지
   ~1.0M 행은 τ·S 가 **g-비 하나로 결정**된다(χ<0 위장 inversion 포함 — F1b NEG 46행의
   모집단). BITLOCK D5(커버 3이온 325,489)의 전-인벤토리 확장이다. ⟹ **재감사 단(#3)의
   대상 규모가 이것이다.**
3. **재감사 대상 목록** (수리 채택 여부와 무관하게 BITLOCK §8 #3 이 요구):

| 대상 | 성격 |
|---|---|
| L6·L6C 계열 전제 재감사(봉인 exports 에 반상태 실재 — BITLOCK §6-1 실측) | 존속 미결의 본체. 단 L6 의 관측 자체(ion IV)는 §2-3 대로 반상태 잣대 비소비 |
| K36 결정론 baseline(R1/R2 residual — B 대장)·DET-SPROD S2 bit항등 | 앵커 재-bake 계급(성질은 재증명 가능) |
| 클램프 무죄 8건(A-12)·[TAU-BY-ION] 류 절대 관측치 | 인구 변경 후 발화·값 재확인 |
| F1b 재측정 | 수리 형에 따라: 승격(B안)→SL-Boltzmann 재판독 / 항등(A안)→FL-단위. **前항은 F1b 가 이미 확보** |

---

## §3. 갈림길을 자르기 전에 — 두 축으로의 분해 (물음 3 의 입력)

이 사안은 서로 다른 두 축이 겹쳐 있고, 섞어 물으면 답이 헛돈다:

- **축 1 — 형상(topology)**: 준위 공간을 무엇으로 푸는가 — 항등(13,015 FL) / 덱-원생
  CMFGEN f_to_s / ARTIS-K100(현행 1,173 SL). **모델 해상도의 선택**이다. 런처가 이미
  "declared super-level topology" 로 봉인해 왔고, 대장 D-이월(«H2 SUPER_CUTOFF 레시피
  변경») 이 이 축의 조정을 fine-tuning 국면으로 미뤄 놓았다.
- **축 2 — 자기일관성**: 선택된 형상과 f-가중·노브·배너·헤더 계약이 **서로 맞는가.**
  반상태는 이 축의 결함이다: 형상은 SL 인데 f≡1(어느 관례에도 없음, §5-3) + 행렬 무가중
  조립(§0) + 배너 거짓말 + 헤더 계약(`lumina.h:723-724`) 위반.

**고려사항 1 의 판정 — «CMFGEN 발자국» 지침이 적용되는가**: **축 1 에는 적용되고, 축 2 는
보호하지 않는다.** CMFGEN 자신이 SL 을 푼다(§5-3 [저자 실행]) — 따라서 «재현 전 개량 금지»
지침 아래서 **full-FL 복원(A안)은 오히려 발자국 이탈**(CMFGEN 이 풀지 않는 것을 푸는 개량)
이다. 반면 f≡1 은 CMFGEN 의 단점 계승이 아니다 — CMFGEN 에 그런 상태가 없다. ⟹ 반상태는
«CMFGEN 을 따라가려던 기계의 반쪽 활성화» 이며, 성격은 **«노브가 거짓말하는 결함»** 쪽이다.
두 성격이 섞여 보였던 이유는 축을 안 갈랐기 때문이다.

---

## §4. 물음 2 — 갈림길 열거 (증거 · 비용 · falsifier)

### A안 — 항등 복원 (13,015 FL solve)

- **내용**: `fl_to_super` 항등 회복. 실현 경로 3종(§0 신규 실측 2): src 수리(V5 접촉) /
  덱 재생성(super_level 열 제거 — 덱 해시 앵커 전멸 동반) / env 트릭 `K=99999`(+ 런처
  preflight `==100` 수정). **env 제거만으로는 불가.**
- **증거(찬)**: 헤더 계약 문면(`super_mode==0 → 항등·byte-동일`)의 문자 그대로 회복.
  FL-단위 응답이 측정 가능해짐.
- **비용**: ⓐ solve LU 비용 **~5.9×10³ ×** [산술: 이온쌍 FL 차원 {Si 272·Ca 294·S 578·
  Fe 4,099·Co 5,772·Ni 2,000} vs SL {202×5·163}, ΣN³ 비 2.69e11/4.55e7], 조립(N²) ~2.4×10² ×,
  최대 행렬 5,772² ≈ 266 MB/셸 ⓑ ★**도입 사유의 재개봉**: 컷오프는 full 준위 행렬의
  ~250-order 스팬 ill-conditioning 때문에 도입됐다(C48 고고학·로더 주석) — **full-FL 해가
  현 솔버에서 존재·수렴하는지 자체가 미검증** ⓒ CMFGEN 발자국 이탈(§3) ⓓ bit-앵커 전멸(§6)
  ⓔ **06-08 이후 어느 시대와도 다른 신형상** — 대장 전체가 본 적 없는 구성이라 재감사 부담
  최대.
- **falsifier**: 소형 픽스처(Fe II/III 쌍 1셸)로 FL-행렬 조건수·완주·비용 실측 — 조건수가
  감당되면 ⓑ 논거 소멸, 비용 실측이 ⓐ 산술을 교정.

### B안 — 의도된 SL 모드로 승격 (`LUMINA_SUPER_LEVELS=1`)

- **내용**: env 1행. `super_mode=1` → f = 준위별 Boltzmann(SL 앵커 기준,
  `population_contract.c:241-259`) — 재분배와 행렬 가중이 **동시에** 고쳐진다(§0: 둘 다
  `within_sl_frac` 소비). src 무접촉(V5 보존). 배너 자동으로 `ACTIVE` (정직해짐).
- **증거(찬)**: ⓐ 코드 자신의 활성화 지시(배너·주석 "Activate together with") ⓑ 77본
  스크립트의 활성화 의도 흔적 + 런처의 topology 선언(§1) ⓒ **7월 검증 계보가 정확히 이
  모드였다**(ACTIVE 배너 실측) — C 본표와 모드 연속 ⓓ CMFGEN·ARTIS 관례 정합(§5-3)
  ⓔ F1b 가 수리-후 비교의 前항을 이미 확보.
- **효과 예측(반증 가능한 형태)** [저자 실행 — 덱 전수 산술]: 같은 lump 안 등g 쌍이라도
  **에너지가 bit-동일한 쌍은 인벤토리 전체에서 0행**(1,537,729행 전수: 등g 539,149 ·
  등g∧등E **0**) ⟹ Boltzmann f 에서 `n_l/g_l = n_u/g_u` 의 대수 필연 계급이 **소멸**한다
  — hazard 221,382 는 구조 계급으로는 0 이 되어야 한다(잔여는 부동소수 우연 충돌 계급뿐
  [추정 — exp 왕복의 비트 충돌, 사전 계산 불가]). NEG(위장 inversion) 계급도 동일 논리로
  소멸. F1b 의 BL 동결 계급은 «τ 실재·T_e-계 응답» 계급으로 전환된다.
- **비용·한계**: ⓐ bit-앵커 전멸(§6 — A안과 동일 계급) ⓑ lumped 준위 내부비는 여전히
  Boltzmann(T_e) — **준위별 NLTE 가 아니라 SL 근사의 본질**(CMFGEN 도 같은 근사를 쓴다,
  §5-3) ⓒ Lumina 구현은 CMFGEN 대비 occupation probability(W)·보간열(INT_SEQ) 부재
  [저자 실행 — §5-3] — 관례의 부분 구현 ⓓ solve 비용 불변(차원 동일).
- **falsifier**: 승격 1런 + F1b형 재판독 — «hazard 구조 계급 0» 예측이 깨지면(등E 0 실측과
  모순되는 hazard 잔존) 이 안의 기전 이해가 틀린 것.

### C안 — 현상 유지 + 잣대 한정 (모든 결론에 «SL-단위·반상태» 병기)

- **내용**: 아무것도 바꾸지 않고, 측정·판정의 문면에 한정을 강제한다.
- **증거(찬)**: ⓐ F1b 가 실제로 이 길로 본선 첫 관측을 착지시켰다 — 반상태 아래서도 응답
  가능 계급 790행의 측정이 성립했다 ⓑ L6C-R 계약 §3-1 의 5근거(«계약의 대상은 실물 생산
  경로» · 응답 자유도가 분모의 89.3% · 동결은 이름 있는 계급으로 완전 분리 · 수리는 이 단이
  결정 못 함 · 지금 재야 前항이 생긴다) ⓒ 앵커 계보 보존(§6 비용 회피 — 정확히는 **이연**).
- **비용**: ⓐ 비물리 인공물의 지속 — 인벤토리 선 1,537,729행(덱의 59.4%)의 τ·S 가 g-비
  아티팩트로 결정되고(τ=0 539,149·위장 inversion 포함), 이것이 **모든 후속 측정·수렴·
  스펙트럼**에 실린다 ⓑ SL 방정식 무가중 조립(§0) — f≡1 은 어떤 물리 모드도 아니므로
  «물리 모드 하나»(NO-OVERFITTING) 규약과 긴장 ⓒ ★**앵커 부채의 복리**: 반상태 위에 본선
  단(수렴·자유-T·타셸)이 쌓일수록 §6 의 전멸 목록이 자란다 — 지금의 «보존»은 미래의 더 큰
  전멸이다 ⓓ 자기모순 배너의 지속(6주간 반복 오도의 실적이 이미 있다 — BITLOCK 이전까지
  아무도 못 봤다).
- **falsifier**: 승격-vs-반상태 1쌍 차동에서 관측 하중 계량(응답 분포·스펙트럼)이 실질
  무차이면 유지 정당 — 사전 기대는 반대다(행렬 가중이 다르다, §0).

### D안 — 정직화만 (배너·헤더 계약 수리, 값 무변)

`:16482-16484` 배너와 `lumina.h:723-724` 문면을 실태에 맞게 고친다(소규모 src 접촉, 인구
불변 — 앵커 보존). **처분이 아니라 위생**이다 — A·B·C 어느 안과도 병행 가능하고, C안을
고르는 경우에는 **필수 동반**을 권고한다(거짓 배너를 알고도 두는 것은 다음 관측자에 대한
함정 존치다) [판단].

### E안 — 덱-원생 f_to_s 복귀 + 승격 (CMFGEN 최충실 형상)

컷오프를 빼고 CMFGEN f_to_s 층위(Fe II 342 SL 등)로 + Boltzmann f. **축 1 의 변경**이다 —
D-이월 H2 계급(fine-tuning 국면 몫)이고, 컷오프 도입 사유(일부 이온이 f_to_s 아래서 항등
잔존 → ill-conditioning)를 재개봉한다. **이번 갈림길의 답 후보가 아니라 fine-tuning 국면의
후보로 존치** [판단].

---

## §5. 반드시 다룰 고려사항 셋 (과제 명시 요구)

### 5-1. «CMFGEN 발자국 그대로» 지침의 적용 여부

§3 에서 판정했다: **적용된다 — 단 축 1(형상)에만.** 반상태(축 2)는 «CMFGEN 을 따라간
단점»이 아니라 «노브가 거짓말하는 결함»이다 — CMFGEN 에는 f≡1 상태가 존재하지 않는다.
역으로, 이 지침은 **A안(항등 복원)을 견제한다**: full-FL 은 CMFGEN 이 풀지 않는 개량이고,
재현 전 개량은 차이의 귀속을 불가능하게 한다.

### 5-2. bit-앵커 계보 — 선행 판정문의 언급 확인 + 비용 정량

**직접 확인 [저자 실행]**: BITLOCK 계약 §8-1(*"귀속 후 수리는 별도 단 — V5 권한·본선 영향
평가·bit-앵커 계보 단절 검토"*) · BITLOCK 판정 §6-4/§8 #4(*"V5·bit-앵커 계보 검토 필요"*) ·
★L6C-R 계약 §3-1 근거 4 — 가장 구체적이다: *"src/env 접촉 + **bit-앵커 계보 전멸**(hazard
221,382·앵커 행·SUMMARY 토큰 전부 반상태 위의 실측 — **수리하면 Z-O→Z-2A 사슬의 검증
기준선이 사라진다**)"*. 정량은 §6.

### 5-3. CMFGEN 자체의 super-level 관례 — 확인했다 (정성·소스 직독, 오라클 정량 인용 0)

**확인 가능했다** [저자 실행 — `/gpfs/kjhan/cmfgen_src/cur_cmf` 직독]:

- **CMFGEN**: `subs/sup_to_full_v3b.f` — SL해 → FL 재분배는
  `C2_F(I,K) = C2_S(L,K) * C2LTE_F_ON_S(I,K)`, 즉 **FL 인구 = SL 인구 × (SL 내부 LTE 분율)**.
  LTE 분율은 g·Boltzmann(T)·occupation probability(W, level dissolution) 포함(`:137-183`).
  = **SL 구성원의 departure coefficient 를 같게 두는 관례**(등분 b). 추가로 보간열
  (`INT_SEQ`)에 대해서는 b 를 에너지-보간(`:198-249`)해 상수-b 보다도 세밀하다. 총량 정합
  재정규화 후 종료.
- **ARTIS**: `../artis-ref/nltepop.cc:888` `population / superlevel_partfuncs[ion] *
  superlevel_boltzmann(…)` · `:1235` `s_renorm = superlevel_boltzmann/partfunc` — **SL 내부
  Boltzmann 분배.**
- **Lumina 와의 대응**: 덱 `super_level` 열 = CMFGEN `f_to_s` 직수입(§0). `super_mode=1`
  분기(`population_contract.c:241-259` — SL 앵커 대비 g·exp(−ΔE/kT))는 이 관례의 **부분
  구현**(W 부재·INT_SEQ 부재·anchor-기준 상대 에너지). `super_mode=0`+비항등 map(반상태의
  f≡1)에 해당하는 상태는 **CMFGEN·ARTIS 어느 쪽에도 없다.**

---

## §6. bit-앵커 전멸 비용 — 정량

**전멸 조건**: 인구를 바꾸는 모든 안(A·B — E 포함). C·D 는 이연(단 C 는 부채 복리, §4-C).

### 6-1. 사망하는 앵커 (등재처 명기)

| 앵커 | 값(대표) | 등재처 |
|---|---|---|
| Z-O hazard census | 221,382쌍 목록·다중도 {26:106,502·27:105,685·28:9,195}·상대변화 중앙 1.442e-2 | 대장 5회차·Z-O 판정 |
| zero-스트림 sha256 | `84f977bb…`/`64ccbb87…` (R-II≡census 결정론 거울) | F1b 부록 A-5 |
| 결정론 앵커 행 `line=669992` 4필드 bit | `n_upper=4.437688873013011e-4`(iter0)·`0x1.daa141c1395e6p-12`(iter1) 등 | Z-O E-Z3 · Z-2A E-I3/4 · L6C-R RD-6 · F1b — **등록 4곳** |
| BITLOCK lump hex 6종·D3 부호 이동값 3종·dmax=1 | `0x1.d1534dba20c26p-12` 외 | 대장 6회차 |
| Z-2A f_bit | 1.0 (442,764행 byte-동일) | 대장 7회차 |
| F1b 판별 전량 | ff=95/885 · 계급 census 8셀 · d 분포(최소 2.928690524872657e-6) · S/B med 28.186… | 대장 8회차 |
| K36 결정론 baseline | R1 residual 9.6662782724980344e-9(45회)·R2 8.1222406993212508e-9(52회)·R6 셀 109,014,300 대조 | B 대장(3회차) |
| DET-SPROD IV | S2 bit항등 1282/1282·총가중 721.726 | 진행부·L6 기대식 입력 |
| L6·A207 판정 수치 앵커 | S/B 중앙 0.9999927651925586·공통모드 6.66e-16·W4 3값 | L6·A207 판정문 |
| 3층 인수 프로토콜의 parity 마이크로픽스처 기준선 | (반상태 인구 위에서 bake 된 전 기준선) | 인수 인프라 |

### 6-2. 생존하는 것

**덱-계급 사실**(등g 221,382·등g∧등E 0·f_lu 값 — 덱 불변이면 영구) · **PC-2 census**(LTE
모델 — 비커버 이온용 존속) · **소스 감사 전건**(BITLOCK A1-A4·A207 W-A·死블록 경계) ·
**게이트·selftest·NC 픽스처**(합성) · **판정문·봉인 자체**(반상태 시대의 역사 기록으로 영구
유효 — 뒤집히는 확정 **0**, §2-4).

### 6-3. 재구축 비용의 실측 근거

새 baseline 사슬의 재봉인 = census 급 1런(**02:42:56**, a100×2 — job 324388 실측) + R-II 급
1런(**04:34:36**) ± L6 급(04:34:10) ⟹ **GPU 2~3런(각 3~5시간, a100×2) + 판독·판정·감리
사이클 1회분.** F1b 가 前항(계급 census·행별 d 전량)을 확보해 두었으므로 수리 직후 1런이
곧 **수리 전후 차동 측정**을 겸한다 — 재구축이 순비용이 아니라 측정이 된다.

### 6-4. 비용의 시간 구조 [판단]

전멸 목록은 **반상태 위에서 봉인이 하나 늘 때마다 자란다.** F1b 이후 본선의 다음 단(수렴·
자유-T·타셸·타이온)이 전부 반상태 위에 앵커를 얹게 되므로, 처분 결정은 **다음 본선 발주
전**이 가장 싸다. 지금 정하지 않는 것 자체가 이자를 무는 선택이다.

---

## §7. 물음 3 — 이 갈림길이 옳은 질문인가

**부분 기각한다.** «수리 A vs 승격 B vs 유지 C» 3자 택일로 물으면 축이 섞인다(§3):

1. **A안과 E안은 축 1(형상)의 질문**이다 — 형상은 런처가 fail-closed 로 선언해 온
   topology 이고, 그 재선정(H2)은 대장이 **fine-tuning 국면으로 이월**해 놓았다. CMFGEN
   발자국 지침도 같은 방향을 가리킨다(full-FL 은 CMFGEN 이 풀지 않는 개량). ⟹ **이 국면의
   갈림길에서 A·E 를 빼는 것이 옳다** — 단, A 는 «기각»이 아니라 «국면 밖 이월»이다(FL
   응답 측정이 필요해지는 국면이 오면 재개봉).
2. **남는 질문은 축 2 하나다**: «이미 선언된 K=100 형상의 f-가중을 일관되게 만들 것인가
   (B), 언제 만들 것인가(지금 vs C안의 이연)». 이것이 옳게 절단된 갈림길이다.
3. **그 앞에 다른 질문이 있는가**: «본선(A2-07 J̄ 응답 → 수렴 → 스펙트럼 → CMFGEN 재현)이
   FL-해상 인구를 요구하는가» — **아니오** [판단]: 목표 잣대인 CMFGEN 자신이 SL 을 풀고
   SL 내부를 LTE-분율로 채운다(§5-3). 따라서 반상태 처분이 본선을 막고 있지도, 본선이
   반상태 처분을 무의미하게 하지도 않는다 — 갈림길의 **제기 시점은 옳다**(BITLOCK #4 가
   물었고, §6-4 의 시간 구조가 지금 답하기를 요구한다). 재절단만 필요하다.

---

## §8. 권고

**증거로 강제되는 부분** (user 가 어느 안을 골라도 성립):

- 반상태는 결함이다 — 자기모순 배너 + 헤더 계약 위반 + 어느 관례에도 없는 f≡1 + SL 방정식
  무가중 조립(§0). «CMFGEN 발자국» 지침의 보호를 받지 않는다(§5-1).
- 어떤 인구-변경 수리든 §6-1 의 앵커 전멸을 수반하되, **확정 결론은 하나도 뒤집지 않는다**
  (§2-4). 前항(F1b)·덱 산술 예측(§4-B)·재구축 비용 실측 근거(§6-3)가 이미 갖춰져 있다.
- 잣대-한정 병기(C안의 문면 규율)는 이미 F1b 가 시행 중이므로, «아무 조치 없음»이라는
  선택지는 실재하지 않는다 — C안을 골라도 D안(정직화)은 병행해야 한다(§4-D) [이 마지막
  문장은 판단].

**저자의 판단** (강제 아님 — user 가 다르게 고를 수 있게 각 안의 결정 근거를 §4 에 나란히
남겼다):

> **B안(`LUMINA_SUPER_LEVELS=1` 승격)을, 다음 본선 단 발주 전에, BITLOCK #3 의 11이온
> 재감사 단과 한 묶음으로 집행하는 것이 유력하다.**

이유 다섯: ① env 1행 — src 무접촉(V5 보존), 수리 계급 중 최소 침습 ② CMFGEN·ARTIS 관례로의
복귀이자 **7월 검증 계보(ACTIVE 모드)와의 모드 재접속** — 대장 C 본표가 이 모드 위에서 살아
있다 ③ hazard·위장 inversion 의 구조 계급 소멸이 덱 산술로 사전 예측되고(등g∧등E = 0
[저자 실행]) 그 예측 자체가 falsifier 다 ④ F1b 前항 덕에 수리 직후 1런이 곧 차동 측정이
된다 ⑤ 이연(C안)은 앵커 부채의 복리를 문다(§6-4).

**user 가 C안(이연)을 고를 결정적 근거** — F1b 실증: 반상태 아래서도 응답 가능 계급 790행의
본선 측정이 성립했다. 본선 최우선(수렴·자유-T)을 반상태 위에서 몇 단 더 진행하고 수리를
그 뒤로 미루는 것은 **일관된 선택**이다 — 대가는 §6-4 한 줄이다.

**user 가 A안(항등)을 고를 결정적 근거** — 헤더 계약 문면의 문자 그대로 회복 + FL-단위
응답의 측정 가능성. 대가는 ~5.9×10³× 비용·ill-conditioning 재개봉·발자국 이탈·최대 재감사
(§4-A) — 그리고 이 셋은 소형 픽스처 실측(falsifier)으로 먼저 잴 수 있다.

---

## 부록 A. 저자 실행 명령 (전량; 봉인 read-only)

```bash
# §0 소스 재확인 (HEAD af7441d; src 는 a849306 과 동일 — diff 공백)
git log -1 --format="%H %ci" ; git diff --stat a849306 HEAD -- src/
sed -n '1438,1482p' src/lumina_atomic.c            # 로더: 열 로드·항등 기본·컷오프 블록(:1468/:1470/:1473-1474/:1476-1477)
sed -n '16344,16352p;16380,16390p;16454,16460p;16482,16486p' src/lumina_plasma.c
sed -n '160,260p' src/population_contract.c        # f 빌더: !super_mode → 1.0 / super_mode → Boltzmann
sed -n '19384,19440p' src/lumina_plasma.c          # 재분배 + 정규화
sed -n '16975,17005p' src/lumina_plasma.c          # ★SOLVE_OF/FRAC_OF — f 가 행렬 조립에 들어감
sed -n '720,728p' src/lumina.h                     # 헤더 계약
grep -n 'getenv("SUPER_LEVELS")' src/ -r           # → 0건 (bare 노브는 死노브)
grep -n "super_mode" src/lumina_plasma.c           # → 16350·16484·19642 (전수)
# 死블록 (자체 awk 깊이 오토마톤)
awk 'NR>=1300&&NR<=1700{...depth...}' src/lumina_plasma.c   # → 1324 open / 1693 close
awk 'NR>=1080&&NR<=2070{...depth...}' src/lumina_cuda.cu    # → 1083 open / 2063 close

# §1 계보
git show -s --format="%ci %s" 9a4e91d 66586d6 a97d0e1
grep -rln "LUMINA_SUPER_LEVELS" scripts/*.sh scripts/*.slurm | wc -l        # → 26 (launch_parity 17/18 포함)
grep -rn " SUPER_LEVELS=" scripts/*.sh | grep -v LUMINA_ | wc -l            # → 77 (bare — 死노브)
grep -rL "SUPER_LEVELS" $(grep -rl "SUPER_CUTOFF" scripts/)                 # → det_convergence 등 3건만
sed -n '270,290p' scripts/run_det_convergence_2026-08-08.slurm              # → CUTOFF==100 fail-closed + topology 선언
grep -n "SUPER" docs/PARITY59_INSTR.env                                     # → 양 노브 + bare
for p in 45 47 49 57; do grep -m1 'Super-levels' logs/coevolve_consume_parity$p/stdout.log; done
#  → 전건 "ACTIVE (21038 FL -> 2828 SL)"
grep -m3 -i super logs/coevolve_consume_parity44/stdout.log                 # → SUPER_LEVELS=1·LUMINA_SUPER_LEVELS=1
grep -m1 "Super-levels" /gpfs/kjhan/lumina/a210_cancellation_census_mgpu/det1234_*/stdout.log
#  → "off (identity) (13015 FL -> 1173 SL)" (08-15)
for d in <census> <R-II> <L6C> <L6>; do grep -E "SUPER|ELEMENT_WIDE" $d/input/resolved_lumina.exports; done
#  → 전건 LUMINA_SUPER_CUTOFF="100" 단독
grep -m1 "Super-levels" <L6|L6C>/stdout.log                                 # → off (identity) 13015→1173

# §2-4 ②·§4-B 덱 전수 census (grammar-debug, nested ssh + stdin 파이프; read-only)
# 인벤토리 12슬롯 · 양준위 lumped(level_number>=100) 선 전수 + 등g + 등E(bit) 계수:
ssh grammar "ssh grammar-debug 'python3 -'" < (csv 스트리밍 조인 스크립트)
#  → lumped-pair lines = 1,537,729 (덱 2,588,798 의 59.4%) · eq-g = 539,149 · eq-g∧eq-E(bit) = 0

# §5-3 CMFGEN·ARTIS 관례 (정성 직독 — 오라클 산출물 아님)
sed -n '1,260p' /gpfs/kjhan/cmfgen_src/cur_cmf/subs/sup_to_full_v3b.f       # 등분-b·LTE-분율·W·INT_SEQ
grep -n "superlevel_boltzmann\|s_renorm" ../artis-ref/nltepop.cc            # :888·:1235 — Boltzmann 분배
grep -n "f_to_s" scripts/expand_atomic_data_cmfgen.py                       # :159-190 — 덱 열 = CMFGEN f_to_s 직수입

# §6-3 재구축 비용 근거 = 선행 판정문 등재 실측치 인용 (Z-O: job 324388 02:42:56 /
#  Z-2A: 324557 04:34:36 / L6: 321636 04:34:10) — 본 문서에서 sacct 재실행하지 않음
```

**[미확인] 목록 (전수)**: SUPER_LEVELS 가 K36 계보 env 에서 탈락한 정확한 커밋·의도(08-05~08
Codex 자율 창) · A-3/A-6 행의 실행 exports(무관 판정은 스코프·차동 논거로 성립) · C 본표
절대-정량 3행의 의도-SL 럼핑 민감성(별개 축, H2/D 계급) · B안 승격 후 부동소수 우연 충돌
계급의 크기(사전 계산 불가 — falsifier 런이 잰다).
