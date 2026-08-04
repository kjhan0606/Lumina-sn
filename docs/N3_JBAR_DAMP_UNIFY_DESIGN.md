# N3 수리 설계: jbar 감쇠 분열 통일 게이트 (LUMINA_JBAR_DAMP_UNIFY)

작성 2026-07-30, 역할=N3 수리 설계 dig(fable). READ-ONLY 산출물 — 코드 수정 없음.
결함 정본: `validation/cmfgen_toy06_19p48d/analysis/physics_wiring_audit/REPORT.md:6` —
"(+N3: jbar EMA0.5 vs jblue raw)"; 같은 파일 :19 — "0.5 감쇠 자체는 합법(ARTIS 부류)",
결함은 **분열**(같은 MC 장 추정치를 소비자마다 다른 세대로 소비).
행번호 기준: 현 워킹트리(브랜치 thenmc-macroatom-fluorescence, src 미커밋본). 감사 REPORT의
:3714 등 구번호는 드리프트되어 있어 아래는 전부 재실측 번호다.

---

## 1. LUMINA 분열 지도 (생산 1 · 소비자 N, 전수 실측)

### 1.1 생산자 — 탤리는 하나, 배열은 둘, 후처리가 갈라짐

| 단계 | jbar_line | jblue_line | 근거 (file:line) |
|---|---|---|---|
| GPU 탤리 (동일 crossing, **동일 증분** `pkt_energy*doppler_factor`) | `g_jbar_line` += , `g_jbar_count` += 1 | `g_jblue_line` += (count 배열 없음) | `src/lumina_cuda.cu:3541-3544` / `:3549-3551` |
| 무장 조건 | co-evolve면 항상 (`g_coevolve`) | `LUMINA_IUP_JBLUE` env **또는** `artis_parity_enabled()` | `:6974-6979` / `:6982-6987` |
| 호스트 정규화 (동일 계수 `c·t_exp/(4π·t_sim·ν_l·V)`) | `:7945-7959` | `:8015-8027` | — |
| **이터 간 감쇠** | **EMA**: `jbar = f·new + (1−f)·prev`, `f=LUMINA_COEVOLVE_JBAR_DAMP` (코드 default 1.0=off; **런처가 0.5 강제**: `scripts/run_coevolve_s01.sh:136`) | **없음 — 항상 raw** (블록 자체가 부재) | `:7986-8002` vs `:8015-8027` |
| raw 스냅샷 (관측자) | `g_jbar_raw_snap` — EMA **직전** 복사, `LUMINA_NLTE_FINAL_RESOLVE` 게이트 | — | `:7960-7981` |

증분이 문자 그대로 동일하므로(위 :3541-3551) 두 배열의 **유일한** 수치 차이는 EMA 유무다.
비-coevolve frozen THEN_MC 경로(`:9074-9092`)는 jbar를 raw로만 정규화(EMA 블록 없음) — 생산 경로 아님.

주의(배열 재사용 위험): 순수-CMFGEN 반증 게이트 `LUMINA_CMF_JINC_CONT`는 결정론 `cs.J`를
`opac->jbar_line`에 **덮어쓴다**(`src/lumina_cmfgen.c:3240-3285`) — 다른 생산자가 같은 배열을
쓰는 부류로, 본 게이트 구현 시 co-evolve 경로와 배타임을 확인하고 손대지 말 것(이 config 휴면).

### 1.2 소비자 전수 — 누가 EMA본, 누가 raw본

| # | 소비자 | 읽는 본 | 근거 (file:line) |
|---|---|---|---|
| C1 | **NLTE 행렬 조립 bb rate, mode-3** (`LUMINA_NLTE_JBAR_POPS=3`, CPU 전용 — 런처 `LUMINA_NLTE_ASSEMBLE_GPU=0`, `run_coevolve_s01.sh:127-129`) | **EMA본** `opacity->jbar_line`; 문턱 `jbar_min`=`LUMINA_JBAR_MIN`(생산 3) else 10/50; 미달 시 binned `nlte_get_J_at_nu` 폴백 | `src/lumina_plasma.c:13834-13840` (읽기), `:13946-13951` (β·J_inc rate) |
| C2 | **MA internal-up `J_line` 읽기** (compute_transition_probabilities) | **EMA본** `opacity->jbar_line` (count ≥ `g_ctp_jbar_min_ma`=10 하드코딩; Y6 ON 시 JBAR_MIN); 미달 시 binned 폴백 | `:3978-3982`, 폴백 `:3984`; 문턱 선언 `:2937-2942`; Y6 `:3376-3399` |
| C3 | **MA internal-up IUP-JBLUE up-rate** (`(B_lu−B_ul·n_u/n_l)·β·J_blue`; parity 마스터게이트로 default-ON: `:3408-3413`, 커밋 192a2c3) | **raw본** `opacity->jblue_line`; 0/NULL이면 `J_line`(=C2의 EMA본 or binned) 폴백 | `:4012-4013` (읽기), `:4057` (폴백), `:4070-4089` (rate) |
| C4 | JBLUE-ANCHOR/ANCHOR2 진단 — raw jblue vs `J_line` log-비 | raw vs EMA **혼합 비교** (잣대 주의: anchor 오프셋에 EMA 지연이 섞임) | `:4016-4055` |
| C5 | JBAR_DUMP 관측자 (`LUMINA_JBAR_DUMP`, parity_baseline.env:48) | EMA본 (스스로 자백: "the EMA-consumed" 문구) | `:13516-13557` |
| C6 | FINAL_RESOLVE 관측자 — resolve_ema/resolve_raw 이중 재솔브 | 양쪽 모두 (본 결함 전용 A/B 계기, 기설치) | `src/lumina_cuda.cu:8486-8600` |
| C7 | 관측자-frame Sobolev 스펙트럼 소스 (`LUMINA_CMF_OBS_SRCJ=1`, default 0 → **이 config 휴면**) | EMA본 | `src/lumina_cmfgen.c:1299-1305`, 게이트 `:1365` |

**분열의 정확한 형태(생산 config: parity 런처 → `run_coevolve_s01.sh consume`)**:
같은 이터레이션 안에서 —
- 행렬 인구(C1)와 MA `J_line`(C2)은 **J̄의 EMA(0.5)본**을,
- MA internal-up 실효 rate(C3, parity에서 항상 무장)는 **같은 탤리의 raw본**을 소비한다.
- 추가 세부 분열: C1/C2의 **문턱 미달 폴백**은 parity C3-게이트에서 nlte.J_nu = 당-이터 MC per-bin 장
  (raw; `src/lumina_cuda.cu:7093-7095` 우회 + `:7857-7866` 재구축)이므로, **한 소비자 안에서도**
  count≥문턱 선은 EMA장, 미달 선은 raw장 — 세대 혼식이 문턱을 경계로 재발한다.
  (비-parity에서는 폴백이 `LUMINA_J_DAMP=0.5` 감쇠된 cs.J — `:7058-7082` — 이면 반대 방향 혼식.)

## 2. ARTIS 원배선 실측 (../artis-ref/, 실소스)

### 2.1 추정기: raw 단일본, EMA 부재

- 누적: 패킷이 선 공명 도달 시 `update_lineestimator` — `rpkt.cc:156-158`(상호작용), `:184-186`(통과),
  `radfield.cc:709-714` (`Jb_lu_raw += increment`).
- 정규화: `radfield.cc:819-826` `normalise_J`:
  `prev_Jb_lu_normed[i].value = Jb_lu_raw[i].value * estimator_normfactor_over4pi` — **순수 덮어쓰기.
  이전 값과의 블렌딩(EMA) 없음.** "prev"는 감쇠가 아니라 "직전 타임스텝에 수집, 현 타임스텝에 소비"라는
  1-스텝 지연(lagged Λ)의 이름이다(`radfield.cc:81` 주석 "value from the previous timestep").
- 소비: `get_Jb_lu`(`radfield.cc:650-654`)가 유일 접근자.

### 2.2 소비자: 단일 함수 → 분열이 구조적으로 불가능

`rad_excitation_ratecoeff`(`macroatom.cc:571-604`; detailed 분기 `:588-593`
`return R_over_J_nu * radfield::get_Jb_lu(...)`, 폴백 `:596` `radfield(nu_trans)`)를
**세 소비처가 공유**한다:
- MA internal-up: `macroatom.cc:125`
- NLTE rate 행렬: `nltepop.cc:538`
- 비열적: `nonthermal.cc:1685`

즉 ARTIS에서 "행렬은 감쇠본, MA는 raw본" 같은 분열은 **함수가 하나이므로 성립 자체가 불가능**하다.
REPORT.md:19의 "CMFGEN은 α·Γ가 같은 σ의 DB 짝이라 K11 부류가 구조적으로 불가능"과 동형 논리.

### 2.3 ARTIS의 감쇠 장치는 존재하되 — 의도적으로 꺼져 있고, 선 추정기에는 아예 없다

- `titer_J`(`radfield.cc:897-903`): `J = (J + J_saved)/2` — 정확히 0.5 블렌드. 그러나
  (a) `#ifdef DO_TITER` — **레포 전체에 `-DDO_TITER`/`#define DO_TITER` 부재**(grep 전수: ifdef 지점과
  주석 `radfield.cc:600` "used to damp out fluctuations ... if DO_TITER is defined"뿐),
  (b) `sn3d.cc:943` `globals::n_titer = 1` 하드와이어 + `:947-948` non-DO_TITER 시 `assert(n_titer==1)`,
  (c) 켜져도 대상은 **full-spectrum J/nuJ뿐** — `Jb_lu`와 multibin `J_raw`는 건드리지 않는다
  (호출부 `update_grid.cc:444`; `normalise_J`의 Jb_lu 덮어쓰기는 무조건부).
- 판독: ARTIS는 (i) 감쇠를 전역-J 한정 **선택 사양**으로 만들었고 그것도 기본 봉인, (ii) **선 추정기에는
  감쇠 옵션 자체를 제공한 적이 없다**. 시간-전진 코드라 타임스텝 지연이 곧 유일한 평활이다.
- 참고: 배포 artisoptions 전 변형에서 `DETAILED_LINE_ESTIMATORS_ON = false`
  (`artisoptions.h:74` 등 8개 파일) — OFF면 전 소비자가 binned `radfield(nu_trans)`을 읽어 역시 단일본.
  ON일 때 선정 기준은 Fe, lowerlevel≤15, A_ul>0 (`radfield.cc:496-526`).

### 2.4 LUMINA IUP-JBLUE가 미러하는 쪽

`src/lumina_plasma.c:2961` 주석이 `macroatom.cc:591`(get_Jb_lu 반환식)을 축자 인용 — IUP-JBLUE(C3)는
ARTIS **raw-lagged 소비**의 미러가 맞다. 어긋난 쪽은 행렬(C1)·`J_line`(C2)의 EMA 소비다:
ARTIS라면 nltepop.cc:538이 같은 raw본을 읽는다.

**판별 결론: ARTIS-충실 통일 방향 = raw** (전 소비자가 같은 1-이터-지연 raw 추정치).

## 3. 오프라인 진폭 실측 (기존 덤프, 신규 런 없음)

parity42 보존물 `logs/coevolve_consume_parity42/lumina_levelpop_resolve_{ema,raw}.csv`
(C6 계기 산출, 2026-07-29 11:14, 각 1,051,900행) 대조 — 같은 수렴 상태에서 행렬만
EMA장/raw장으로 재솔브한 순수 장-효과:

- 전체 213,454 (shell,Z,ion,level) 쌍: median |Δlog₁₀ n| = 0.007 dex, p95 = 0.12 dex — 본체는 미소.
- **꼬리가 정확히 결함 예측 지점에 있다**: |Δlog|>1 dex가 172항목, Si III(Z=14 ion2)·S III(Z=16 ion2)
  들뜬준위에 집중, 최대 ~37 dex — 1e-30 위생플로어로의/로부터의 **존재 자체 플립**
  (예: s11 S III lev2 2.3e6 → 3.1e-31; s19 S III lev7 5.9e-31 → 1.3e0; s11 Si III lev20-22
  E≈24.7 eV 플로어 → 1e-2 대). parity26-diag의 Si III ¹P° 부양(1113Å 형광 초열장) 가족과 동일 계열.
- 함의: EMA-vs-raw 선택은 평균 인구가 아니라 **희소-선/문턱-인접 준위의 점등 여부**를 바꾼다 —
  N2(문턱 분열)와 N3(감쇠 분열)이 같은 꼬리에서 결합 작용. 1e-30 플립은 플로어 잣대 오염 가능성이
  있으므로(클램프 대장 참조) 판정 메트릭으로는 log-비가 아닌 "플립 카운트"를 쓴다. 수치는 대장 기재용
  관찰이며 여기서 어떤 튜닝도 하지 않는다.

## 4. 통일 게이트 스펙 (제안)

- **env**: `LUMINA_JBAR_DAMP_UNIFY` — unset/0 = OFF(현행 유지, byte-identical), `1` = raw-통일
  (ARTIS-충실, 권고), `2` = EMA-통일(예비 arm; §6 위험 발현 시에만).
- **default OFF-동일성 요건**: 게이트 미설정 시 실행 경로·RNG·산출물 전부 byte-identical.
  변경 지점이 전부 "게이트 ON일 때만 분기 진입"형이므로 기계 검증은 OFF-런 diff로 족함.

### ON(=1, raw-통일) 시 바뀌는 지점 — 전부 실측 지점

| 지점 | 변경 | file:line |
|---|---|---|
| 1 | co-evolve consume 블록의 EMA 블렌드를 **건너뜀** (`jbar_damp`를 1.0로 강제 + `[JBAR-DAMP-UNIFY]` 배너 1회 출력). 결과: `opacity.jbar_line` = raw = `jblue_line`과 수치 동일(§1.1 동일 증분·동일 정규화) → C1·C2·C3·C5·C7 전 소비자가 자동으로 같은 raw본 | `src/lumina_cuda.cu:7986-8002` (유일한 EMA 지점) |
| 2 | (파생, 코드 0) C6 resolve_ema/resolve_raw는 동일장 이중솔브가 됨 — **기계 자기검증**: 두 CSV 일치가 게이트 작동의 증명 | `src/lumina_cuda.cu:8486-8600` |

즉 최소 구현은 **한 지점**이다. 주의: 이는 `LUMINA_COEVOLVE_JBAR_DAMP=1.0` 설정과 수치 등가지만,
별도 명명 게이트로 두는 이유는 (i) 판정런 RESOLVED CONFIG에 의도가 명시되고, (ii) 런처가
`${LUMINA_COEVOLVE_JBAR_DAMP:-0.5}`로 감쇠를 재주입하는 사고(`run_coevolve_s01.sh:136`)를 차단하며,
(iii) `=2` 방향과 하나의 스위치로 묶기 위함. 구현 시 두 env 충돌은 명시 우선순위+경고 배너로 처리.

ON(=2, EMA-통일) 시: `:8015-8027` jblue 정규화 직후에 `:7986-8002`와 동형의 EMA 블록(자체 prev 버퍼,
같은 f)을 추가. C3·C4가 EMA본을 읽게 됨. ARTIS 근거 없음(§2.3) — parity 사다리 기본안 아님.

### 잔여 비통일 (이 게이트가 닫지 않는 것 — 명시)

- N2 문턱 분열(C1 jbar_min=3 vs C2 하드코딩 10)은 **Y6**(`LUMINA_JBAR_UNIFY`, plasma.c:3376-3399)의
  관할. §7 상호작용 참조.
- 문턱 미달 선의 binned 폴백 장(§1.2 말미)은 별도 사안(N6 인접) — 본 게이트 범위 밖, 대장 기재만.
- C3의 stim 보정 인구=dilute-Boltz 전셸핀은 N5 별건(plasma.c:4074-4087).

## 5. 판정런 사전등록 초안 (offline-first 3요건 충족 상태)

기전 특정=§1(분열 실측) + §2(ARTIS 원배선) 완료, 수리안 검증=§4(1지점, OFF-동일성 자명),
기대치 사전등록=아래. 발주는 운전석 승인 후 1회.

- **Arms**: A=현행 champion config(EMA0.5+raw 분열; parity42 계열), B=A+`LUMINA_JBAR_DAMP_UNIFY=1`.
  (C=EMA-통일은 B가 HS-1에 걸릴 때만 후속 등록.)
- **W (wiring, hard)**: B의 stdout에 `[JBAR-DAMP-UNIFY]` 배너 1회 + RESOLVED CONFIG에 env 존재.
  그리고 `lumina_levelpop_resolve_ema.csv` ≡ `lumina_levelpop_resolve_raw.csv`(b_k 열 상대차 < 1e-10) —
  불일치 시 게이트 미작동 = **런 VOID**(부정 결과 아님).
- **I-1 (불변량, 등록 가능)**: JBLUE-ANCHOR thin 버킷 |log-mean| 급감. raw-통일에서 count≥문턱 선은
  `J_line == J_blue` 항등이므로 잔여 오프셋은 폴백/문턱-미달 선만 기여(plasma.c:4019-4031 실측 정의 기준).
- **D-1 (방향)**: §3의 "1e-30 플립" 172항목 — B에서 Si III/S III 들뜬준위 인구가 **raw-솔브 쪽**
  (parity42 resolve_raw 값)으로 이동. 정합 판정은 CMFGEN toy06 지상진리 대비 divergence 지도 형식으로.
- **M (특성화, 판정 아님)**: b4(Si III ¹P° 부양 잣대, parity26-diag 정의) 재측정 / s8 이온분율표
  (Fe IV·S II 등 parity42 잣대 유지) / `scripts/jbl_verdict.py` 밴드 배터리 / **진동 계측**:
  최종 4이터의 소비 jbar장 per-iter 상대 L1 변화열(EMA 제거가 자극하는 flicker의 정량 — §6).
- **HS (hard stop)**: HS-1 진동 — B의 마지막 3이터 L1 변화 평균이 A의 2배 초과 & 비수축 → 불안정 판정,
  튜닝 금지, C-arm 결정으로 회부. HS-2 NaN/음수 pops 또는 FORMAL-CONS 위반. HS-3 W 실패=VOID.

## 6. 위험 — EMA는 안정장치이기도 하다

- **raw-통일의 진동 위험 (실재 이력 있음)**: EMA0.5는 장식이 아니라 "far-outer hot-band runaway"
  대책으로 도입됐다(`run_coevolve_s01.sh:136` 주석; epay25 자기강화 UV 루프 — `src/lumina_cuda.cu:7984-7985`
  주석, `LUMINA_J_DAMP` 미러). 구조 차이가 본질: **ARTIS는 시간-전진**(타임스텝마다 물리 상태가
  전진하므로 1-스텝 지연 raw 소비가 자연 평활)이고, **LUMINA co-evolve는 단일 스냅샷(19.48d) 고정점
  반복**이라 무감쇠 lagged-Λ가 진동/폭주할 수 있다. 그래서 HS-1을 사전등록한다.
- **EMA-통일의 지연 편향**: f=0.5는 유효 시정수 ~1/f 이터의 과거장 혼입 — §3이 보인 대로 문턱-인접
  준위의 점등을 과거장이 결정하게 됨 + ARTIS 원배선 근거 부재(§2.3).
- **UNRESOLVED-1**: ARTIS가 raw를 "선택한 이유"의 명문 근거는 코드에 없다 — 실측 가능한 것은
  (i) 선 추정기에 감쇠 옵션이 없다는 사실(§2.1-2.3), (ii) 전역-J 감쇠(titer)를 만들었다가 전 배포
  config에서 봉인했다는 사실뿐. "시간-전진이라 불필요"는 구조 정합적 **해석**이며 코드 주석 인용으로
  단정할 수 없다. 따라서 방향 권고(raw)는 "ARTIS-충실"(원배선 재현) 근거로만 서고, 안정성 우위 주장은
  하지 않는다.
- **UNRESOLVED-2**: 현행 parity 스택(진화된 게이트 다수)에서 hot-band runaway가 재발할지는 오프라인
  판정 불가 — 판정런 HS-1이 결정한다. 재발 시 처분(=2 채택 vs 별도 안정화)은 그때 별도 설계.

## 7. Y6(LUMINA_JBAR_UNIFY)과의 상호작용

- Y6은 **문턱**(N2)을, 본 게이트는 **세대**(N3)를 통일 — 직교. 둘 다 OFF-default라 독립 A/B 가능.
- **동시 ON의 의미**: count ≥ JBAR_MIN(3)인 모든 (line,shell)에서 C1·C2·C3이 **동일한 raw J̄** 소비 —
  ARTIS의 "단일 함수·단일 배열" 상태(§2.2)와 소비 의미론이 일치(잔여: §4 잔여 비통일 항목).
  N2+N3 완전 폐합 판정런은 Y6+본 게이트 동시 ON arm으로 등록하는 것이 맞다.
- 본 게이트만 ON(Y6 OFF)이면 3-9회 교차 선은 여전히 C1(신뢰)·C2(폴백) 분기 — raw-통일이어도
  그 대역은 "같은 장의 다른 축약"(per-line raw vs binned)을 읽는다. 판정 해석 시 혼동 금지.
