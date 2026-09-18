# 사전등록 — DET-DR-ON: DR 부스트 노브를 켜면 중간 셸 황 이온화가 오프라인 예측대로 이동하는가 (2026-09-19)

기안 = **Fable fresh**(`claude-fable-5-1`, 개정25-b). 상태 = **기안 / 경량 사전감사 대기(개정25-d, Q-GOAL·Q-LEAN 전용)**. 실행 승인이 아니다.
근거 = `docs/BRIEF_DR_ON_2026-09-19.md`(운전자 브리프, fb59d60) · `docs/READOUT_CONT_NLTE_SUPERTHERMAL_2026-09-18.md` §4(DR=0 근인) · 직전 계약 `docs/PREREG_DET_CONT_NLTE_BALANCE_2026-09-18.md`(판정 미완·취소, 형식 계승). base = 판정런 **376251**(`/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/run5_h200`, DR off, Kirchhoff-LTE 연속체 소스, 램프 소등 상태).

## 1. 계약 — 물리 계약 1개

**「안정 base 소스(Kirchhoff-LTE 연속체, DET-LTEUNMAP-EPS 수리 유지) 위에서 DR 을 켜면(`LUMINA_FROZENIN_DR=1` + 부스트 노브 4 키 =1) 중간 셸 황 II→III 이온화 비 r 이 오프라인 율비 예측 r/(1+α_DR/α_RR) 대로 이동하며, 그 이동량이 DR-off 가 중간 셸 저재결합에 기여한 몫이다.」**
판정런 1회(twin-T_e, 반복수는 §6 결정 항목). A/B 대조군 = base 376251(이미 존재) ⟹ 차이 = DR 토글(+ 채널 덤프 게이트, 출력 파일만).
⚠**이 계약이 말하지 않는 것**: CMFGEN 도달 여부(오프라인 예측 자체가 4–20× 미달 — 이 단은 **필요·불충분 확인**이다), Ca IV/V(DR_TABLE 항 부재), Si·Ca 착지의 정량, 스펙트럼, 자유-T_e, 초열 CONT 소스(폐기·별건).

### Q-GOAL (기안자 자기 답 — 경량 사전감사가 독립 재판정)
① 줄이는 칸: 어긋남 지도 **중간 셸 9–24 S II/III**(base f_II .017/.053/.102 대 CMFGEN .85/.80/.71). 예측 이동은 ×2.4(.042/.123/.221) — 칸을 **닫지 않고 부분만** 줄인다. DR 은 실제 물리 채널이고 캠페인 전 런에서 부스트 0 으로 **조용히 꺼져 있었다**(§2) — 켜지 않은 상태의 모든 CMFGEN 대조는 저재결합 편향 위에 있다. ② 산출 = 물리 측정(이온 분율·채널 율). 신규 계측 0·코드 0줄. ③ 더 가까운 대안 표적: 연속체 채움(NLTE 소스)은 직전 단에서 초열 재점화로 실패했고, 비열적·NLTE 승격 IV·V 는 이 이동 위에 얹혀야 귀속이 된다 — DR 은 그 모든 후속 단의 **공통 전제**라 먼저 켠다(user 09-19 「DR 켜기 먼저」). ④ 건너뛰면 이후 모든 단의 S/Si 재결합 분모가 RR-only 로 남아 어긋남 귀속이 DR 결손과 뒤섞인다.

### Q-LEAN (기안자 자기 답)
L1 물리량 = 셸별 S II/III 비 r(`lumina_ion_pops.csv`)와 채널 율 `ne_alpha_dr`·`ne_alpha_rr`(`lumina_ion_channels.csv`, 기존 덤프). 신규 계측 **0**, 코드 변경 **0줄**, 재빌드 **없음**. L2 게이트 3(G-ID·G-RUN·G-DR) — 각각 실재 가능한 실패(다른 키 혼입 / 미완주 / 부스트가 여전히 0). L3 = 1 걸음(env 변경 + 판정런 1). L4 삭제 후보 없음 — 추가하지 않은 것: 새 판독기 열·새 상태 필드·둘째 런(DR-off 재현런은 §6 user 결정 항목으로만 둠).

## 2. 기전 — 런 0회 확정분

- **DR=0 근인**(판독 §4, 저자 grep 재확인): `dr_alpha_eval = Σ c_i e^{−E_i/T} · T^{−1.5} · dr_boost_factor(source)`(repo `src/lumina_plasma.c` 10144–10176); `dr_boost_factor` 는 `LUMINA_DR_BOOST`(전역, 기본 1.0)와 `LUMINA_DR_BOOST_<SOURCE>`(소스별, 기본 1.0, **env 가 있으면 atof 로 덮어씀**)를 곱한다. base 376251 exports 는 `LUMINA_DR_BOOST_{AUTOSTRUCT,BADNELL,MAZZOTTA,NORAD}="0"` 이고 `LUMINA_FROZENIN_DR="0"` [저자 실행 grep — `run5_h200/input/resolved_lumina.exports` 32–35·42 행]. CONT run1 은 `FROZENIN_DR=1` 인데 부스트 0 이라 덤프 `ne_alpha_dr` 전 행 0 [저자 실행: `cont_nlte_20260918/run1/work/lumina_ion_channels.csv` gen 71, Z=16 셸 12·16·20]. ⟹ 켜는 것은 **부스트 4 키**다(`CMFGEN`·`EST_ISOEL` 소스 키는 exports 에 없어 기본 1.0 — 두 소스도 함께 살아난다).
- **경로**: 이온화 비 r 의 분모 α = `frozenin_alpha_rr`(RR, U_II 정정본) + `fz_dr ? dr_alpha_eval(dr_lookup(Z, stage+1), T) : 0`(repo 6572–6587). 덤프의 `ne_alpha_dr` 는 같은 경로 `frozenin_alpha_dr`(동결 트리 `lumina_plasma.c` 6682; `FROZENIN_DR=0` 이면 0)로 갈라 쓴다 — 직전 Fable 코드 감사가 「동일 경로」로 확인(집행 기록 11).
- **오프라인 예측(기안자 재계산, grammar-debug, 브리프와 일치)**: DR_TABLE `{16,2}` Badnell 7항을 쌍둥이 T_e 에서 평가, α_RR = CONT run1 덤프 `ne_alpha_rr/n_e`(T_e 속성·소스 무관): sh12/16/20 (T_e 10138/9386/8849 K) **α_DR/α_RR = 1.472/1.484/1.495** ⟹ γ_phot·n_e 불변 1-step 근사에서 r_new = r_base/(1+q): 57.0→23.1, 17.7→7.12, 8.79→3.52, **log10(r_base/r_new) = 0.393/0.395/0.397 dex**, f_II .0416/.1231/.2211. 충돌항 c_rec 은 RR 의 10⁻⁹ 급이라 무시 [덤프 실측].
- **전역성(기재)**: 부스트 4 키는 소스별이지 이온별이 아니다 — DR_TABLE 의 모든 항(C·O·Mg·Si·S·Ca·Sc·Ti·Fe I–V·Co III …)이 함께 켜진다. 내핵 IGE 응답은 오프라인 예측을 **하지 않았다**(RR 기준값 없음) — §5 E-3 은 상한만 둔다.
- **바이너리 캐비앗(기재)**: 재사용 바이너리 8ceafa36(`cuda_build4`, CONT 게이트·채널 덤프 코드 포함)은 base 376251 의 4e385989 와 **다르다**. 「CONT 게이트 OFF = base byte-동일」은 selftest OFF-byte 등식 + Fable 코드 감사(BLOCKING 0)의 **오프라인 주장**이지 in-vivo 시연이 아니다. 따라서 이 단의 A/B 는 「DR 토글 단독」을 그 주장 조건부로 말한다(§8). 해소 옵션 = §6 ③.
- **미확정(이 계약 밖)**: 반복 수렴에서 γ_phot 되먹임(S 는 EUV 불투명도 기여가 작아 1-step 근사가 좋을 것으로 보나 측정 대상), Si `{14,2}` DR 의 정량, Ca IV/V DR 부재.

## 3. 수리안 — 변경집합

**코드 변경 0줄 · 재빌드 없음.** 변경은 exports 키뿐(런 구성 = §6). 채널 덤프 `LUMINA_ION_CHANNEL_DUMP=1` 은 기존 게이트(직전 계약 R1-3 구현분, 동결 트리에만 존재·repo 미이식)를 켜는 것이다.

### 변경집합 — DR-ON (표)

| 경로 | 담당 | 내용 |
|---|---|---|
| (없음) | — | 소스 변경 0 파일. 동결 트리 `…/lteunmap_eps_20260917/source` 는 `source.sha256` 대로 불변이어야 한다(G-ID) |

### 변경집합 끝

⚠**프리플라이트 처리(운전자 확인 항목)**: `check_prereg_preflight` PF-1 은 표의 경로 집합과 `symbol` grep 집합의 양방향 차집합을 본다. 이 계약은 표가 비어 있고 `symbol` 을 소스 어디에도 없는 문자열 `DET_DR_ON_CHANGESET_EMPTY` 로 두어 **양쪽 다 공집합**이 되게 했다 [저자 실행: `grep -r` src·tests 0건]. 기대 출력 = `changeset-consistent files=0`. 이는 게이트를 우회한 것이 아니라 「변경 없음」의 정직한 인코딩이며, 실제 「코드 0줄」 보증은 G-ID 의 `source.sha256` 대조가 맡는다. `files=0` 이 아니면 발주 금지.

## 4. 게이트 명부 (표)

| ID | 조건 | 음성대조 |
|---|---|---|
| G-ID | 새 런 exports 를 376251 계보(`/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/run5_h200/input/resolved_lumina.exports`)와 diff — 바뀐 키가 정확히 `LUMINA_FROZENIN_DR`(0→1)·`LUMINA_DR_BOOST_AUTOSTRUCT`·`LUMINA_DR_BOOST_BADNELL`·`LUMINA_DR_BOOST_MAZZOTTA`·`LUMINA_DR_BOOST_NORAD`(각 0→1), 추가 키가 정확히 `LUMINA_ION_CHANNEL_DUMP`(=1) [§6 ① B 선택 시 `LUMINA_PURE_CMFGEN_ITER`(2→10) 추가·프로필 종속 `LUMINA_CMF_FINE_MGPU_DEVICES` 허용]; `LUMINA_CMF_CONT_NLTE_SOURCE`·`LUMINA_ION_NT`·`LUMINA_NT_FULL`·`LUMINA_DR_BOOST`(전역) **부재**; `input/binary.sha256` = 8ceafa36…(`…/lteunmap_eps_20260917/cuda_build4/lumina_cuda`)이고 동결 트리 `…/lteunmap_eps_20260917/source.sha256` 의 `sha256sum -c` non-OK 0(소스 변경 0 파일); 덱 `input/deck.sha256` OK; 러너 편집 = job-name·`--time`(B 선택 시 ITER 검사 116 행·`outer_iterations` 132 행 추가 — 직접 편집, sed `\n` 금지)뿐, 러너 해시 STAGE_IDENTITY 기재 | 다른 키 주입·CONT/NT 키 혼입·소스 파일 1 변조 ⟹ FAIL |
| G-RUN | `RUN_FOOTER.txt` `terminal_state=SUCCESS`·`model_rc=0`·`outer_iterations`=선택 ITER; `work/lumina_ion_pops.csv`·`work/physics_comparison/physics_DET_iter<ITER−1>.ion.csv`·`work/lumina_ion_channels.csv` 실재 | 미완주·덤프 결손 ⟹ FAIL |
| G-DR | `work/lumina_ion_channels.csv` 최대 `gen` 의 Z=16 셸 12·16·20 행에서 (a) `ne_alpha_dr` **> 0** (b) `ne_alpha_dr/ne_alpha_rr` 가 §2 오프라인 1.472/1.484/1.495 의 ±5 % 안(같은 식·같은 핀 T_e 라 **구성상 참 — 배선 확인이지 예측 아님**; 전역 `LUMINA_DR_BOOST` 혼입·다른 소스 키 오설정을 잡는다) (c) `gamma_nt` == 0(NT 미설정 — 교란 차단) | base 376251 = 덤프 없음 ⟹ FAIL; CONT run1 덤프(부스트 0) = `ne_alpha_dr` 0 ⟹ FAIL [저자 실행 확인]; 비 1.0 위조 ⟹ (b) FAIL |

### 게이트 명부 끝

## 5. 기대치 — 런에서만 얻는 수 (잣대를 base 에 먼저 적용: base 는 E-1 을 넘지 못한다 — 이동 0 dex)

| ID | 양 | base(376251) | 예측(§2, 1-step) | **기대** |
|---|---|---|---|---|
| E-1 S 중간 이동 | Δ = log10(r_base / r_run), r = n_III/n_II (`work/lumina_ion_pops.csv`, S 셸 12·16·20; f_II 병기) | r 57.0/17.7/8.79 (f_II .017/.053/.102) | Δ 0.393/0.395/0.397 (f_II .042/.123/.221) | **셋 다 0.20 ≤ Δ ≤ 0.60** |
| E-2 외곽 유지 | S f_III 셸 28·32·36 | .998/.994/.902 | DR 은 외곽에서도 III→II 를 올리나 r ≫ 10² 라 소폭 | **셋 다 ≥ 0.70** |
| E-3 내핵 상한 | Fe f_IV 셸 0·1·2·3 의 \|Δ\| | .982/.997/.997/.979 | 예측 없음(DR 전역성, §2) | **넷 다 \|Δ\| ≤ 0.10** |
| (기재) | Si f_II 셸 12·20 · S f_II 반복별 궤적(`physics_DET_iter000k.ion.csv`, B 선택 시) · Ca II/III | .000/.000 · — | — | 판정 조건 아님, 판정문에 기재 |

문턱 근거: E-1 하한 0.20 = 예측 0.39 의 절반(1-step 이 γ_phot 되먹임으로 줄어도 「DR 이 r 을 움직였다」로 읽을 수 있는 최소, base 0 dex 와 명확히 갈림); 상한 0.60 = 예측 +0.2 dex(율비만으로는 넘을 수 없는 선 — 넘으면 되먹임·다른 채널 개입). E-2 0.70 = 직전 계약과 같은 램프 재점화 감시선. E-3 0.10 = 「부작용 최소」의 기재 가능한 상한(예측 부재라 위반은 실패가 아니라 DR-B 명명 대상).

### 분기 (표)

| 분기 | 조건 | 의미 |
|---|---|---|
| DR-X | G-ID/G-RUN/G-DR 중 하나라도 미충족 | 미시연 |
| DR-A | 게이트 전건 **and** E-1 **and** E-2 **and** E-3 | **예측대로 이동 — 필요·불충분 확정**(CMFGEN 미도달은 성공 조건이 아니라 기재) |
| DR-B | 게이트 전건 **and** (E-1 상한 초과 **or not** E-2 **or not** E-3) | 이동은 있으나 예측 밖(과응답 또는 외곽·내핵 부작용) — 판정자가 채널 덤프·궤적으로 귀속·명명 |
| DR-C | 게이트 전건 **and** E-1 하한 미달 **and** E-2 **and** E-3 | DR 이 켜졌는데 r 이 예측만큼 안 움직임 — 판정자가 명명(γ_phot 되먹임·율 경로 불일치 후보) |

### 분기 끝

재분기 금지. 판정기 = `/gpfs/kjhan/lumina/cmfgen_map/repro_claude_20260917/judge_contnlte.py` 의 G-CHAN 판독부 재사용 + E-1~3 계산(운전자 도구, 해시 `READOUT_TOOLS.sha256` 기재; 소스 아님). 음성대조: base 376251 자기검사 ⟹ G-DR FAIL·E-1 0 dex ⟹ DR-X; CONT run1 덤프 ⟹ G-DR FAIL.

## 6. 자원 · 실행 · **결정 항목(운전자·user)**

새 RUN 루트 `/gpfs/kjhan/lumina/cmfgen_map/dr_on_20260919/run1`(신설, 형제 루트), 입력 = 376251 입력 복사 + 바이너리 8ceafa36 + exports 편집(§4 G-ID). GPU 1 런, h200x1 우선. 봉인 base 는 읽기만.

① **ITER 수 — 기안자가 못 박지 않는다**:
| | A: ITER=2 (run5 와 동일) | B: ITER=8–10 (수렴점) |
|---|---|---|
| 장점 | base 와 **반복수까지 동일**한 A/B(교란 1개 적음); 1-step 예측이 가장 잘 맞는 조건; ≈5 h(376251 실측 18067 s), `--time=08:00:00` | 수렴값을 얻음(다음 단의 base 로 쓸 수 있음); γ_phot 되먹임 크기가 궤적에서 읽힘 |
| 단점 | 수렴값이 아님(다음 단이 다시 돌려야 할 수 있음) | ≈20–25 h·상한 40 GPU-h; base 와 반복수가 달라 E-1 의 이탈이 DR 몫인지 반복 몫인지 섞임(직전 계약 §2 「2회는 구성상 미결」의 반대 방향 교란); 러너 편집 2곳 추가 |
| 계약 적용 | E-1 문턱 그대로 | E-1 문턱 그대로 두되, 판정문이 반복별 궤적으로 이탈을 귀속 |

② 헤지 프로필(h100-a100x2) 사용 여부 — 먼저 시작하는 쪽만.
③ **DR-off 재현런**(바이너리 8ceafa36·DR off·ITER 2, ≈5 h): §2 바이너리 캐비앗을 in-vivo 로 닫는다(base 와 byte-동일 기대). 기본은 **하지 않음**(Q-LEAN) — user 가 「DR 토글 단독」 주장을 무조건부로 원하면 추가.

순서: 경량 사전감사(Opus 4.8 fresh, Q-GOAL·Q-LEAN) → 게이트 4종(prereg-preflight `files=0`·gate_referents·doc_canonical·dispatch-preflight) → 스테이징(운전자) → 제출 → 판독 → 판정(운전자) → 종료감사(Opus 4.8 fresh; 물리 주장 등급이 걸린 단이라 생략하지 않음).

## 7. 분장 장부 (개정25 · 규약 / 실제 / 위반)

| 단계 | 규약 | 실제 | 위반 |
|---|---|---|---|
| 기안 | Fable fresh(25-b) | **Fable fresh**(`claude-fable-5-1`, 이 문서) | — |
| 경량 사전감사(Q-GOAL·Q-LEAN) | Opus 4.8 fresh(25-d) | 미집행 | |
| 구현 | Opus 4.8 | **해당 없음**(코드 0줄) | |
| 코드 간단 감사 | Fable fresh(25-a) | 해당 없음(코드 0줄) — 스테이징 exports 편집은 G-ID 가 대신 검사 | |
| 스테이징·제출·판독·판정 | 운전자 | 미집행 | |
| 종료감사 | Opus 4.8 fresh | 미집행 | |

## 8. 주장 상한

말할 수 있는 것: 「DR 을 켜면 중간 셸 S II/III 비가 오프라인 율비만큼(≈0.4 dex, ×2.4) 움직이는가」와 「그 몫이 CMFGEN 어긋남(4–20× 잔여)의 어느 부분인가(필요·불충분)」. 오프라인 2.4× 는 **1-step 율 추정**이고 수렴값은 런이 정한다. 「DR 토글 단독」 은 §2 바이너리 캐비앗 조건부(§6 ③ 미실행 시). Ca IV/V DR 부재·Si 정량·내핵 IGE DR 의 물리적 옳음(Badnell/Mazzotta 계수 자체)·스펙트럼·자유-T_e 는 말하지 않는다. twin-T_e 핀 조건부.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DR-ON (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)",
    "symbol": "DET_DR_ON_CHANGESET_EMPTY",
    "roots": ["src", "tests"],
    "expected_extra": []
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "clean": "sum(1 for x in v if 0.5 <= x < 1.5)", "under": "sum(1 for x in v if 1.5 <= x < 2.5)", "over_or_side": "sum(1 for x in v if x >= 2.5)"},
    "rules": [
      {"name": "DR-X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "DR-A", "predicate": "count > 0 and gate_fail == 0 and under == 0 and over_or_side == 0"},
      {"name": "DR-B", "predicate": "count > 0 and gate_fail == 0 and over_or_side > 0"},
      {"name": "DR-C", "predicate": "count > 0 and gate_fail == 0 and under > 0 and over_or_side == 0"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "gate-fail", "mix": [[0.0, 1], [1.0, 2]]},
      {"name": "clean", "mix": [[1.0, 3]]},
      {"name": "under", "mix": [[1.0, 2], [2.0, 1]]},
      {"name": "over", "mix": [[1.0, 2], [3.0, 1]]},
      {"name": "under-and-over", "mix": [[2.0, 1], [3.0, 1]]}
    ],
    "residual": "DR-X"
  },
  "references": [
    {"path": "docs/BRIEF_DR_ON_2026-09-19.md"},
    {"path": "docs/READOUT_CONT_NLTE_SUPERTHERMAL_2026-09-18.md"},
    {"path": "docs/PREREG_DET_CONT_NLTE_BALANCE_2026-09-18.md"},
    {"path": "CLAUDE.md"},
    {"path": "src/lumina_plasma.c", "flags_existing": ["dr_boost_factor", "dr_alpha_eval", "dr_lookup", "DR_TABLE", "LUMINA_FROZENIN_DR", "frozenin_alpha_rr"]},
    {"path": "src/env_universe.h", "flags_existing": ["LUMINA_DR_BOOST_AUTOSTRUCT", "LUMINA_DR_BOOST_BADNELL", "LUMINA_DR_BOOST_MAZZOTTA", "LUMINA_DR_BOOST_NORAD", "LUMINA_FROZENIN_DR"]}
  ]
}
```

## 9. 집행 기록

1. (운전자 기입) 경량 사전감사 결과 · 게이트 4종 rc · `[미실행]` 주장 실행 결과 · ITER/헤지/재현런 결정(§6 ①②③) · 스테이징 해시 · job id.
