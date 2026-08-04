# Clamp 수정 우선순위 정본 레지스트리

## 운용 규약

본 레지스트리는 단계별 검사(Gate B oracle·KA 사다리·캡처 런·relT 앵커) 완료 시 물리검증 결과와 조인되어 **최순위 코드 수정 리스트**로 승격된다(user 지시 2026-07-31). 개별 선행 패치 금지.

- 정본 원천: `docs/CODEX_CLAMP_CENSUS_2026-07-31.md`
- 정렬 규칙: 위험 45건을 표 상단에 배치하고, 그 안에서는 대체형 23건을 우선 배치한다. 이후 비위험 49건은 원본 ID 순서를 유지한다.
- `Ctr`: Y=발화 카운터 존재, P=부분·제한 카운터, N=없음.
- join 값은 아래 코드북을 따른다. Gate B는 동일 행의 1%급 전후효과만 조건부 적격이며, 절대 CMFGEN 진리 판정은 수렴 released-T 앵커가 착지해야 확정한다.
- `intro_reason`은 도입 사유 고고학 분류(A/B/C/D), `upstream_bug`는 A형이 차단한 상위 결함을 기록한다. A형이 아니면 `—`로 둔다.

## 전수 레지스트리

| ID | file:line | 물리량 | 유형 | 게이트 | 3문 분류 | 카운터 | 88대장 대응 | oracle_verdict(Gate B 대조 결과) | band_link(비보존 대역 연루) | fix_stage(동등화 Stage 매핑) | priority(최종 순위) | intro_reason | upstream_bug |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C08 | `lumina_cmfgen.c:177-234` | `eps_floor=1e-5`, 미등록 ε→1 | 대체 | `LINE_EPS_PHYS` | 위험 | N | 기존 C4 | GB-bb 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 1 | P036 | (A) | 원자 충돌률·ε 메타데이터 부재 |
| C08b | `lumina_plasma.c:8472` | `denom<=0 → return 1.0` (정의되지 않은 `0/(0+0)`에 ε=1 대입) | 대체 | `LINE_EPS_PHYS` | 위험 | **미측정** | **신규(08-03)** | C08과 **다른 분기** — C08은 미등록 line→`-1`→caller가 1. 이쪽은 `C_ul=A_ul=0`인 정의불가 케이스에 완전열화를 대입 | I: ε는 `chi_line_th`·`eta_line`·`S_fixed` 전부를 통과 | Stage 1 | — | (A) | **사유별 카운터 부재** — post-clamp `el`만 저장돼 "원래 1"과 "발화 결과 1"을 구별 불가. rung1 v4의 `eps_applied_diff`가 최초 계기이나 E8 epoch 소급 불가 |
| C09 | `lumina_cmfgen.c:227,1277-84,2359,2398`; `lumina_plasma.c:10944,17143-91` | 미해결 선원함수→`B`/`WB`/0 | 대체 | 경로별 | 위험 | P | 기존 F1/C1/C2, FORMAL_FIX에서 변경 | GB-bb 일부(1% 전후); relT 착지 후 확정 | D: formal 직접(×18.07; 2500–5000 Å 72%) | FORMAL_FIX 부분; Wave3→Stage 2·4·6 | P001 | (A) | operator-split 선원함수 폭주·NLTE-network-out source 미기록 |
| C10 | `lumina_cmfgen.c:269-518` | EPAY 열방출 재척도·τ 게이트 | 대체 | `LUMINA_CMF_EPAY*` | 위험 | P | 변경 C7, `TAUEFF` 추가 | GB-thermal 일부(1% 전후); relT 착지 후 확정 | D: formal 직접(×18.07; 2500–5000 Å 72%) | Stage 6 | P020 | (C) | — |
| C13 | `lumina_cuda.cu:1543-52`; `lumina_plasma.c:16045-68` | 음수 NLTE 인구→`1e-30` | 대체 | 기본 경로 | 위험 | N | 기존 A1 | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 2→4 | P027 | (A) | 냉각·근특이 NLTE 해의 음수 인구 |
| C14 | `lumina_cuda.cu:1473-1543`; `lumina_plasma.c:15996-16068` | LTE-relative repair/floor | 대체 | `LTE_FLOOR/LTE_REPAIR` | 위험 | N | 변경 A3 | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 2→4 | P028 | (A) | 냉각·근특이 NLTE 행렬의 ill-conditioning |
| C22 | `lumina_cuda.cu:1807`; `lumina_plasma.c:16318` | `S_l` 분모 컷→0→소비자 fallback | 대체 | 없음 | 위험 | N | 기존 A14 | GB-bb 일부(1% 전후); relT 착지 후 확정 | D: formal 직접(×18.07; 2500–5000 Å 72%) | Wave3→Stage 2·4·6 | P002 | (A) | NLTE-network-out source=0 센티널 |
| C23 | `lumina_bf_gemm.cu:83-93`; `lumina_plasma.c:1830-53,4116-18,5787,8197-99` | bf 하준위 인구·분배함수 | 대체 | 대부분 없음 | 위험 | N | 기존 A18/A19/A37/H24 | GB-bf 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 1 | P059 | (C) | — |
| C24 | `lumina_nlte_gemm.cu:186-90,379`; `lumina_atomic.c:961-65`; `lumina_plasma.c:12694,14922-32` | 미등록 σ_bf→Kramers | 대체 | 데이터 의존 | 위험 | Y | 기존 A16/L6 | GB-bf 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 1 | P037 | (A) | σ_bf 데이터·평가기 미커버 |
| C38 | `lumina_plasma.c:2305,2356,2453,2662-63,6166,6206-21,8331,11897,11916` | n_ion·n_e 비정상→양의 상수 | 대체 | 경로별 | 위험 | N | 변경/확장 A38 | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 2 | P061 | (C) | — |
| C40 | `lumina_plasma.c:9950-78,11258-81,11306-39,13285` | Te bracket/HOLD/500·1000 K floor | 대체 | solver별 | 위험 | Y | 변경 A30 | GB-thermal 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 4 | P003 | (A) | 열평형 no-root·endpoint 자기조명 feedback |
| C46 | `lumina_plasma.c:1246-90,1520` | W>1e4/비유한→TR refit 또는 장 0 | 대체 | radiation-field 경로 | 위험 | N | 기존 A43 | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 3 | P047 | (A) | C1 장 적합 rail·기하 bin 불일치(G-3) |
| C47 | `lumina_atomic.c:364-77`; `lumina_plasma.c:869,1240-90` | T_rad 고정·TEPIN·W cap | 대체 | `TRAD_COLOR_FIX` 등 | 위험 | P | 기존 A44/L5/H8 | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Wave3; Stage 3 | P054 | (B) | — |
| C49 | `lumina_atomic.c:435-55,961-65,1058,1122,1222` | 로더 불일치→0/Kramers/ground/Axelrod | 대체 | 데이터 의존 | 위험 | P | 기존 L1/L2/L6/L7 | GB-bf 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 0→1 | P042 | (A) | 원자 로더·스키마·데이터 결손 |
| C59 | `lumina_plasma.c:4896-5055,6760-6925`; `lumina_cuda.cu:4781-4910,5599-5805` | k-packet fb Kramers 확률·대표 에지 | 대체 | `KPKT_FB_MULTI` | 위험 | Y | 신규 H1; Z에서 edge-failure counter 추가 | W1/W2 0변화; packet·GPU oracle 밖 | E: packet-energy 연루; 대역 귀속 계측 필요 | Wave1 gate→제거 가능; Stage 5 event KA | P013 | (B) | — |
| C64 | `lumina_plasma.c:15594-694` | bb-isolated·top-stage 행을 Boltzmann anchor로 교체 | 대체 | `FLOOR_REG`, `TOPSTAGE_THERMALIZE` | 위험 | P | 신규 | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 2 | P034 | (A) | D-5 상위-stage-blind 행렬·고립행 |
| C66 | `lumina_cuda.cu:9370-82` | `S/B>100`이면 pops rollback·J̄ 영구차단 | 대체 | THEN-MC/JBAR pops | 위험 | Y | 신규 H6 | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 3→4 | P005 | (A) | operator-split J̄→pops 폭주(run 165510) |
| C68 | `lumina_plasma.c:16942-45,17195-97` | thick-line `S_l→B(Te)` | 대체 | `FI_CLAMP_SL` | 위험 | N | 신규 | GB 범위 밖 | D: formal 직접(×18.07; 2500–5000 Å 72%) | Wave3 격리; Stage 6 | P007 | (B) | — |
| C69 | `lumina_plasma.c:16957-59,17206-12` | IGE forest opacity 제거 | 대체 | `FI_FOREST_NOBLANK` | 위험 | P | 신규 | GB 범위 밖 | D: formal 직접(×18.07; 2500–5000 Å 72%) | Wave3 격리; Stage 6 | P008 | (B) | — |
| C70 | `lumina_plasma.c:16234-41,16321-30` | Fe 창내 `S_l *= X` | 대체 | `FLUOR_ORACLE_X` | 위험 | Y | 신규/oracle falsifier | GB 범위 밖 | D: formal 직접(×18.07; 2500–5000 Å 72%) | Wave3 격리; Stage 6 | P009 | (B) | — |
| C71 | `lumina_cuda.cu:6539-61`; `scripts/run_coevolve_s01.sh:54` | line re-emission→`B(Te)` | 대체 | `LINE_THERM` | 위험 | N | 변경 H3: Z에서 배너만 수정 | W2 0변화; CUDA 방출 oracle 밖 | E: packet-energy 연루; 대역 귀속 계측 필요 | Wave2 gate→제거 가능; Stage 5 KA | P017 | (B) | — |
| SC13 | `expand_atomic_data_cmfgen.py:406-25,739-775,1004-54,1128`; `build_cmfgen_coldata_all.py:455-95` | 데이터 부재→cap/Kramers/skip | 대체 | 데이터 의존 | 위험 | P | 신규 | GB 범위 밖 | I: 상태·율 간접; relT 착지 후 확정 | Stage 1 | P043 | (A) | 원자 fit·data evaluator 결손 |
| SC16 | `offline_fluor_field_test.py:64,69,72,82,92,96` | gbar=0.2, f≥1e-6, pops≥0 | 대체 | prototype | 위험 | N | 신규 | GB 범위 밖 | I: 상태·율 간접; relT 착지 후 확정 | Wave3; Stage 1 prototype | P058 | (B) | — |
| SC21 | `build_cmfgen_coldata_all.py:18,455,486,495`; `expand_atomic_data_cmfgen.py:425`; `bake_coiii_real_sigma.py:78` | cap 위 전이·비양수 Ω 데이터 제거 | 대체 | 데이터 생성 | 위험 | P | 신규 | GB 범위 밖 | I: 상태·율 간접; relT 착지 후 확정 | Stage 0→1 | P044 | (A) | SC12 준위 절단·원자 데이터 불일치 |
| C15 | `lumina_cuda.cu:987-1002,1510-24`; `lumina_plasma.c:16000-41` | FLOORM LTE floor | floor | `FLOOR_MODE=1` | 위험 | N | 기존 A4 | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 2→4 | P029 | (A) | 냉각·근특이 NLTE 행렬의 ill-conditioning |
| C16 | `lumina_cuda.cu:1460-70,1560-75` | `b_k` ceiling | cap | `BK_CEIL` | 위험 | N | 기존 A5 | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 2→4 | P030 | (A) | NLTE 행렬 ill-conditioning의 초열적 b_k 쓰레기 |
| C27 | `lumina_nlte_assemble.cu:123`; `lumina_plasma.c:13677,13775`; `lumina_cmfgen.c:1089,1303` | 빈/창밖 J→`1e-30`, 0, fallback | floor | 경로별 | 위험 | N | 기존 A24/A25/P9/L9 | GB-bb 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 3 | P046 | (A) | 주파수창·estimator 미커버(G-3) |
| C28 | `lumina_plasma.c:3441-63` | J→factor·`WB` 상하한 | cap | `J_CAP/​FLOOR_FACTOR` | 위험 | N | 신규(H9는 과거 88 이후) | W2 0변화; MA 확률 oracle 밖 | E: packet-energy 연루; 대역 귀속 계측 필요 | Wave2 gate→제거 가능; Stage 5 KA | P015 | (B) | — |
| C29 | `lumina_plasma.c:13706-68` | UV Jν→`W_cap Bν` | cap | `J_NU_UV_CAP` | 위험 | Y | 신규(H9 이후) | GB-bf 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Wave3; Stage 3 | P016 | (B) | — |
| C32 | `lumina_plasma.c:7871-95,8048-68` | Υ 하한 | floor | `RADEQ_OMEGA_FLOOR` | 위험 | N | **변경 A29**: parity 기본 off, CMFGEN tier와 상호배타 | GB-coll strict 0; 부적격 | I: 상태·율 간접; relT 착지 후 확정 | Wave3; Stage 1 | P052 | (B) | — |
| C34 | `lumina_plasma.c:350-480,8061-68,14420-14717`; `lumina_nlte_assemble.cu:209-10` | gbar/Axelrod/forbidden Ω | floor | parity·mode | 위험 | P | 기존 A20/A26/A29/L7 | GB-coll strict 0; 부적격 | I: 상태·율 간접; relT 착지 후 확정 | Stage 1 | P040 | (A) | Ω·collision-strength 데이터 미커버 |
| C36 | `lumina_plasma.c:15446-64` | 전 이온쌍 α_DR 하한 | floor | `DR_FLOOR_CMS` | 위험 | N | 신규 | GB-bf 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 1 | P050 | (A) | K6·저온 DR resonance 데이터 미커버 |
| C37 | `lumina_plasma.c:2134-38,2285-95,2438-44,8389-99,11461-64,11844-45,15089` | 이온비·연쇄곱 `1e28/1e30` | cap | 경로별 | 위험 | N | 기존 A27/A38 | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 2 | P060 | (C) | — |
| C44 | `lumina_plasma.c:10153-62,10691-716` | 음수 선냉각→0 | floor | `COOL_NONNEG` | 위험 | N | 신규 | GB-thermal 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 4 | P004 | (A) | lagged/non-SE 인구의 가짜 역전 |
| C45 | `lumina_plasma.c:10261-77,10474-76` | 상준위≤LTE, η_lag≥0 | cap | line-response 경로 | 위험 | N | 신규 | GB-bb 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Wave3; Stage 4 | P053 | (B) | — |
| C48 | `lumina_atomic.c:723-39` | `SUPER_CUTOFF` 이상 준위 lump·LTE 내부분배 | cap | `SUPER_CUTOFF` | 위험 | Y | 신규(H2 이후) | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 1→2 | P033 | (A) | full-level 행렬의 250-order conditioning |
| C52 | `lumina_cuda.cu:2724-64,5342,5856-90,6297-321` | packet interaction 절단·에너지 drop/force escape | cap | `MAX_INTERACTIONS` | 위험 | Y | 기존 M1 | GB 범위 밖 | E: packet-energy 연루; 대역 귀속 계측 필요 | Stage 5 | P012 | (A) | interaction 계수 오배선·cap-hit packet 에너지 삭제(비보존) |
| C53 | `lumina_cuda.cu:4226,4491,6331-36`; `lumina_transport.c:419-49` | MA 내부 cascade | cap | `MA_INTERNAL_CAP` | 위험 | Y | 기존 M2 | GB 범위 밖 | E: packet-energy 연루; 대역 귀속 계측 필요 | Stage 5 계측 | P018 | (D) | — |
| C65 | `lumina_plasma.c:9415-24,9487-90` | stage-IV 하준위 `b_k≤1000` | cap | `STAGE4_BK_CAP` | 위험 | N | 신규 | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 2 | P035 | (A) | stage-IV metastable continuum drain 부재 |
| SC04 | `build_ddc15_epoch.py:74`; `build_ddc15_initial_epoch.py:131`; `build_ddc15_real_composition.py:144` | isotope mass fraction→비음수 | floor | 없음 | 위험 | N | 신규 | GB 범위 밖 | I: 상태·율 간접; relT 착지 후 확정 | 유지; Stage 0 simplex KA | P088 | (C) | — |
| SC05 | 아래 별도 전개한 86개 `slurm_*.sh` | 외곽 Fe 질량분율 `X_Fe≥5e-4` | floor | launcher 분기 | 위험 | N | 신규 | GB 범위 밖 | I: 상태·율 간접; relT 착지 후 확정 | Wave3; Stage 0 입력 정리 | P055 | (B) | — |
| SC06 | `analyze_ddc15_F1_oskip.py:61`; `G1:60`; `H1:60`; `H1b:65`; `H1p:64`; `H2:66`; `I1/J1/K1/K1b:62`; `phase_D_si_red_validation.py:70` | continuum normalization≥peak 1% | floor | 없음 | 위험 | N | 신규 | GB 범위 밖 | M: 판정 metric·비교자 연루 | 계측 | P023 | (C) | — |
| SC08 | `empirical_pcygni_ml.py:122,151` | kernel width≥0.01, τ≤8 | cap | 없음 | 위험 | N | 신규 | GB 범위 밖 | N: 직접 연루 근거 없음 | 계측 | P090 | (C) | — |
| SC11 | `score_nw.py:133`; `check_mode_equivalence.py:90` | χ·검증 tolerance | cap | CLI/MC | 위험 | N | 신규 | GB 범위 밖 | M: 판정 metric·비교자 연루 | Stage 0 acceptance 계측 | P024 | (C) | — |
| SC12 | `expand_atomic_data_cmfgen.py:62-155,425`; `bake_coiii_real_sigma.py:78` | 원자 준위 수 | cap | config | 위험 | N | 신규 | GB 범위 밖 | I: 상태·율 간접; relT 착지 후 확정 | Wave3; Stage 0→1 projection | P056 | (B) | — |
| SC15 | `offline_cell_balance.py:220,226-28` | ETLA 상준위≤LTE | cap | prototype | 위험 | N | 신규 | GB 범위 밖 | I: 상태·율 간접; relT 착지 후 확정 | Wave3; Stage 1 prototype | P057 | (B) | — |
| C01 | `lumina_main.c:58-61`; `lumina_cuda.cu:3446-49,4551-65,4782,4872-4910,5196-99,5599,5654,5677,5805`; `lumina_plasma.c:7015-18` | RNG `log(0)` | floor | 없음 | 정당 | N | 기존 A48/H21 | GB 범위 밖 | E: packet-energy 연루; 대역 귀속 계측 필요 | 유지; Stage 5 KA | P072 | (C) | — |
| C02 | `lumina_cmfgen.c:37-41`; `lumina_nlte_assemble.cu:115`; `lumina_plasma.c:1025,1037,2687,4116,5787,8199,8265,8427,8431,16845-51` | Planck/지수 underflow | cap | 없음 | 정당 | N | 기존 A18/A37/C12/F4 | GB 범위 밖 | N: 직접 연루 근거 없음 | 유지; Stage 1·6 KA | P073 | (C) | — |
| C03 | `lumina_plasma.c:2680-81,10171-72`; `lumina_cmfgen.c:121,132,150,161` | escape probability 점근식 | 대체 | 없음 | 정당 | N | 기존 A39/H22 | GB 범위 밖 | N: 직접 연루 근거 없음 | 유지; Stage 6 KA | P074 | (C) | — |
| C04 | `lumina_plasma.c:488-495,7921-22,14767-74,14858-59`; `lumina_radeq_col_pairs.h:67-70` | Ω(T), ζ 등 표 범위 밖 끝값 | 대체 | 경로별 | 조건부 | N | 기존 H23/H25 | GB 범위 밖 | N: 직접 연루 근거 없음 | 유지; Stage 1 KA | P075 | (C) | — |
| C05 | `lumina_cuda.cu:5047,5091,5127`; `lumina_plasma.c:16878` | 확률·sqrt 정의역 | floor | 없음 | 정당 | N | 기존 H20/H25 | GB 범위 밖 | E: packet-energy 연루; 대역 귀속 계측 필요 | 유지; Stage 5 KA | P076 | (C) | — |
| C06 | `lumina_cmfgen.c:368,377,608,633,923,1038,1046,1135-37,1220-39,1872,1880` | χ, η, Δτ 비음수화 | floor | 없음 | 조건부 | N | 기존 C9/M6 | GB-bf 일부(1% 전후); relT 착지 후 확정 | D: formal 직접(×18.07; 2500–5000 Å 72%) | Stage 6 | P019 | (C) | — |
| C07 | `lumina_cmfgen.c:1950,2145-46`; `lumina_cuda.cu:6383,6402,8425` | ε·확률 `[0,1]` | cap | env | 정당 | N | 기존 C3/C4 | GB 범위 밖 | E: packet-energy 연루; 대역 귀속 계측 필요 | 유지; Stage 5 KA | P077 | (C) | — |
| C11 | `lumina_cmfgen.c:739-797`; `lumina_cmf_solve.cu:220` | ALI 분모·음수/NaN J | floor | ALI 경로 | 조건부 | N | 기존 C8/C9/P5 | GB-bb 일부(1% 전후); relT 착지 후 확정 | N: 직접 연루 근거 없음 | Stage 3 | P078 | (C) | — |
| C12 | `lumina_cmfgen.c:1121,1197`; `lumina.h:144-151` | 적분 substep·고정 작업공간 | cap | 없음/env | 조건부 | N | 기존 P6/P7 | GB 범위 밖 | N: 직접 연루 근거 없음 | Stage 3·6 계측 | P079 | (C) | — |
| C17 | `lumina_cuda.cu:1270-1323`; `lumina_plasma.c:15913-16134` | INV/grey/residual 실패→Boltzmann | 대체 | 복수 env | 조건부 | P | 기존 A2/A6/H7 | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 2→4 | P031 | (A) | NLTE LU·grey·residual 실패 |
| C18 | `lumina_cuda.cu:1116`; `lumina_plasma.c:15920` | 희박 ion pair skip→fallback | 대체 | `SKIP_DEAD` | 조건부 | N | 기존 A7 | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 2 계측 | P080 | (C) | — |
| C19 | `lumina_cuda.cu:1424-35`; `lumina_plasma.c:15776-84` | BK_PARTIAL 참조 인구 | floor | `BK_PARTIAL` | 조건부 | N | 기존 A10 | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 2→4 | P032 | (A) | cold-Te rate-matrix ill-conditioning |
| C20 | `lumina_cuda.cu:1798`; `lumina_plasma.c:2524-2606,16307`; `lumina_cuda.cu:9541` | Sobolev τ zero sentinel | floor | 없음 | 조건부 | P | 기존 A12/M6 | GB-bb 일부(1% 전후); relT 착지 후 확정 | D: formal 직접(×18.07; 2500–5000 Å 72%) | Stage 1·6 | P021 | (C) | — |
| C21 | `lumina_cuda.cu:1794`; `lumina_plasma.c:2600,4400,16302` | inversion/maser 흡수율→0 | floor | 경로별 | 조건부 | N | 기존 A13/H12 | GB-bb 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Wave3; Stage 1·3·6 | P051 | (B) | — |
| C25 | `lumina_nlte_gemm.cu:182` | χ 부재→`1e10 eV` | 대체 | 없음 | 조건부 | N | 기존 A17 | GB-bf 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 1 | P038 | (A) | ionization-energy 데이터 부재 |
| C26 | `lumina_plasma.c:3249-53,3688-3713,14209`; `lumina_cuda.cu:8499,8562` | MC J̄ crossing 미달→binned J | 대체 | `JBAR_MIN` | 조건부 | Y | 기존 A21/M9 | GB-bb 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 3 | P045 | (A) | MC crossing undersampling |
| C30 | `lumina_plasma.c:8914,10028,10679-85,13296`; `lumina_cuda.cu:7153,8104-39` | Te/ion/J 반복 감쇠 | damping | 복수 env | 조건부 | N | 기존 A32 | GB-thermal 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 4 | P064 | (C) | — |
| C31 | `lumina_cuda.cu:816-49,8108-39,8169-81` | AA J̄/Jblue raw·EMA 통일 | damping | `JBAR_DAMP_UNIFY=1/2` | 조건부 | N | **신규 AA** | GB-bb 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 3→4 | P065 | (C) | — |
| C33 | `lumina_plasma.c:502-680,7878-8080` | 미등록 Ω→vR/`OMEGA_SET` | 대체 | `OMEGA_CMFGEN` | 조건부 | Y | 신규 | GB-coll strict 0; 부적격 | I: 상태·율 간접; relT 착지 후 확정 | Stage 1 | P039 | (A) | Ω 데이터 미등록 |
| C35 | `lumina_plasma.c:14094-108,14456-65`; `lumina_nlte_assemble.cu:419-21` | `C_down≥εA`, DB로 C_up 재생성 | floor | `NLTE_COLL_FLOOR` | 조건부 | N | 신규 | GB-coll strict 0; 부적격 | I: 상태·율 간접; relT 착지 후 확정 | Stage 1→2 | P041 | (A) | 충돌자료 결손·rate-matrix conditioning |
| C39 | `lumina_plasma.c:8219-31,9989,11344,13300` | Te `[0.5,2]×Told` | damping | `TE_STEP_CLAMP` | 조건부 | N | 기존 A31 | GB-thermal 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 4 | P066 | (C) | — |
| C41 | `lumina_plasma.c:9787-88,11148-59,12787-98` | line cooling contribution cull | cap | `RADEQ_LINE_CULL` | 조건부 | N | 기존 A36 | GB-thermal 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 4 | P062 | (C) | — |
| C42 | `lumina_plasma.c:10193-201` | H-response trust region | damping | `HRESP_CLAMP` | 조건부 | N | 기존 A35 | GB-thermal 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 4 | P067 | (C) | — |
| C43 | `lumina_plasma.c:12068-74,12301-11,13009,13254-96` | S42 증폭·Newton 15% step·line search | damping | coupled solver | 조건부 | P | 신규(H11 이후) | GB-thermal 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 4 | P063 | (C) | — |
| C50 | `lumina_atomic.c:902-22` | multiplicity `[0,127]` 표현범위 | cap | spin table | 정당 | N | 신규(H25 이후) | GB 범위 밖 | N: 직접 연루 근거 없음 | 유지; Stage 0 projection | P081 | (C) | — |
| C51 | `lumina.h:144-51,471-84,1252`; `lumina_atomic.c:1142,1238-43`; `lumina_plasma.c:8732` | shell/col-ion/collpair/network 정적 크기 | cap | 컴파일·loader | 조건부 | P | 신규 H14-H19/H25 | GB 범위 밖 | N: 직접 연루 근거 없음 | Stage 0 projection·계측 | P082 | (C) | — |
| C54 | `lumina_cuda.cu:2755,5342,6321-26`; `lumina_transport.c:523` | total step·CPU loop | cap | env/경로 | 조건부 | N | 기존 M3/M5 | GB 범위 밖 | E: packet-energy 연루; 대역 귀속 계측 필요 | Stage 5 계측 | P068 | (D) | — |
| C55 | `lumina.h:144-51`; `lumina_cuda.cu:4025,7035-53` | census/event-log 저장량 | cap | 계기 gate | 정당 | Y | **변경 M11**: 기본 400M→32M records | GB 범위 밖 | N: 직접 연루 근거 없음 | 유지; Stage 0·5 계측 | P083 | (C) | — |
| C56 | `lumina_cuda.cu:5158,7577-84,7625,8954`; `lumina_main.c:813-28` | vpacket τ, injection τ, SED·광선 수 | cap | 복수 env | 조건부 | P | 기존 F3/F5/F6/M7/M8/H17/H20 | GB 범위 밖 | E: packet-energy 연루; 대역 귀속 계측 필요 | Stage 5 계측 | P069 | (C) | — |
| C57 | `lumina_cuda.cu:3486-3504`; `lumina_main.c:41-67` | Planck sampler 반복 실패→대역 uniform | 대체 | 경로 | 조건부 | N | 신규 H10/H21 | GB 범위 밖 | E: packet-energy 연루; 대역 귀속 계측 필요 | Stage 5 | P070 | (C) | — |
| C58 | `lumina_plasma.c:6450-68`; `lumina_cuda.cu:3358-61` | bf 격자 밖 0·마지막 빈 유지 | 대체 | 없음 | 조건부 | N | 신규 H4 | GB 범위 밖 | E: packet-energy 연루; 대역 귀속 계측 필요 | Stage 1·5 계측 | P071 | (D) | — |
| C60 | `lumina_plasma.c:14945-54,15014-54` | C1 GEMM→C2 estimator/빈별 fallback | 대체 | `C2_MATRIX_BF` | 조건부 | Y | **신규 Y3** | GB-bf 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 3 | P048 | (A) | C2 estimator 빈칸·C1 provenance |
| C61 | `lumina_plasma.c:2803-936,14962,15092-108,15311-73` | spin-forbidden 재결합율→0 | 대체 | `REC_SPINGATE` | 조건부 | P | **신규 Y4** | W1 η −4.31…−65.25%; CMFGEN η 앵커 없음; relT 착지 후 확정 | E: packet-energy 연루; 대역 귀속 계측 필요 | Wave1 부분수리; Wave3→Stage 1 | P014 | (B) | — |
| C62 | `lumina_plasma.c:3688-3713` | MA J̄ 문턱 10→`JBAR_MIN` | 대체 | `JBAR_UNIFY` | 조건부 | N | **신규 Y6** | GB 범위 밖 | I: 상태·율 간접; relT 착지 후 확정 | Stage 3 | P084 | (C) | — |
| C63 | `lumina_plasma.c:371-405,2018-2453,12694` | rate-SE field 선택·0/0 prior·ratio caps | 대체 | `RATES_FIX` | 조건부 | P | 신규 | GB-state 일부(1% 전후); relT 착지 후 확정 | I: 상태·율 간접; relT 착지 후 확정 | Stage 3 | P049 | (A) | rate-field·upper-ion routing 결손 |
| C67 | `lumina_plasma.c:16914-40,17028-91`; `lumina.h:1154-86` | formal ray·continuum·τ/S provenance 경로 교체 | 대체 | `FORMAL_FIX` | 조건부 | Y | **변경 F1/F2** | GB 범위 밖 | D: formal 직접(×18.07; 2500–5000 Å 72%) | 기수리 FORMAL_FIX; Stage 6 계측 | P006 | (C) | — |
| C72 | `lumina_main.c:813-21`; `lumina_cuda.cu:9778-84`; `lumina.h:1178-80` | formal impact-ray 해상도 | cap | `CMF_NIMPACT` | 조건부 | N | **변경 F5**: 고정 100→env, 기본 50 | GB 범위 밖 | D: formal 직접(×18.07; 2500–5000 Å 72%) | Stage 6 계측 | P010 | (D) | — |
| C73 | `lumina_cmf_selftest.c:307-51,512,955,1077,1440`; `cmf_pcygni_b1.c:199` | selftest 잔차·분모 정의역 | floor | test 전용 | 정당 | N | 신규/검증 전용 | GB 범위 밖 | N: 직접 연루 근거 없음 | 유지; Stage 0 계측 | P085 | (C) | — |
| SC01 | `analyze_jnu_sed.py:20`; `euv_planck_check_s8.py:44`; `formal_integral_obsframe.py:21,87`; `frozen_in_milne_prototype.py:98`; `offline_bk_per_shell.py:23`; `patch_transprob_aul_weighted.py:57`; `expand_atomic_data_cmfgen.py:643`; `offline_macroatom_calc.py:25`; `lte_inversion_F1.py:108`; `cascade_walk_fe2.py:63`; `cascade_multicycle.py:52` | Planck/exp 정의역 | cap | 없음 | 정당 | N | 신규(script) | GB 범위 밖 | N: 직접 연루 근거 없음 | 유지; Stage 0 KA | P086 | (C) | — |
| SC02 | `score_blondin_fscl_sn2002bo.py:76,119,145`; `diag_sl_vs_jline.py:37,58,62`; `compare_narrowband.py:76-77`; `analyze_nlte_matrix_svd.py:35-37,73`; `finalize_cmfgen_ref_npy.py:110`; 다수 plot/ratio | 로그·비율 분모 | floor | 없음 | 조건부 | N | 신규 | GB 범위 밖 | M: 판정 metric·비교자 연루 | Stage 0 metric 계측 | P026 | (C) | — |
| SC03 | `plot_hst_pcygni_map.py:53,58`; `single_fe2_line_pcygni.py:59,66`; `compare_ne_vs_cmfgen.py:92`; `build_toy06_epoch.py:194` | sqrt·기하학 범위 | floor | 없음 | 정당 | N | 신규 | GB 범위 밖 | N: 직접 연루 근거 없음 | 유지; 계측 | P087 | (C) | — |
| SC07 | `frozen_in_milne_prototype.py:215`; `frozen_in_multistage_prototype.py:141`; `frozen_in_ode_test.py:104` | ODE ion fraction `[0,1]` | cap | prototype | 조건부 | N | 신규 | GB 범위 밖 | N: 직접 연루 근거 없음 | Stage 7T prototype 계측 | P089 | (C) | — |
| SC09 | `formal_integral_obsframe.py:38,50,52,87` | scattering fraction·ν endpoint·τ | 대체 | 없음 | 조건부 | N | 신규 | GB 범위 밖 | D: formal 직접(×18.07; 2500–5000 Å 72%) | Stage 6 진단 | P022 | (C) | — |
| SC10 | `g2_inverse_regression.py:74,168,170,185,203,216`; `g1_jacobian_sensitivity.py:86-87` | 회귀 파라미터 feasible bounds | cap | 명시 범위 | 조건부 | N | 신규 | GB 범위 밖 | N: 직접 연루 근거 없음 | Stage 0 acceptance 계측 | P091 | (C) | — |
| SC14 | `build_dr_cob3.py:196`; `parse_adasdr_adf09.py:85,107,134,137` | DR fit coefficient·weight 양수화 | floor | fit 경로 | 조건부 | N | 신규 | GB 범위 밖 | N: 직접 연루 근거 없음 | Stage 1 data 계측 | P092 | (C) | — |
| SC17 | `validate_plasma.py:130-248`; `debug_neutral_tau.py:128,244,263`; `per_ion_tau_attr_si2_6355.py:107-54` | production floor의 검증 미러 | 대체 | 검증 전용 | 조건부 | N | 신규 | GB 범위 밖 | N: 직접 연루 근거 없음 | 계측 | P093 | (C) | — |
| SC18 | `oracle_compare_cmfgen.py:348-51,408,446` | CMFGEN depth/frequency nearest-neighbor | 대체 | report-only | 조건부 | N | **신규 AB/oracle** | 판정기 자체; Stage 0 재검증; relT 착지 후 확정 | M: 판정 metric·비교자 연루 | Stage 0 comparator 계측 | P011 | (D) | — |
| SC19 | `check_mode_equivalence.py:90`; `mode_convergence_telemetry.py:147` | MC noise tolerance·ratio denom | floor | 진단 | 조건부 | N | 신규 | GB 범위 밖 | M: 판정 metric·비교자 연루 | Stage 0 통계 계측 | P025 | (C) | — |
| SC20 | `compare_smooth_baseline.py:42`; `analyze_bfdark.py:77`; `plot_jbmap_pump_location.py:21` | 표시용 log/semilogy floor | floor | plot 전용 | 정당 | N | 신규 | GB 범위 밖 | N: 직접 연루 근거 없음 | 유지; 계측 | P094 | (C) | — |

## SC05 86개 발생 위치

모두 같은 `X_Fe = max(5e-4, ...)` 물리 조성 floor이며, 전수표 SC05의 `file:line` 전개이다.

```text
slurm_ddc15_223_ionlock_smoke.sh:146
slurm_ddc15_223_perionresc_smoke.sh:141
slurm_ddc15_A1_eps_fine.sh:141
slurm_ddc15_A1b_eps_dense.sh:141
slurm_ddc15_A1c_eps_top.sh:139
slurm_ddc15_A1d_e1p00_mcvar.sh:142
slurm_ddc15_A1e_e0p70_mcvar.sh:143
slurm_ddc15_A2_s2W.sh:156
slurm_ddc15_C2_xFeOuter.sh:129
slurm_ddc15_D1_Linner.sh:130
slurm_ddc15_F1_oskip.sh:131
slurm_ddc15_FI_ablation.sh:149
slurm_ddc15_FI_prod.sh:143
slurm_ddc15_G1_xFeInner.sh:134
slurm_ddc15_H1_epsUV.sh:134
slurm_ddc15_H1b_epsUV_knee.sh:131
slurm_ddc15_H1p_production.sh:135
slurm_ddc15_H2_epsUVred.sh:132
slurm_ddc15_H2p_redonly.sh:147
slurm_ddc15_H3_fate_attribution.sh:128
slurm_ddc15_I1_NiII_UVidown.sh:133
slurm_ddc15_J1_SiII_UVidown.sh:135
slurm_ddc15_K1_NiII_Aul.sh:136
slurm_ddc15_K1b_NiII_Aul_strong.sh:135
slurm_ddc15_KL1_stack.sh:142
slurm_ddc15_L1_SiII_Aul.sh:137
slurm_ddc15_M1_FeII_Aul.sh:136
slurm_ddc15_N1_KLM_stack.sh:142
slurm_ddc15_O1_CoCr_stack.sh:141
slurm_ddc15_P1_FeCo_push.sh:143
slurm_ddc15_P2_FeCo_stack.sh:142
slurm_ddc15_Q1_eps_uv_on_stack.sh:145
slurm_ddc15_Q1b_lambdamin_iron3_red.sh:144
slurm_ddc15_Q1c_lambdamin_wide.sh:141
slurm_ddc15_R1_eps_uv_2step.sh:149
slurm_ddc15_R2_aszeta.sh:150
slurm_ddc15_S1_siII_opt_aul.sh:149
slurm_ddc15_S2_siII_opt_finer.sh:147
slurm_ddc15_T1_s1_h2_stack.sh:151
slurm_ddc15_U1_feII_opt_aul.sh:152
slurm_ddc15_U1_ni2_opt.sh:151
slurm_ddc15_V1_c2_h1b_prod.sh:140
slurm_ddc15_W1_ca2_boost.sh:147
slurm_ddc15_X1_u1f005_ca2.sh:142
slurm_nlte3_diag.sh:138
slurm_nlte3fix_femerge.sh:153
slurm_nlte3fix_femerge_bare.sh:146
slurm_nlte3fix_femerge_combo.sh:145
slurm_nlte3fix_femerge_drop.sh:152
slurm_nlte3fix_femerge_optscan.sh:143
slurm_nlte3fix_femerge_probez.sh:150
slurm_nlte3fix_femerge_struct.sh:151
slurm_nlte3fix_femerge_sweep134.sh:145
slurm_nlte3fix_femerge_sweep34.sh:148
slurm_nlte3fix_optA.sh:145
slurm_nlte3fix_optAp.sh:148
slurm_nlte3fix_optC.sh:154
slurm_nlte3fix_optD.sh:153
slurm_nlte_o_prod.sh:144
slurm_nlte_o_recal.sh:145
slurm_nlte_o_recal_prod.sh:155
slurm_nlte_o_recal_seed.sh:149
slurm_nlte_o_seed.sh:148
slurm_nlte_o_smoke.sh:144
slurm_o_triplet_prod.sh:147
slurm_o_triplet_smoke.sh:147
slurm_plain_ddc15_sn2002bo.sh:153
slurm_v1_epoch_bracket.sh:157
slurm_v2_hst_epoch_bracket.sh:140
slurm_v32_4epoch.sh:145
slurm_v3_4epoch_w5frozen.sh:146
slurm_v3_de_l_sweep.sh:134
slurm_v3_epsir_sweep.sh:133
slurm_v3_vinner_sweep.sh:133
slurm_v4_ablation.sh:164
slurm_v4_inversion.sh:134
slurm_v4_probe.sh:133
slurm_v4_smoke.sh:143
slurm_viLzeta_grid.sh:158
slurm_w1_p37_retune.sh:145
slurm_w2_logL_red5_diag.sh:148
slurm_w3_nir_damp.sh:148
slurm_w4_ni2_nir_push.sh:146
slurm_w5_stack_closer.sh:146
slurm_zeta_clean_mcvar.sh:161
slurm_zeta_clean_smoke.sh:158
```

## 원본 대비 항목 수 검산

- 원본 전수표: C01–C73 73행 + SC01–SC21 21행 = **94행**
- 본 레지스트리: 위험 45행(대체형 23행 + 기타 유형 22행) + 비위험 49행 = **94행**
- ID 집합 대조: 누락 0, 중복 0, 추가 0
- 물리검증 join 열 대조: 94행 × 4열 모두 기입, `priority` P001–P094 중복·누락 0
- 도입사유 열 대조: `intro_reason` 94행 모두 기입, A형 30행의 `upstream_bug` 모두 기입

## 조인 코드북과 산정 규칙

- `oracle_verdict`: `GB-*-일부`는 Gate B의 strict/context 행 가운데 동일 물리범주가 있어 **동일 행의 1%급 전후효과**만 계량 가능하다는 뜻이다. 전체 582행 중 strict 99행(17.01%)이므로 전축 성공 판정으로 확대하지 않는다. `GB-coll strict 0`과 `GB 범위 밖`에는 성공 판정을 붙이지 않는다. Wave 1/2 표적 네 건은 실측 효과를 직접 기록했다.
- `band_link`: `D`는 formal source/opacity/τ·구적에 직접 연결, `E`는 packet energy/event ledger 연결, `I`는 state/rate를 통한 간접 연결, `M`은 판정 metric/comparator, `N`은 기존 문서에서 직접 연결 근거가 없다는 뜻이다. `D`의 공통 관측은 총입력 대비 ×18.07 비보존과 formal 출력의 2500–5000 Å 점유율 71.80–72.49%다. `I`는 반드시 `relT 착지 후 확정`이다.
- `fix_stage`: Wave1/2 및 `FORMAL_FIX`가 이미 정물리 우회경로를 만든 경우를 먼저 표기하고, 나머지는 동등화 계획 v2의 최초 구조 수리 Stage를 적었다. `계측`은 삭제가 아니라 오차상한·발화량·에너지 장부를 먼저 확보한다는 뜻이다.
- `priority`: P001이 최상위다. 정렬 잣대는 **판정 잣대 오염 > 보존 위반 연루 > 인구 대체 > 기타**이며, 같은 층에서는 B형 즉시성, A형 군집 승수, 직접 계량, 생산경로 여부 순으로 정했다. C09 중심 군집(C09/C22/C40/C44/C66)은 한 상류 수리가 다섯 clamp를 함께 퇴역시킬 수 있어 승수를 적용했다.
