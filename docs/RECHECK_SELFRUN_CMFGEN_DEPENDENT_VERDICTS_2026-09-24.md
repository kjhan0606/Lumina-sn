> 작성: 2026-09-24, 조사 = general-purpose 서브에이전트(`claude-opus-5-5`, 읽기 전용), 발주 = 운전자. user 지시: 「지금까지 검증에서 실패한 것들도 다시 돌아봐야 한다 — 틀렸다고 한 결정들이 아닐 수도 있으니」.
> 계기: 자체 CMFGEN 런(`/gpfs/kjhan/cmfgen_runs/toy06_19.48d*`)의 에너지 수지 파탄 확인(붕괴 가열 0·Lstar 7.55e43 = 공개 L_bol 의 5.9×) — memory `project_s_overionization_shared_cause_2026-09-24.md`.
> **운전자 추가 확인(2026-09-24, StaNdaRT 논문 A&A 668, A163 CMFGEN 절)**: 공개 CMFGEN 은 비열 과정 포함(Spencer-Fano)·시간의존 율·전달·에너지 방정식·비국소 MC γ선 침적. ⟹ 1순위(NT OFF 결정)의 근거는 공개 CMFGEN 과 **반대**다. 또 07-20 NO-COLD-ROOT 는 장 고정 시험이라 장-인구 쌍안정을 원리적으로 볼 수 없다(운전자 판독).
> 이 문서는 재점검 **목록**이다 — 판정 번복이 아니다. 각 항목은 별도 재판정을 거친다.

# 자체 CMFGEN 런에 기댄 과거 판정·결정 목록 (읽기 전용, 수정 없음)

**결론:** 가장 먼저 다시 볼 것은 비열(NT) OFF 결정입니다. 근거는 자체 런 설정 한 줄뿐이고, 1주일 사다리의 동기도 이 결정에 기대고 있습니다. 현 생산 런(run2)의 입력에는 자체 런 파생물이 없습니다. 다만 7월 FUV·Γ 수사와 7~8월의 장·광도 판정 상당수는 자체 런 값을 분모로 썼습니다.

## 0. 공통 사실
- **같은 계열 런은 전부 같은 결함을 가집니다.** 대상은 `toy06_19.48d`, `_jnu4`, `_conv`, `toy06_19p48d_modern`, `_relT3`, `_ophys` 입니다.
  - 모두 `INC_RAD_DECAYS=T`, `DO_DDT=F` 입니다.
  - RVTJ 에 "Was T fixed?: T" 가 찍혀 있습니다. `_conv` 도 VADAT 는 FIX_T=F 이지만 RVTJ 는 T 입니다.
  - 붕괴 광도는 0 입니다(OBSFLUX "Total Rad. decay luminosity 0.0000E+00"). jnu4 는 RVTJ 의 "Heating: radioactive decay" 블록도 전부 0 입니다.
- **VADAT 는 우리가 만든 설정입니다.** `docs/CMFGEN_BUILD_RUN_GUIDE.md:52-53` 에 `F [TRT_NON_TE] ← 첫 모델 단순화; 완전물리는 T` 라고 적혀 있습니다. 따라서 자체 런 설정은 공개 CMFGEN 설정의 증거가 아닙니다.
- **이미 있었던 경고들**
  - 07-23 dig_B6 가 "잣대 사고 6호: jtable 자체가 미수렴 자체런의 장"을 기록했습니다(memory `project_artis_parity_campaign.md:88`).
  - 07-31 `CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:18-20` 이 고정온도 오라클 주장을 철회했습니다.
  - 08-04 `CODEX_A2_00_ORACLE_ELIGIBILITY.md:166` 이 INELIGIBLE 로 판정했습니다.
  - 08-20 `RETRACTION_INELIGIBLE_ORACLE_CITATIONS_2026-08-20.md` 가 인용을 회수했습니다.
  - 세 경고 모두 사유는 수렴·FIX_T 였고, 에너지 수지는 다루지 않았습니다.
- **jnu4 가 쓸 만한 구간은 s0–s8 뿐입니다.** 자체 런 n_e 는 공개값 대비 v≲10,300 km/s 에서만 5% 이내입니다. v=14,000 km/s 에서 0.048×, 30,000 km/s 에서 0.29× 입니다(`docs/OUTSIDE_LOOP_POOL.md:1095-1119, 1200-1227`). sh24(황) 영역은 오염 구간입니다.
- **완화 증거 (추정):**
  - dig_F9(memory `project_artis_parity_campaign.md:129`)에서 jnu4 GENCOOL 의 d54(s8) 순냉각이 4.508e-5 였습니다. 이는 공개 침적 4.574e-5 의 0.986× 입니다.
  - jnu4 J 를 이식했을 때 Fe f(IV) 가 공개 ionfrac 을 재현했습니다(메모 `reference_cmfgen_published_toy06_19p48d.md:57,66`).
  - 따라서 심부·광구(s0–s8)의 J 는 크게 틀리지 않았을 수 있습니다. 과잉 광도는 경계 쪽이나 외곽에 몰렸을 가능성이 있습니다. 그래서 s0–s8 판정은 "무효"가 아니라 "부분"으로 분류했습니다.

## 1. 판정·결정 표

| 날짜 | 경로:행 | 요지 | 잣대 | 자체 런이 무효면 뒤집히나 | 우선 |
|---|---|---|---|---|---|
| 07-18 | validation/cmfgen_toy06_19p48d/analysis/GRADIENT_BUDGET_VERDICT.md:125 | Fe 기울기 결손 4.4 dex 는 장의 몫, 율은 무죄 | 혼합: J·Γ 는 jnu4, T·n_e 는 공개 | 부분. +6.67 dex 크기는 jnu4 값이고, 방향은 공개 ionfrac 로 재확인됨 | 중 |
| 07-19 | analysis/F0_F1_FUV_GRADIENT_VERDICT.md:11; FUV_GRADIENT_ATTACK_DESIGN.md:56,83 | 심부 FUV 결손 −1.54 dex, F3-T(T 핀) FAIL | 혼합: FUV 목표는 jnu4, TE_TABLE 은 공개 | 예. 목표 2.02e-4 가 과광도 런의 값 | 상 |
| 07-19 | reference 메모 :75,79 | F3-B 경계색 단독 "기각"(+0.46 dex) | 자체 런 진폭 | 예. 기각 문턱이 jnu4 진폭 기준 | 상 |
| 07-19 | analysis/trapping_audit/VERDICT.md:16-18 | 트래핑 기각, u(s0) 0.576×, "가스가 욕보다 차다" | 자체 런: u_CMFGEN=695 는 EDDFACTOR 적분 | 예 | 상 |
| 07-19 | analysis/radeq_ledger_audit/VERDICT.md:94 | radeq 균형 무죄(CMFGEN J 를 넣으면 T_e 18277 K ≈ 공개 18760 K) | 혼합 | 부분. 대조 대상 T 가 공개값이라 비교적 강건 | 중 |
| 07-19 | 메모 :56-57,66 (#33 jtable, 쌍둥이 179596) | 장 주입만으로 기울기 출현, "Fe CMFGEN 완전 재현·율 무죄" | 입력은 자체 런 J, 판정은 공개 ionfrac | 판정 쪽은 아니오. "CMFGEN 장 = 정답" 전제만 자체 런 | 중 |
| 07-19~20 | reddening_localization, axis2_valley_forensics:106,150 | Lumina 장의 적색 편중, 골짜기 cs_J 가 CMFGEN 초과 | 자체 런 J | 부분/예 | 중 |
| 07-20 | 메모 :124-126 (kpr3/kpr4) | "심부 FUV 진폭 CMFGEN 일치", Phase 1 완결 | 자체 런 | 예. 공개 기준으로는 과대 복원일 수 있음(추정) | 상 |
| 07-20 | 메모 :144; analysis/over_recomb_s4 | s4 Γ 30× 과소, α 무죄, SPINGATE 기각 | 자체 런 Γ_C | 예 | 상 |
| 07-20 | 메모 :154-156; photosphere_ledger | COLD-CASE-P: CMFGEN Fe III b_k 열적, lev17 트랩 | 자체 런 FeIIIOUT | 부분. ARTIS 로컬 런도 b_k≈1 을 지지(메모 :210) | 중 |
| 07-20 | analysis/formation_map/FORMATION_MAP_VERDICT.md:4 | FUV 형성 속도 CMFGEN 10,706 km/s 대 Lumina 31,928 km/s | 자체 런 (`_cmfflux`) | 예 | 중 |
| 07-20 | bistability, residual_offset_budget, te_bias_budget VERDICT | NO-COLD-ROOT, Wien 그림자 기각, C1 기각 | 혼합 | 부분 | 중 |
| 07-21 | 메모 :184,186 | s8 FeIII Gph 가 CMFGEN 대비 2209× 과대, CMFGEN J/B 0.06–0.08 이 정상 기준 | 자체 런이 분모 | 부분. f(FeIV) 사상은 공개, 배수는 jnu4 | 중 |
| 07-23 | memory project_artis_parity_campaign.md:81 (dig_B2) | 콤 = Lumina 솔버 아티팩트, lev17 부호 반대, "lev33-40 저인구" 철회 | 자체 런 POP (it40 미수렴) | 부분. dig_B4(:86)가 v<10k 에서만 강하다고 한정함 | 중 |
| 07-23 | 같은 파일 :88 (dig_B6) | S ×9 는 율 무죄, 진범은 EUV 장(+633 K) | 혼합 | 이미 "잣대 사고 6호"로 인정됨. 결론은 공개 r=5.53 기준 | 하 |
| 07-24 | 같은 파일 :111,128 (dig_F3·F8) | lam 형식 무죄, "EUV 부양 = T_e +904 K × 콤" | 자체 런 J(진리 J) | 부분 (s8 이면 유효 구간) | 중 |
| 07-24 | 같은 파일 :129-130 (dig_F9, withParityN) | +904 K = 선충돌냉각 33%, Si III 12× 결손; 이후 CMFGEN b=0.975 로 F9 가설 반증 | 자체 런 GENCOOL·b_k | 예. "CMFGEN 냉각 4.51e-5" 가 FIX_T 런의 항별 값 | 상 |
| 07-25 | 같은 파일 :137-139 (1113 Å 미지채널, EUV 진리) | 1113 Å 초열 = MC 아티팩트, "J=B(T_e) 폴백 수리안 기각" | 자체 런 J (s8) | 부분. 스파이크 부재 판정은 강건, 배율(10.5×)은 jnu4 기준 | 중 |
| 07-29~30 | 같은 파일 :857,903; VERIFICATION_REGISTERS.md:135 | K6: DIE 전부 F 인 설정을 "잣대 충실"로 보고 전 DR-OFF(p44) 채택 | 자체 런 **설정** | 부분. 공개 CMFGEN 의 DR 설정은 미확인. 09-19 DR-ON 이 부스트를 켬 | 중 |
| 07-29 | rates_certification, gamma_coiii_alllevel | 율 기계 산술 인증(~1%) | 자체 런 내부 일관성 | 아니오. 산술 검사라 해가 틀려도 유효 | 하 |
| 07-31 | CODEX_W3_GAMMA_TRIPLE_COMPARE:7-8 | "동결장 내용이 진범"(Fe III C48, S II SL4) | 자체 런 J | 예. S II 는 오염 구간 | 상 |
| 08-02 | CODEX_EMISS_E8:8-14; FABLE_UV_CENSUS | s8 장 11.98× 과잉이 산란 재순환으로 닫힘 | 자체 런이 분모 | 부분 | 중 |
| 08-03 | CODEX_I8_LUMINOSITY_SUMMARY:5-12 | I8 잔류: CMFGEN L(3900)이 Lumina L_inj 보다 15% 높음 | 자체 런 광도 | 예. L 이 5.9× 부푼 런의 광도 | 상 |
| 08-03 | CODEX_LEVER_PARITY59:24; OUTSIDE_LOOP_POOL:753 | T_e +2468 K 비가산 결합; 덱의 <Z>=1 placeholder | 혼합 | 부분 | 중 |
| 08-04~05 | ORDER_L0_JNU_OWNER; CODEX_IMPL_A2_02B | L0 J 소유자, A2-02B CMFGEN J 표를 jnu4 로 구축 | 자체 런 | 예 (이 표를 소비한 A2 판정 전부, 추정) | 중 |
| 08-17~18 | VERDICT_A210_STAGE4_JO:6, VERDICT_DET_SPROD_IV:6 | 흡수 결손 1/172, net 비 4.37e5 | 자체 런 `_ophys` | 이미 08-20 회수. 남겨 둔 정성 명제 "CMFGEN 은 양 부호"도 재검토 필요 | 중 |
| 08-19 | ORACLE_DIAGNOSIS_2026-08-19:5-30 | "잣대는 `_modern` 을 더 돌리면 싸게 복구된다" | 자체 런 수렴 이력 | 예. decay 0 이면 수렴해도 틀린 해 | 중 |
| 08-20~09-09 | STAGE12, BIT-LOCKSTEP, L6C, SLB, Z2A, DUAL-AUTH(0.884), DC1-NECLOSE(NC-B), 쌍둥이 지도 | 각 판정 | Lumina 내부 또는 공개 | 아니오 | 하 |
| 09-09 | VERIFICATION_REGISTERS.md:545 | toy06_19.48d 인용 146건 성격 분류 — 미완 | — | 이번 작업의 기존 대기 항목 | 상 |
| 09-17 | READOUT_S_IONIZATION_EUV_LTE:65-78 | 중간 셸 "이온화 쌍안정 후보", 외곽 J 10³–10⁴× 과잉 | 자체 런 jtable | 예 | 중 |
| 09-17 | PREREG_EUV_JSUBST:9,151 | EUV J 를 jtable 로 치환하는 계약 (미실행) | 자체 런 | 재가동 금지 | 하 |
| 09-17 | CLOSEOUT_CODEX_PERIOD:28-44 | 쌍둥이 T_e 로 평균전하 차 0.097→0.031 dex | 공개 | 아니오 | 하 |
| 09-18~19 | VERDICT_DET_LTEUNMAP_EPS, VERDICT_DET_DR_ON:24,42 | LU-A; DR 필요·불충분(공개 대비 3.2–20×) | 공개 | 잣대 쪽은 아니오. 단 "불충분" 잔여를 NT 가 메울 수 있음 | 중 |
| 09-18→21 | READOUT_DEGENERATE_LEDGER:12-17 | NT 포함 f_II .42–.64 예측을 "폐기된 전제, 인용 금지"로 처리 | 폐기 근거가 자체 런 | 예 | 상 |
| **09-21** | **memory project_det_djdn_drive_closed_2026-09-21.md:32; PREREG_DET_DJDN_SPLIT:199** | **비열 기각: 자체 런 VADAT `F [TRT_NON_TE]` 를 근거로 Lumina 도 NT OFF** | **자체 런 설정 (가이드가 "첫 모델 단순화"라고 명시)** | **예** | **상** |
| 09-21 | PREREG_DET_PATHB_FULL:28, PATHB_LOCAL:40 | "축 = 솔버, Path B(NT OFF) 확정 선" | 혼합 | 부분/예 (전제의 절반이 NT OFF) | 상 |
| 09-20~24 | VERDICT DJDN_MVP / DRIVE / PATHB_LOCAL / XSHELL | 착지 0/5 등의 측정 | Lumina 내부 | 측정은 아니오. 해석 틀은 NT OFF 전제에 달림 | 중 |
| 09-24 | PREREG_DET_CMFGEN_FIXEDPOINT:11,20,37-41 | 권위 상태 = jnu4 IREC 66 | 자체 런 | 예. 재기안 필요. referent x_C=0.724 만 공개 | 상 |
| 09-24 | PREREG_DET_CMFGEN_FIXEDPOINT_S1:7-19 | 율 반쪽 비교: jnu4 J_C 와 SCRTEMP65 로 Γ_L 대 Γ_C, K | 자체 런 | 부분. α_rr·σ 산술 대조는 유지 가능, Γ 절대값과 K 는 무효 | 상 |

## 2. 현 생산 입력(run2) 중 자체 런 파생물

| 파일·env | 원천 | run2 에서 켜져 있나 | 영향 |
|---|---|---|---|
| `LUMINA_FIXED_TE_PROFILE` → `input/fixed_te_profile.txt` | **공개** phys @19.480, temp 열. 최근접 샘플로 5자리까지 같음 (s0=18900 ← v4183, s24=9231.3 ← v21630). `scripts/cmfgen_twin_map.py:1053` 가 evidence 에서 복사. 자체 런 T 도 FIX_T 라 같은 값 | 켜져 있음 | 없음 |
| `LUMINA_DEPOSITION_FILE` → `model/deposition_cmfgen.csv` | **공개** `edep_toy06_cmfgen.txt` 19.48d 열을 셸 중앙 속도로 보간. s0=1.5069e-3 대 공개 보간 ≈1.507e-3, s24=6.63e-8 대 6.71e-8(v21705) | 켜져 있음 | 없음 |
| `data/cmfgen_jtable_toy06_19p48d.bin` (`LUMINA_GPH_JTABLE`) | **자체 런**: `.json` 에 `source_edd = toy06_19.48d_jnu4/EDDFACTOR`, `build_cmfgen_jtable.py:119-120` | **꺼져 있음** (exports 에 없음) | 7월 판정만 해당 |
| `data/cmfgen_te_table_toy06_19p48d.csv` (`LUMINA_TE_TABLE`) | 공개 phys (`build_cmfgen_te_table.py:10`) | 꺼져 있음 | 없음 |
| `LUMINA_TINNER_COLOR=18760` | 공개 phys s0 부근 T | 꺼져 있음 | 없음 |
| `input/cmfgen_coordinate_reference.json`, `cmfgen_netrate_path.txt` | **자체 런** `toy06_19p48d_ophys` 의 LINEHEAT/NETRATE/RVTJ | `job.slurm` 에서 참조 없음. `compare_a210_cmfgen_line_saturation.py`·`monitor_*.sh` 만 소비 | 물리 영향 없음. 진단 비교치만 오염 |
| `DECK_PROVENANCE.json` 의 `cmfgen_links` (`_ophys/atomic_links.txt`), `cmfgen_sigma_bf.bin`, `*_col_cmfgen*` | CMFGEN **원자자료** (런의 해가 아님) | 켜져 있음 | 없음 |
| `electron_densities.csv` | TARDIS 계열 초기값 (s0 1.61e9 대 공개 5.23e9). 자체 런 아님 | 초기값 | 없음 (NE_CLOSURE 가 다시 풂) |
| **Lumina 설정 결정**: NT OFF (`LUMINA_ION_NT` 기본 OFF), 전 DR-OFF 기준선 (K6) | **자체 런 VADAT 설정**을 "CMFGEN 발자국"으로 삼은 것 | NT 는 OFF. DR 부스트는 09-19 부터 ON | **큼** — 아래 1번 |

## 3. 뒤집힐 가능성이 큰 상위 5건

1. **NT OFF 결정 (09-21)과 퇴화 장부 NT 예측의 "인용 금지" 처리.**
   - 근거가 자체 런 VADAT `F [TRT_NON_TE]` 하나뿐입니다. 그런데 가이드 스스로 이것을 "첫 모델 단순화, 완전물리는 T"라고 적었습니다(`CMFGEN_BUILD_RUN_GUIDE.md:53`).
   - 공개 toy06 CMFGEN 의 비열 설정은 확인된 적이 없습니다.
   - 반면 NT 를 포함한 퇴화 예측(f_II .42–.64, r_deg 기준 .75/.66/.55)은 공개값(.85/.80/.71)에 가장 가까운 예측이었습니다.
   - 이 결정이 09-21 이후 "축 = 솔버, Path B" 사다리 전체의 전제입니다.
2. **DET-CMFGEN-FIXEDPOINT 와 S1 계약.** 주입 상태와 J_C 가 모두 jnu4 에서 오고, sh24 는 n_e 가 오염된 구간입니다. 살릴 수 있는 것은 α_rr·σ 산술 대조뿐입니다.
3. **7월 FUV·u 진폭 계열.** F0/F1 결손 −1.54 dex, trapping 기각(u 695), F3-B 기각, kpr3/kpr4 "CMFGEN 일치"가 여기에 속합니다. 목표값이 모두 광도가 5.9× 부푼 런의 J 입니다. 기각된 가설(경계색, 트래핑)이 실제로는 충분했을 수 있고, "일치"는 과보정일 수 있습니다.
4. **자체 런 J·Γ·광도를 분모로 쓴 배수형 판정.**
   - I8 경계 광도 +15% 잔류
   - W3 "동결장 내용이 진범"
   - s4 Γ 30× 과소, s8 Gph 2209× 과대
   - dig_F9 "선충돌냉각 33%"
   - 배수의 크기와, 그 크기로 내린 기각·귀속이 흔들립니다.
5. **K6 전 DR-OFF 기준선.** "잣대 충실"의 근거가 자체 런의 DIE=F 설정이었습니다. 공개 CMFGEN 의 DR 처리는 미확인입니다.
   - 같은 계열로, 08-19 "잣대는 `_modern` 을 더 돌리면 복구된다"는 결론도 decay 0 이면 성립하지 않습니다.

**유지될 가능성이 큰 것:**
- 공개 ionfrac 로 판정한 jtable·쌍둥이·kpr5jt 의 Fe 결과
- 율 산술 인증(rates_certification, gamma_coiii)
- 쌍둥이 지도 0.097→0.031 dex
- LU-A, DR-ON 의 잣대 측(공개)
- 08-20~09-09 의 Lumina 내부 판정 전부

**권고 (추정):**
- 공개 StaNdaRT 배포본에서 toy06 CMFGEN 의 NT·DR 설정 출처를 먼저 확인하는 것이 1순위입니다.
- 그다음 VERIFICATION_REGISTERS.md:545 의 미완 분류(146건)를 이 표로 대체하면 됩니다.
