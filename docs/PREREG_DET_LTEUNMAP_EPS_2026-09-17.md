# 사전등록 — DET-LTEUNMAP-EPS: NLTE 밖 이온 선의 LTE 열적 방출 폐기(두준위 ε_phys 소스) (2026-09-17)

저자 = 운전자(Claude, 개정25). 상태 = **기안 / AUDIT_PENDING**. 실행 승인이 아니다.
근거 판독 = `docs/READOUT_IMPLIED_J_LEDGER_2026-09-17.md` + `docs/READOUT_LAMP_PREDICTION_2026-09-17.md`. 상위 = `docs/CLOSEOUT_CODEX_PERIOD_2026-09-17.md`.

## 1. 계약 — 물리 계약 1개

**「`cmf_fine` 조립기는 NLTE 밖 이온(LTE-unmapped)의 선을 LTE 상준위 채움으로 열적 방출하지 않는다 — 두준위 원자: 열적 분율 ε_phys(선,셸) 만 B(T_e) 로 방출하고, (1−ε_phys) 는 산란 불투명도로 접는다.」**
판정런 1회(twin-T_e, 343204 계보와 env 1 키 차이). 참이면 외곽 창 빈 스파이크(J/B 10³–10⁶)가 꺼지고 셸 25–40 의 S·Si 가 IV 에서 III 로 복귀한다.
⚠이 계약은 **중간 셸(9–24)의 S II 가 CMFGEN 으로 복귀한다고 말하지 않는다**(그것은 연속체 Kirchhoff-LTE 채움 ①, 별개 rung — 판독 A §3). **NLTE 밖 이온을 NLTE 로 푸는 것(CMFGEN 발자국)이 정도(正道)**이나 램프 이온이 S V·Ca V 라 기존 STAGE4 표로 닿지 않는다; 이 계약은 「우리가 풀지 않는 이온에 대해 코드가 이미 가진 ε 기전을 parity 분기에도 적용」하는 것으로, 새 물리를 넣지 않는다.

### Q-GOAL (기안자 자기 답 — 사전감사가 독립 재판정)
① 줄이는 칸: 어긋남 지도의 **외곽 25–40 S/Si III→IV 과이온화**(S IV 0.5–0.99 → CMFGEN 0.03–0.15)와 중간 셸 σ-가중 Γ 초과(3–55×)의 램프 성분. 크기 = 판독 A §2 표. ② 물리 측정(이온 분율·창 빈 J 응답). 신규 계측 = stderr census 1 행/셸(§4 G-EPS 가 소비). ③ 대안 = ① 연속체 채움 수리 — 질량이 큰 중간 셸을 겨냥하나 퇴화(판독 A §3)로 단독 예측이 불가해 **뒤**에 둔다; NLTE 승격(S/Ca IV·V) — 정도이나 표·슬롯·GPU 메모리 변경이 커 다음 단. ④ 건너뛰면 외곽 과이온화의 귀속(램프 대 다른 구동)이 미확정으로 남고, ① 수리 시 램프가 결과를 오염한다.

### Q-LEAN (기안자 자기 답)
L1 물리량 = 창 빈 J/B·σ-가중 Γ/Γ(B)·셸별 단계 분율(기존 판독 도구 재사용: `window_leak.py`·`gamma_locate.py`·`ion_share.py`·지도 판독기). 신규 = env 1·census 1 행/셸·selftest 1. L2 게이트 3(§4) — 각각 실재 가능한 실패(정체성 이탈·미완주·ε 미적용/무음 열화)만. L3 = 1 걸음(수리 1 + 판정런 1). L4 삭제 후보: 없음 — 추가하지 않은 것: 새 판독기·새 상태 필드·새 판정기·두 번째 GPU 런(음성대조는 CPU selftest).

## 2. 기전 — 런 0회 확정분

- 이온비 = σ-가중 J(재적분 0.5–3 % 재현; `Γ(B)/α=Φ` 0.1 %). 초과의 45–85 % 는 창 빈(557.7 Ca V·696.7/922.5 S V …), 방출력 96–99 % 가 셸 41–49(T_e 12–24 kK, twin 핀).
- 방출 코드: `cmf_fine_line_material` parity 분기(`src/lumina_cmfgen.c`, `cmf_fine_line_material`, 4498–4540 행) η₀ = n_upper·A_ul·hν/4π(`line_net_sobolev_material`, `src/line_net_rate.c` 106 행); n_upper = `upper_population_cache` ← `lumina_line_upper_population_fill_for_tau`(`src/lumina_plasma.c` 9019 행) ← `a209_upper_population_for_tau` 의 `lte_level_density`(**`used_nlte==0`** 인 선) ⟹ S_l = B(T_e) 정확히(실측 η/χ = 1.02 B; χ_bb 는 Saha+Boltzmann+Sobolev 로 3자리 재현).
- parity 분기는 `line_eps`/`radeq_line_eps_phys` 를 **보지 않는다**(legacy 분기만 `emit = line_eps<1 ? line_eps·B : source`). 런의 `LUMINA_CMFGEN_LINE_EPS_PHYS=1` 은 bin 조립기용이다.
- ★**RADEQ 표는 343204 에서 한 번도 빌드되지 않았다**(stdout `[RADEQ] collisional line-cooling table` 인쇄 0 — T_e 고정 프로필이라 `radeq_simul_all` 미진입; 빌더 `build_radeq_line_table` 은 그 안과 `#ifdef LUMINA_FROZEN_ORACLE` 래퍼에만 있다). parity 패스는 **iter 0 조립에서 이미 돈다**(343204 stderr 20 행 `CMFGEN_NONOVERLAP_SOBOLEV`, `[R7][PHASE] iter=0` 이전). ⟹ ε 소비 전에 표를 **명시적으로 빌드**해야 한다. 표의 선 선택 술어(이온 pop 실재·두 준위 실재·dE>0)는 S V·Ca V 선을 포함한다(동결 트리 `lumina_plasma.c` 11087–11103 행 직독).
- NLTE 표적 31 슬롯 = 15 원소 × ion II·III(런 배너 `Z=.. ion=1/2`); 램프 이온 S V·Ca V 는 어떤 승격 표에도 없다.
- ★**개정 2 (구현자 위험 지적 → 운전자 소스 직독으로 확정)**: **커밋되는 J 는 fine 필드가 아니다.** 드라이버(동결 트리 `lumina_cmfgen.c` 7609–7690 행)는 매 패스 `cmfgen_assemble` → `cmfgen_solve_J`(binned) → `cmfgen_fine_jbar`(선 J̄ 생산만; 연산자 패스의 fine 필드는 `CONTINUUM_ONLY`, deposit 두 분기 5643·5722 행 모두 `!sobolev_operator`) → `cmfgen_commit_jnu(cs.J)` 순이다. ⟹ 발행·소비되는 J 의 램프 생산자는 **bin 조립기 선 루프**(`cmfgen_assemble_impl` 1868–1905 행): `Sl = src_nlte ? line_source_S : 0; if(Sl≤0) Sl=B` (런은 `LUMINA_CMFGEN_SRC_NLTE` 미설정 ⟹ 전 선 S_l=B) 이고, `LUMINA_CMFGEN_LINE_EPS_PHYS=1` 이 설정돼 `el = radeq_line_eps_phys(...)` 를 부르지만 **표 미빌드로 `el<0 → 1.0` 무음 열화**(1890 행 주석 `table not built: thermal`) ⟹ 전 선 완전 열적. 조립기는 두준위 접힘을 이미 갖고 있다(`chi_es = χ_e + (χ_ln − χ_ln,th)` 2167 행, `S_fixed=(χ_abs B+η_line)/χ_tot` 2175 행, 솔버 `S = S_fixed + (χ_es/χ_tot) J` 2664 행) — 결함은 표 부재의 무음 폴백 하나다. fine 경로 parity 분기의 LTE 채움(§2 위)은 INIT 패스 fine 필드와 선 J̄ 초기화에만 닿는다.
- **미확정(이 계약 밖)**: ① 연속체 `S_fixed=(χ_abs·B+η)/χ_tot`(`:5686`, bin 조립기도 동형) 의 LTE 채움 · 퇴화-파괴 채널(DR·비열적) · CMF 선숲 흡수의 정량(정적 예측기의 sh≤24 과대) · NLTE 사상 선의 ε 채널 활성(런 설정 의도이나 별개 rung — 아래 §3).

## 3. 수리안 — 변경집합

원칙: **parity 분기에서 `used_nlte==0` 인 선에 한해** ε_phys 적용. NLTE 사상 선·연속체·이온화 solve·율·원자자료·T_e 는 건드리지 않는다. env 부재 ⟹ **byte-동일**(음성대조 자격). ε 를 얻지 못한 unmapped 선(τ>1e-12) 이 하나라도 있으면 **fail-closed**(무음 열화 금지).
★**소스 기준 = 동결 트리** `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source`(343204 빌드 소스 `consistent_ionization_state_20260911/source` 의 검증된 복사본, 182 파일 해시 일치) — repo HEAD 가 아니다(HEAD 는 22 파일이 다르다). 변경집합 밖 파일은 byte 불변.
- **표 빌드 보장**: `lumina_plasma.c` 에 `#ifdef` 없는 공개 래퍼 `radeq_line_table_ensure(nlte, atom, opacity)`(내부 `build_radeq_line_table`, 멱등)를 두고, `cmf_fine` 은 게이트 ON 이면 parity 패스의 ε 소비 **전에** 이를 호출한다. 표가 빌드되지 않은 채 ε 소비에 들어가면 **시작 시 BLOCKED**(무음 진행 금지).
- **반환 의미**: `radeq_line_eps_phys` 의 `-1`(표 미빌드·미지 선)은 docstring 의 「legacy 열적 폴백」을 따르지 **않고** fail-closed 로 처리한다. `denom≤0 → 1.0` 반환은 `A_ul=0`(η₀=0 이라 무해) 또는 `C_ul=0 ∧ β_esc=0` 에서만 생기며, 후자를 만나면 census 에 계수하고 fail-closed 한다.

- 분류: 채움 루틴이 이미 계산하는 `used_nlte`(`a209_upper_population_for_tau` 출력)를 (선,셸) 바이트 배열로 **함께 내보내** `cmf_fine` 이 그것을 읽는다 — 재유도 금지(같은 술어).
- 적용: unmapped 선에서 `eps = radeq_line_eps_phys(line, n_e[s], T_e[s], τ)`; `η₀ ← η₀·eps`; 불투명도는 총량 보존 — 열적 `eps·χ₀` 는 `chi_line`, 산란 `(1−eps)·χ₀` 는 `chi_es`(기존 ALI 산란 알베도 경로) 로 적립. 프로파일 가중 p 는 둘 다 같은 식. ★per-line 접힘은 **누적단**(선 프로파일을 `chi_line`/`eta` 에 더하는 자리, 동결 트리 `lumina_cmfgen.c` 5611·5654 행 근방)에서 하고, 전역 스칼라 `line_eps` 조립단(5682 행 근방)은 건드리지 않는다 — `cmf_fine_line_material` 은 ε(또는 산란분 χ)을 **호출부로 반환**하도록 시그니처를 늘린다.
- census: 셸당 stderr 1 행 `[cmf_fine][UNMAPPED-EPS] shell=<s> n_unmapped=<n> n_mapped=<m> eps_min=<..> eps_median=<..> eps_max=<..> thermal_emit_fraction=<Σε η₀/Σ η₀>`(INIT 패스) 및 **같은 형식의 `[cmfgen_assemble][UNMAPPED-EPS] pass=<p> shell=<s> …`**(bin 조립기, 매 패스) — G-EPS 는 **조립기의 마지막 패스** 행을 소비.
- ★**개정 2 — 커밋 경로(bin 조립기) 적용**: `cmfgen_assemble_impl` 선 루프에서 게이트 ON 이면 ① 조립 전 `radeq_line_table_ensure()` 호출(표 부재 시 시작 BLOCKED) ② `used_nlte==0`(같은 발행 배열·같은 술어) 인 선은 `el = radeq_line_eps_phys(...)` **원값**(eps_floor/eps_cap 미적용, −1 → fail-closed) ③ `used_nlte==1` 인 선은 **현 동작 유지**(`el=1.0`, 열적) — 표가 빌드됐다고 mapped 선의 ε 채널이 켜지지 않도록 게이트 안에서 명시(그 활성은 별개 rung) ④ η_line·chi_line_th·chi_line_cls 는 기존 식 그대로(`el` 만 바뀜). 게이트 OFF ⟹ byte-동일(표도 빌드하지 않음).

### 변경집합 — CMF_FINE_UNMAPPED_EPS (표)

| 경로 | 담당 | 내용 |
|---|---|---|
| `src/lumina_cmfgen.c` | Opus 4.8 | **개정 5 (user 결정 1번)**: ① 조립기는 unmapped 선의 산란 몫 `w·(1−el)` 을 별도 배열 `chi_line_sc_unmapped[셸,빈]`(할당·해제 동반)에 누적하고 `chi_es` 합성은 그대로(bin 솔버 = 커밋 필드는 접힘을 본다) ② fine 솔버의 연속체 보간(`fs.chi_es = interp(csb->chi_es)`, ~5427 행)은 게이트 ON 이면 `csb->chi_es − csb->chi_line_sc_unmapped` 를 보간 — fine 연속체가 base 와 같은 구성(선 산란 미포함)이 되어 알베도 상한이 base 급으로 복귀 ③ bin 솔버 `cmfgen_solve_J` 마지막 ALI 반복의 셀별 상대 변화를 패스당 stderr 1 행 `[cmfgen_solve_J][CONV] pass=<p> n_ali=<n> max_rel=<..> p99_rel=<..> cells=<..>` 로 발행(게이트 무관, 기존 `rel` 계산 재사용; 판정에서 E-1 등급 결정에 소비) ④ 게이트 OFF ⟹ 배열은 0 이고 보간 결과 byte-동일. **개정 4: fine INIT 패스(parity 분기)에는 ε 를 적용하지 않는다** — 374587 실측: (1−ε)χ 산란 접힘으로 fine 필드 알베도 1−4×10⁻¹² → exact 고정점 128회 미수렴(residual 0.076, base 45회 수렴) → R6 BLOCKED. fine 경로 코드는 게이트와 분리해 **기본 비활성**(별도 env `LUMINA_CMF_FINE_UNMAPPED_EPS_FINE`, 미설정 = 원본 deposit 그대로)로 두고, census 도 조립기 것만 소비; env `LUMINA_CMF_FINE_UNMAPPED_EPS`(기본 0 = byte-동일)는 **bin 조립기 선 루프(커밋 경로)** 의 표 보장·unmapped 원값 ε·mapped `el=1` 유지·fail-closed·조립기 census 만 켠다 |
| `src/lumina_plasma.c` | Opus 4.8 | `lumina_line_upper_population_fill_for_tau` 가 `used_nlte` (선,셸) 바이트 배열을 함께 발행(함수포인터 typedef·등록·호출부 동반); `radeq_line_table_ensure()` 공개 래퍼(`#ifdef` 없음, 멱등) |
| `src/env_universe.h` | Opus 4.8 | `LUMINA_CMF_FINE_UNMAPPED_EPS` 등록(strict env 수용) |
| `tests/cmf_fine_unmapped_eps_selftest.c` | Opus 4.8 | CPU selftest: env 부재 (χ₀,η₀,chi_es) byte-동일 · unmapped 선 η₀ = ε·χ₀·B(1e-12) 및 chi_es += (1−ε)χ₀ · mapped 선 불변 · ε 부재(−1) → rc≠0 · 표 미빌드 → rc≠0 · census 행 형식 · **개정 5: `chi_line_sc_unmapped` 누적 = Σw(1−el)(unmapped 만) · fine 보간 입력 `chi_es − chi_line_sc_unmapped` = 게이트 OFF 의 `chi_es`(base 구성) · CONV 행 형식** |

### 변경집합 끝

## 4. 게이트 명부 (표)

| ID | 조건 | 음성대조 |
|---|---|---|
| G-ID | 343204 계보(`/gpfs/kjhan/lumina/cmfgen_map/consistent_ionization_state_20260911/retry_20260912/physical_run/{RUN_FOOTER.txt,input_manifest.sha256}`) 대비 차이가 exports 에서는 `LUMINA_CMF_FINE_UNMAPPED_EPS` 키 하나(★개정 3: 러너 GPU 프로필에 종속된 `LUMINA_CMF_FINE_MGPU_DEVICES` 는 물리 키가 아니므로 프로필 `h100-a100x2` 루트에서는 값 2 로 달라도 허용 — 동결 스테이저의 `sealed_mgpu_devices` 프로필 기제와 동일), source 에서는 §3 변경집합 표의 행 전부(`src/lumina_cmfgen.c`·`src/lumina_plasma.c`·`src/env_universe.h`·`tests/cmf_fine_unmapped_eps_selftest.c`)뿐임을 `identity.json` 으로 확인 | 이전 계보의 exports 해시를 identity 에 주입 ⟹ FAIL |
| G-RUN | 2 외부반복 완주, stderr `[cmf_fine][EXACT-MULTIGPU-EPOCH] status=OK` ≥ 2, `work/physics_comparison/physics_DET_iter0001.spectral.csv`·`work/lumina_ion_pops.csv`·`work/nlte_levels_final.csv` 실재 | 반복수 변조·덤프 결손 주입 |
| G-EPS | stderr **`[cmfgen_assemble][UNMAPPED-EPS]` 마지막 패스** 행이 셸 50개 전부 있고, 셸 41–49 에서 `thermal_emit_fraction ≤ 1e-2` 이며 `n_unmapped > 0` (개정 4: `[cmf_fine]` 행은 없어야 정상 — fine 몫 비활성) | census 를 ε=1 로 위조한 로그 ⟹ FAIL(thermal_emit_fraction=1); 행 결손 ⟹ FAIL; selftest 에서 표 미빌드 주입(ensure 미호출) ⟹ 시작 시 BLOCKED 시연 |

### 게이트 명부 끝

## 5. 기대치 — 런에서만 얻는 수 (판독 A §2 정적 예측; 자릿수 기대)

잣대를 base(343204 산출)에 먼저 적용: base 는 E-1·E-2·E-3·E-4 어느 문턱도 넘지 못한다(각 「지금」 값이 문턱의 10–10⁴ 배 밖).

| ID | 양 (기존 도구) | base(343204) | **기대(런 후)** |
|---|---|---|---|
| E-1 램프 | 창 빈 853/898/758/811 의 J/B(T_e), 셸 28·32·36 (12 값, `window_leak.py`) | 최소 1.4e3, 최대 6.8e6 | **12 값 전부 < 100** (정적 예측 0.03–15) |
| E-2 Γ | S II σ-가중 Γ(J)/Γ(B), 셸 28·32·36 (`rebalance_check.py`) | 2.2e3 / 5.1e3 / 3.1e3 | **셋 다 < 30** (예측 1.9/1.2/0.6) |
| E-3 S 외곽 | S f_IV 셸 32·36 (`lumina_ion_pops.csv`) | 0.933 / 0.992 | **둘 다 < 0.20**, f_III > 0.70 (예측 .990/.918) |
| E-4 Si 외곽 | Si f_IV 셸 28·32 | 0.963 / 0.997 | **둘 다 < 0.30** (예측 .016/.114) |
| E-5 중간 | S II→III Γ/Γ(B) 셸 12·16·20 | 3.2 / 12.7 / 55 | < 2 / < 4 / < 20 (예측 0.8/1.3/8.4) |
| E-6 대조(불변) | 셸 0–8 Fe·Co·Ni IV 분율; S f_II 셸 12 | IV 0.98; 0.004 | \|Δ\| < 0.02; **f_II(sh12) < 0.05**(CMFGEN 0.845 에 **가지 않는다** — ① 의 몫) |

### 분기 (표)

| 분기 | 조건 | 의미 |
|---|---|---|
| LU-X | G-ID/G-RUN/G-EPS 중 미충족 | 미시연 — 원인 명명 |
| LU-A | 게이트 전건 충족 **and** E-1 **and** E-2 **and** E-3 **and** E-4 | **램프 = 외곽 LTE-unmapped 선 방출 확정**, 외곽 이온화 III 복귀 ⟹ 다음 = ① 연속체 채움 rung(퇴화-파괴 채널 동반) |
| LU-B | 게이트 전건 충족 **and** E-1 **and not**(E-3 **and** E-4) | 램프는 꺼졌으나 외곽 과이온화 잔존 ⟹ 다른 구동(① 또는 창 밖 빈) — 판정자가 명명 |
| LU-C | 게이트 전건 충족 **and not** E-1 | 방출원이 unmapped 선의 LTE 열적 방출(bin 조립기·fine INIT)이 아니다(연속체·mapped 선·다른 경로) ⟹ 귀속 정정 |
| LU-M | 그 외 | 판정자가 명명, 재분기 금지 |

### 분기 끝

E-5·E-6 은 분기에 참여하지 않는다(기재·정정용). **개정 6(사전감사 R4 반영)**: `[cmfgen_solve_J][CONV] max_rel`/`p99_rel` 은 알베도≈1 인 창 빈에서 참오차를 **과소평가하는 하한**이다(솔버 주석: near-unity albedo 에서 패스당 변화는 오차를 과소평가, 권고 16 패스 이상; 이 런은 ALI 8회 고정). ⟹ 이 런의 **E-1·E-2 판정은 기본적으로 «관측 등급»**이며 분기 라벨에 「(관측)」을 붙인다. CONV 값은 기재하고, `max_rel > 0.1` 이면 「미수렴 의심」을 추가 기재한다. 정격 승격은 ALI 반복수를 늘린 별도 계약(user 결정)에서만. base 에는 이 진단이 없으므로 base 값은 「미측정」. 문턱 근거: base 와 예측 사이 **두 자릿수 이상** 벌어진 곳(정적 예측기의 자릿수 불확실성 ≤ 1 자릿수 — 판독 A §2 검증 표).

## 6. 자원 · 실행

GPU **1 런**, 파티션 h200→h100→a100(343204 실적 a100, 73.5 GB, 약 3.3 h) — `--time=05:00:00`, 상한 **10 GPU-hour**. 새 RUN/EVID 루트(봉인 밖). runner = 343204 의 `job.slurm` 계보 + env 1개(`LUMINA_CMF_FINE_UNMAPPED_EPS=1`). 오프라인 ≤ 1 CPU-hour(selftest·판독). 봉인 base(343204 산출)는 읽기만.
순서(개정16 ②): Opus 4.8 구현 → selftest(봉인 무접촉) → **운전자 코드 간단 감사(Q0 봉인 접촉 가부)** → 빌드 → 스테이징 → 제출.

## 7. 분장 장부 (개정25)

| 단계 | 규약 | 실제 | 비고 |
|---|---|---|---|
| 기안 | 운전자 | 운전자(본 문서) | **자기 기안** |
| 사전감사(Q-GOAL·Q-LEAN) | (개정25 미배정) | 미집행 → **Opus 4.8 fresh 발주 예정**(JSUBST 전례; GPU 지출 단) | 기안자 ≠ 감사자 |
| 구현 + selftest | Opus 4.8 fresh | 미집행 | `model` 생략 발주, 첫 줄 모델 ID 자기보고 |
| 코드 간단 감사 | 운전자 | 미집행 | 구현자 ≠ 검수자 |
| 스테이징·제출·판독·대장·커밋 | 운전자 | 미집행 | |
| 판정 | 운전자 | 미집행 | ⚠판정문에 운전자 이탈이 기재될 수 있음 — 종료감사로 보완 |
| **종료감사** | **Opus 4.8 fresh** | 미집행 | GPU 지출 단·물리 주장 등급 상승 단 — **생략하지 않음** |

## 8. 주장 상한
「외곽 창 빈 램프의 방출원이 LTE-unmapped 선 채움인가」와 「그것을 걷으면 외곽 이온화가 III 로 복귀하는가」까지. 중간 셸의 CMFGEN 복귀·① 의 수리·스펙트럼 영향·NLTE 승격의 필요는 말하지 않는다. twin 핀(외곽 20 kK)은 유지되므로 자유-T_e 에서의 크기는 범위 밖.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — CMF_FINE_UNMAPPED_EPS (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)",
    "symbol": "CMF_FINE_UNMAPPED_EPS",
    "roots": ["src", "tests"],
    "expected_extra": ["src/lumina_cmfgen.c", "src/lumina_plasma.c", "src/env_universe.h", "tests/cmf_fine_unmapped_eps_selftest.c"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "lamp_off_outer_iii": "sum(1 for x in v if 0.5 <= x < 1.5)", "lamp_off_outer_iv": "sum(1 for x in v if 1.5 <= x < 2.5)", "lamp_on": "sum(1 for x in v if 2.5 <= x < 3.5)", "other": "sum(1 for x in v if x >= 3.5)"},
    "rules": [
      {"name": "LU-X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "LU-A", "predicate": "count > 0 and gate_fail == 0 and lamp_off_outer_iv == 0 and lamp_on == 0 and other == 0"},
      {"name": "LU-B", "predicate": "count > 0 and gate_fail == 0 and lamp_off_outer_iv > 0 and lamp_on == 0 and other == 0"},
      {"name": "LU-C", "predicate": "count > 0 and gate_fail == 0 and lamp_on > 0 and other == 0"},
      {"name": "LU-M", "predicate": "count > 0 and gate_fail == 0 and other > 0"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "gate-fail", "mix": [[0.0, 1], [1.0, 2]]},
      {"name": "lamp-off-iii", "mix": [[1.0, 3]]},
      {"name": "lamp-off-iv", "mix": [[1.0, 1], [2.0, 1]]},
      {"name": "lamp-on", "mix": [[3.0, 1]]},
      {"name": "other", "mix": [[4.0, 1]]}
    ],
    "residual": "LU-X"
  },
  "references": [
    {"path": "docs/READOUT_IMPLIED_J_LEDGER_2026-09-17.md"},
    {"path": "docs/READOUT_LAMP_PREDICTION_2026-09-17.md"},
    {"path": "docs/CLOSEOUT_CODEX_PERIOD_2026-09-17.md"},
    {"path": "CLAUDE.md"},
    {"path": "src/lumina_cmfgen.c", "flags_existing": ["cmf_fine_line_material", "upper_population_cache", "LUMINA_CMF_FINE_LINE_EPS", "chi_es"]},
    {"path": "src/lumina_plasma.c", "flags_existing": ["lumina_line_upper_population_fill_for_tau", "a209_upper_population_for_tau", "radeq_line_eps_phys", "lte_unmapped_emit"]},
    {"path": "src/line_net_rate.c", "flags_existing": ["line_net_sobolev_material", "emission_per_sr"]}
  ]
}
```

---

## 집행 기록 1 — 사전감사 (Opus 4.8 fresh, `model` 생략 → 자기보고 `Opus 4.8 / claude-opus-4-8`) (2026-09-17)

**결과**: `Q_GOAL: 부합` · `Q_LEAN: 적정` · **`PRE_AUDIT: 통과 불가` · `BLOCKING: 1`** (171k 토큰, 20 tool use). 기안자(운전자)와 분리된 fresh 컨텍스트.

| | 내용 | 처분 |
|---|---|---|
| **«폐합 차단» 1 (C6)** | `radeq_line_eps_phys` 가 읽는 `radeq_lines` 는 `radeq_simul_all` 안에서만 lazy 빌드 — parity 패스가 iter 0 조립에서 돌면 표 부재 → 전 선 −1 → fail-closed → **런이 물리 없이 LU-X**. (b) 커버리지 미검증. (c) docstring(−1 → 열적 폴백)과 계약(−1 → fail-closed) 충돌; `denom≤0 → 1.0`. | **운전자 오프라인 확인**: 343204 stdout 에 `[RADEQ]` 표 인쇄 **0**(T_e 고정 → 표 미빌드), parity 패스는 stderr 20 행(iter 0 이전) — 지적 실재. 술어 직독(11087–11103 행): S V·Ca V 포함 — (b) 해소. **개정 1**(OLD/NEW 6쌍): `radeq_line_table_ensure()` 공개 래퍼 + 게이트 ON 시 ε 소비 전 호출·표 부재 시 시작 BLOCKED·−1 fail-closed·`denom≤0` 처리·selftest 항목 추가 |
| «기재로 족함» G1 | material 시그니처 확장·누적단(5611/5654) per-line 접힘 명시 필요 | 개정 1 에 반영 |
| G2 | `used_nlte` 는 (선,셸) 배열 — typedef·등록·호출부(5353) 동반 수정 | 개정 1 에 반영 |
| G3 | E-6 은 이온 분율만 잰다(내핵 스펙트럼 미검) — §8 상한과 정합 | 기재 |
| G4 | E-5 예측치(0.8/1.3/8.4)는 오염 구간 예측기 값 — 기대치로 읽지 말 것 | 기재(E-5 분기 불참 유지) |
| G5 | 「557.7 = Ca V 100 %」는 외곽(sh44–48) 한정, IGE 조성 0 에 의존 | 판정문에 기재 |

**독립 확증(감사자 소스 직독)**: parity 분기(4508–4534)는 ε·clamp 미참조 · n_upper ← `lte_level_density`(`!use_nlte`) · χ_line 재현 3자리 · 내핵 창 빈 불투명(J/B=1.00)이라 열화 길이 논증으로 E-6 정합 · ε 도달치 1e−5–1e−7 ≪ 1e−2.
**감사 자기 한계**: parity ε-소비 최초 반복 번호는 소스만으로 미확정(운전자가 로그로 iter 0 확정) · 표 술어 전문·`line_net_sobolev_material` 본문 미독 · ALI 의 내핵 J→B 회복은 정적 논증.
**운전자 처분**: 개정 1 적용 후 게이트 재통과 시 구현 발주(Opus 4.8 fresh, 동결 트리 복사본). 재감사는 개정이 «폐합 차단» 항목의 해소에 한정되므로 발주하지 않는다(정정 회차 규칙: 1회차 전건 반영).

## 집행 기록 2 — 구현 1차 착지 + 개정 2 (2026-09-17, 운전자)

**구현(Opus 4.8 fresh, 자기보고 `Claude / claude-opus-4-8`, 297k 토큰·81 tool use)**: 동결 트리 복사본에 변경집합 4 파일만 수정(`diff -rq` 확인), selftest 전 항목 PASS(fail-closed·mapped 불변·ε 공식 rtol 1e-9·불투명도 보존·census 형식), 두 TU `-Wall` 무경고. **구현자 자진 신고 3건**: ① `lumina.h` 미수정(G-ID 4 파일 우선) → typedef 대신 직접호출 함수 `lumina_line_used_nlte_fill_for_tau` ② `C_ul=0∧β=0` 은 도달 불능(β_esc∈(0,1]) — `A_ul=0` 만 `denom≤0` ③ ★**deposit(ε 접힘)은 INIT 패스에서만 돈다** — 연산자 패스는 선을 deposit 하지 않는다.
**운전자 소스 직독(③ 검증)**: 참이며 더 나아가 **커밋 J 는 bin 조립기 산출**(§2 개정 2 항목). 1차 구현대로 빌드하면 커밋 필드가 불변 ⟹ LU-C 가 **구성상 확정** — 「구성상 참을 예측으로 등록하지 마라」 계급. **빌드·제출 보류**, 개정 2 로 변경집합에 커밋 경로를 추가한다(OLD/NEW 5쌍). fine 경로 구현은 같은 게이트·같은 물리 문장의 INIT 패스 몫으로 유지.
**기재**: 런 env `LUMINA_CMFGEN_LINE_EPS_PHYS=1` 은 표 부재로 **전 선 불활성**이었다(무음 폴백) — 「이름을 측정 대신 쓰지 마라」 계급의 실사고. `eps_floor=1e-5`/`eps_cap` 은 mapped 선 경로의 floor — 조용히 기재(이 계약은 unmapped 원값만 쓴다).

## 집행 기록 3 — 구현 2차 착지 · 재감사(개정 2) · 운전자 코드 간단 감사 · 빌드 제출 (2026-09-17)

- **재감사(개정 2, Opus 4.8 fresh, 144k 토큰)**: `RE_AUDIT_REV2: 통과 · BLOCKING: 0` — R1 커밋 J = `cs.J`(`cmfgen_commit_jnu` 의 `source_J_nu`), `cmfgen_fine_jbar` 는 `csb->J` 읽기만, `cmf_lineres` 비활성(bare env 부재) · R2 `src_nlte=0`·`eps_phys=1`·표 미빌드·`emiss_b=NULL` 확증 · R3 두준위 접힘 대수 정합, mapped `el=1` 명시 고정 필요성 확인 · R4 census 「최대 pass」= 커밋 필드 조립 · R5 개정 2 가 구성상 참을 제거 · R6 전부 «기재로 족함». 자기 한계: 소스가 편집 중이었으므로 제출 전 재동결 요구 → 이행(`source.sha256` 183 파일, 빌드 잡이 전후 검증).
- **구현 2차(Opus 4.8, 385k 토큰)**: `lumina_cmfgen.c` +439/−24 · `lumina_plasma.c` +61 · `env_universe.h` +2/−1 · selftest 322 행. `diff -rq` 4 파일만. selftest 전 항목 PASS(fine 11 + 조립기 5).
- **운전자 코드 간단 감사**: Q0 봉인 무접촉(원본 `source.sha256` 전건 OK, 봉인 런 미접촉) · 게이트 OFF = legacy `eps_phys` 경로 문자 그대로(`if(!emiss_b && (eps_phys||gate_active))`, OFF 이면 표도 미빌드) · mapped `el=1.0` = base 실효값(표 미빌드 → 1.0 → cap 1.0) · unmapped 원값(floor/cap 미적용) · `el<0` → `g_asm_unmapped_eps_fault` → 드라이버 `[CMFGEN][BLOCKED] UNMAPPED_EPS_ASSEMBLE_FAILCLOSED` return −1(solve/commit 전) · fine 경로 `-2` → `eps_fail` → BLOCKED · census formatter 생산·테스트 공유 · selftest 훅은 `#ifdef` 전용. 기재: `lumina_line_used_nlte_fill_for_tau` 가 `build_lte_level_density_cache` 를 한 번 더 만든다(비용만, 결과 무관); 조립기 게이트 ON + `eps_phys=0` 조합은 이 런 밖. selftest 운전자 재실행 rc=0. **차단 없음 → 빌드 진행.**
- **빌드**: `$N/build_cpu.slurm`(CPU, `make -j1 cuda`, 소스 해시 전후 검증) — `check_driver_hygiene` PASS — Slurm **374550** 제출.

## 집행 기록 4 — 빌드 완료 · 스테이징 · 판정런 제출 (2026-09-17)

- **빌드 374550**(CPU, syntax 클러스터): rc=0, 소스 해시 전후 183/183 OK, `lumina_cuda` 4,800,480 B, sha256 `97d83fcd…678a3a`, sm_80/86/90 이미지.
- **seal-build**: `run_build_identity.json`(183 파일) — `CMFGEN_TWIN_RUN_BUILD_V1`.
- **스테이징** `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/run`: 343204 `input/` 복사(5.1 GB, 봉인 원본 읽기만) + 새 바이너리 + exports 경로 치환(7 참조) + `LUMINA_CMF_FINE_UNMAPPED_EPS="1"` 1 키 + `STAGE_IDENTITY.txt` 재생성(h200x1) + 동결 러너(작업명 `LUMINA_lteunmap_eps`, `--time=05:00:00`) + `OWNER.txt`. **G-ID 사전점검**(`identity_precheck.json`): 소스 차이 = 변경집합 4 파일뿐, exports 차이 = 게이트 1 키뿐. 덱 해시 OK, work 비어 있음.
- **운전자 이탈 2건(기재)**: 스테이징 스크립트 초판이 `grep -vc | grep -q` 파이프(rc-after-pipe)와 `sacct -M` 미핀으로 위생 게이트에 걸림 → 수정 후 PASS. grammar-debug 에서의 `sbatch` 는 다른 클러스터라 gres 거부 → syntax 로그인 노드에서 제출.
- **판정런 제출**: Slurm **374570** (syntax, h200, gres gpu:1, `sacct -M syntax` 로 소유 확인, WorkDir = repo). 상태 PENDING(Priority).
- **판정 도구**: `repro_claude_20260917/judge_lteunmap.py <run_root>` — base 런 자기검사 완료(G-EPS 미충족·E-1~4 미충족·LU-X).

## 집행 기록 5 — 개정 3(프로필 종속 키) · 병행 스테이징 (2026-09-17 15:10)

374570 이 h200 대기(앞 4건, Slurm 예상 시작 09-18 04:18)인데 a100 에 2 GPU 여유가 있어, 규약(빈 파티션 즉시·멀티파티션)대로 **같은 입력·같은 바이너리**로 둘째 런 루트 `…/lteunmap_eps_20260917/run_h100a100`(프로필 `h100-a100x2`, `MGPU_DEVICES=2`)을 스테이징해 h100,a100 에 병행 제출한다. **판정런은 1회**: 먼저 RUNNING 이 되는 쪽만 남기고 다른 쪽은 시작 전에 취소(`sacct -M syntax -o WorkDir` 로 소유 확인 후 job id 로 `scancel`). 개정 3 = G-ID 의 프로필 종속 키 허용(OLD/NEW 1쌍). 기재: 두 프로필의 fine 솔버 장치 수(1 대 2)가 물리 산출을 바꾸지 않는다는 것은 09-05 플랫폼 bit-일치(685/685)의 간접 근거뿐이며, 이 판정은 base 와의 물리 비교(E-1~6)에 의존하므로 bit-동일을 요구하지 않는다.

## 집행 기록 6 — 판정런 착수 (2026-09-17 15:15)

- **374587**(`run_h100a100`, 프로필 h100-a100x2) 이 15:15 RUNNING — syn08, `gpu_identity.csv`: NVIDIA H100 NVL ×2(95,830 MiB). **374570**(`run`, h200x1)은 PENDING 상태에서 운전자가 소유 확인(`sacct -M syntax -o JobName,WorkDir`: `LUMINA_lteunmap_eps` @ repo) 후 job id 로 취소 — 시작 전이라 GPU 지출 0. ⟹ 판정런 = 374587 단 1회.

## 집행 기록 7 — 판정런 374587 FAILED · 개정 4 (2026-09-17 15:50)

- **실측**: 374587 이 36분 만에 `MODEL_FAILED`(rc 1). 조립기 pass 1 census 50/50(게이트 작동, 셸 44 thermal_emit_fraction 3.4e-6)까지는 정상. 사망 지점 = fine **INIT 패스** exact 솔버 `[cmf_fine][EXACT-MULTIGPU-EPOCH] status=NOT_CONVERGED iterations=128 residual=0.0764 (tol 1e-8) scattering_ratio_bound=0.999999999996`(base 343204: 45회 수렴, bound 0.99999997683) → `[R6][BLOCKED] DETERMINISTIC_LINE_JBAR_PRODUCER iter=-1` → 커밋 0회. 원인 = 변경집합의 fine 경로 몫(unmapped 선의 (1−ε)χ 를 fine `chi_es` 로 접음 → 알베도 1−4e-12 → 고정점 반복 수렴 불가). 이 몫은 커밋 필드와 무관(INIT 선 J̄ 초기화)한데 런을 죽였다.
- **개정 4**(OLD/NEW 2쌍): fine 경로 ε 적용을 게이트에서 분리해 기본 비활성(별도 env, 미설정 = 원본 deposit) — 판정런은 조립기(커밋 경로) 몫만 켠다. 기재: 「구성상 무관한 경로에까지 같은 규칙을 넣은 것」이 운전자 기안 오류(개정 2 에서 「같은 물리 문장」이라며 유지).
- ⚠**bin 솔버 수렴 캐비앳**: `cmfgen_solve_J` 는 패스당 ALI 8회 고정(수렴 판정·진단 출력 없음, base 도 동일). 게이트 ON 이면 unmapped 선 빈의 알베도가 ≈1 이 되어 8회 ALI 가 커밋 필드를 미수렴으로 남길 수 있다. 판정런은 base 와 **같은 솔버·같은 반복수**로 돌리고(A/B 동등), 판정문에서 창 빈 J 의 패스 간 변화(pass 1→3 의 physics_DET iter0/iter1 J 차)로 수렴 여부를 사후 판독한다 — 미수렴이면 그렇게 기재하고 E-1 판정을 「관측 등급」으로 강등.
- GPU 지출: 374587 36분(2×H100) — 물리 산출 0. 후속 = 개정 4 구현(Opus 4.8) → selftest → 운전자 감사 → 재빌드 → 재스테이징 → 재제출.

## 집행 기록 8 — 개정 4 구현 착지 · 운전자 감사 · 빌드2 (2026-09-17 16:02)

- **구현(Opus 4.8)**: fine 경로 게이트 소스를 별도 env `LUMINA_CMF_FINE_UNMAPPED_EPS_FINE`(기본 0) 로 교체(`lumina_cmfgen.c` 91·5544 행), env 등록(COUNT 521), selftest +3 항목(커밋 게이트 단독 → fine 불변). `lumina_cmfgen.c` 총 +456/−24.
- **운전자 감사**: `diff -rq` 4 파일만 · selftest 재실행 21 PASS/0 FAIL · fine 경로 전 신규 코드가 fine env 뒤 · 조립기 게이트 불변 — 차단 없음. 운전자 이탈 기재: 감사 명령의 `gd` 인용 안 `grep 'FAIL|RESULT'` 의 `|` 가 파이프로 해석돼 배경 명령이 깨짐 → 중지 후 재실행(중복 빌드 제출 방지 가드 확인).
- **재동결** `source.sha256`/`source_before.json` 183 파일(16:02). **빌드2 374637** 제출(CPU). 이후 = `restage_run2.sh h100-a100x2` → 제출.

## 집행 기록 9 — 빌드2 완료 · 재스테이징 · 판정런 2 제출 (2026-09-17 16:0x)

- **빌드2 374637** rc 0, 소스 해시 183/183 전후 OK, 바이너리 sha256 `d5a2765b…65bf4`. `seal-build` → `run_build_identity.json`.
- **재스테이징** `run2`(프로필 h100-a100x2): 입력 = 343204 계보 복사, exports = 게이트 1 키(+프로필 종속 `MGPU_DEVICES=2`), `LUMINA_CMF_FINE_UNMAPPED_EPS_FINE` **미설정**(fine 경로 원본 그대로). G-ID 사전점검 OK(소스 4 파일), 덱 해시 OK.
- **판정런 2 = Slurm 374640** (syntax, h100,a100, gpu:2, 5 h 상한). 374587 은 FAILED 로 보존(증거: fine INIT 미수렴 로그).

## 집행 기록 10 — 판정런 2(374640) FAILED · 정지·보고 (2026-09-17 20:35)

- **경과**: syn08 H100×2, 4h23m. 조립기 census pass 1·2·3 각 50/50(외곽 41–49 최대 thermal_emit_fraction 4.3e-7 / 1.2e-7 — G-EPS 방향 충족). fine exact 솔버: INIT 48회 OK → iter 0 **121회** OK → iter 1 **128회 미수렴**(residual 3.8e-3) → `[R6][BLOCKED] DETERMINISTIC_LINE_JBAR_PRODUCER iter=1` → iter 1 커밋·`physics_DET_iter0001`·`lumina_ion_pops.csv`(ITER=1 덤프) **없음**. 살아남은 산출 = iter0000(시드 필드, base 와 동일)뿐 ⟹ **E-1~E-6 판정 불가**.
- **원인(소스 직독)**: `scattering_ratio_bound` 가 base 1−2×10⁻⁸ → 이 런 **1−3×10⁻¹¹**(세 패스 모두). fine 솔버의 「연속체」 `fs.chi_abs/fs.chi_es` 는 **bin 조립기 배열을 보간**(`lumina_cmfgen.c` ~5203 행)하는데, 조립기는 unmapped 선의 (1−ε)χ 를 `chi_es` 로 접으므로(2167 행) fine 필드에도 알베도≈1 산란 선숲이 들어간다 — 개정 4 가 막은 fine deposit 과 **다른 두 번째 유입로**. fine exact 솔버는 고정점 반복(cap 128)이라 알베도 1−10⁻¹¹ 에서 수렴 불가. iter 0 의 121회는 이미 경고였다.
- **처분**: 판정런 2회 연속 미착지(1차 36 min, 2차 4h23m×2 GPU) — 같은 부류(솔버 수렴)의 실패. 운전자는 **정지하고 user 에게 보고**한다. 다음 rung 후보는 계약 개정(user 결정): ① 조립기가 unmapped 산란 몫을 별도 배열에 두고 fine 보간은 base 와 같은 구성(선 산란 미포함)을 쓰게 함 + bin 솔버(ALI 8회 고정) 수렴 진단 1행 추가 — 커밋 필드에만 ε 물리, fine 은 base 동형 ② fine exact 솔버에 ALI/가속 도입(대형) ③ 계약 보류.
- **기재**: 물리적으로 ε~10⁻⁹ 의 선숲은 거의 보존 산란 매질이며(CMFGEN 은 전 이온 NLTE+ALI 로 처리), 현 fine 고정점 솔버는 그 영역을 풀 수 없다 — 이 한계는 램프 계약과 무관하게 「NLTE 밖 이온을 산란 취급」하는 어떤 수리에도 걸린다.

## 집행 기록 11 — 개정 5 (user 결정 「1번으로 진행해」, 2026-09-17 20:5x)

개정 5 = ① 조립기의 unmapped 산란 몫을 별도 배열로 분리 ② fine 연속체 보간에서 그 몫을 제외(base 구성 복원) ③ bin 솔버 패스별 수렴 진단 1 행 ④ 게이트 OFF byte-동일. OLD/NEW 3쌍. 후속 = 구현(Opus 4.8, 발주서 4) ‖ 개정 5 한정 사전감사(Opus 4.8 fresh) → 운전자 감사 → 빌드3 → run3 스테이징(h100-a100x2) → 제출.

## 집행 기록 12 — 개정 5 구현 착지 · 사전감사(개정 5) · 개정 6 · 빌드3 (2026-09-17 21:2x)

- **구현(Opus 4.8, 466k 토큰)**: 산란 몫 배열은 `CMFGENState` 멤버 대신 **파일-static** `g_chi_line_sc_unmapped[NS×NB]`(헤더 `lumina_cmfgen.h` 가 변경집합 밖 — G-ID 4 파일 우선; 조립기가 채우고 같은 패스의 fine 이 읽음) · fine 보간 감산(가드: 배열 실재·셀 수 일치) · CONV 진단(tri 분기 rel 재사용 + **non-tri 분기 rel 신설**, 별도 버퍼, `cs->J` 읽기만, 항상 발행) · selftest +5(산란 몫 항등·CONV 형식). `lumina_cmfgen.c` 총 +568/−26. `diff -rq` 4 파일.
- **운전자 감사**: 배열 크기 NS×NB·호출마다 memset·게이트 OFF 해제 · 보간 인덱스 `ia=s·n_bins+b` 정합 · `chi_es − sc = chi_e`(mapped 몫 0) · CONV 버퍼만 정렬 — 차단 없음. selftest 26 PASS/0 FAIL. 재동결 183 파일. **빌드3 375671** 제출.
- **사전감사(개정 5, Opus 4.8 fresh, 176k 토큰)**: `RE_AUDIT_REV5: 통과 불가 · BLOCKING 1(R4)`. R1 유입로 귀속 참(run2 fine bound 1−3e-11; fine census 0행으로 개정 4 비활성 확인) · R2 `chi_es − sc = chi_e` 대수 성립, fine 필드는 base 와 구조 동일(차이 = 이온화 상태) · R3 E-1~E-4 미오염, fine 필드는 램프-무결 아님(mapped 선 J̄ 2차 잔여 — 판정문 기재) · R5 구성상 참 없음·OFF byte-동일 · R6 세 번째 수렴 유입로 없음(전수표). **R4(a)** 「rel 재사용」 불가(non-tri 분기에 rel 없음) → 구현자가 이미 non-tri rel 을 신설해 해소. **R4(b)** 문턱 0.1 = 수렴 확증은 원리적 오류 → **개정 6**(OLD/NEW 1쌍): CONV 는 하한, 이 런의 E-1·E-2 는 «관측 등급» 기본, 분기 라벨 (관측). «기재로 족함»: `(chi_e+Δ)−Δ` 상쇄로 fine chi_es 는 base 와 bit-동일이 아니라 허용오차 동일(알베도↓ 방향, 무해).
- 잔여: 빌드3 완료 → run3 스테이징(h100-a100x2) → 제출.

## 집행 기록 13 — 빌드3 완료 · run3 스테이징 · 판정런 3 제출 (2026-09-17 21:3x)

- **빌드3 375671** rc 0, 해시 183/183, 바이너리 `4e385989…c7ea8`. `seal-build` → `run3/input/run_build_identity.json`.
- **run3**(프로필 h100-a100x2): 343204 계보 입력, exports = 게이트 1 키(+`MGPU_DEVICES=2`), fine env 미설정. G-ID 사전점검 OK(소스 4 파일), 덱 해시 OK.
- **판정런 3 = Slurm 375676** (syntax, h100,a100, gpu:2, 5 h). 개정 6 에 따라 이 런의 E-1·E-2 는 «관측 등급». run2 는 FAILED 로 보존(fine 알베도 1−3e-11 증거).
