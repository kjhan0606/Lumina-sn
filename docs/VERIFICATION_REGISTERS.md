# 3대 검증 대장 (VERIFICATION REGISTERS) — 2026-07-29 제정 / **07-29 15:30 갱신(1회차 회전 리뷰 반영)**
**용도: 회전 리뷰.** 라운드마다 임의 대장의 임의 항목이 재도전받고, 실패 시 강등한다.
**전이 규칙**: 신규 주장 → B. B→C = 독립 검증 1회 CONFIRMED(V3급). C→A = 독립 경로 ≥2 + 재현 산출물 + 회전 리뷰 생존. 실패 = 즉시 강등 + 원장 `[정정]`.

## A. 완전 검증 완료 (독립 경로 ≥2, 재현 가능) — 12건
| 항목 | 검증 경로 | 재현 |
|---|---|---|
| verdict_gate.py | KA 3종 + 합성 픽스처 8종 | scripts/verdict_gate.py |
| Saha 하니스 + RATES_FIX F1/F2/F3 | A1 md5 4/4 + A2 196/196(max dev 1.594e-12, OFF 134/196 재확증) + **A3 산출물화 완료(07-29 밤: OFF 4/9→ON 0/9, logs/ioniz_saha_selftest_A3/ — Y 리팩터 후 신빌드로 재현 = 회귀검증 겸)** + 리뷰어 독립 파이썬 재현(196행 ≤4.1e-5) **[2회차 정정 반영: ON arm 재현은 `LUMINA_RATES_FIX=1 bash scripts/run_ioniz_saha_selftest.sh`; 게이트는 F1F2F3**F5** 일괄(F5 미검증=B2)]** | make selftest_ioniz_saha (OFF) + RATES_FIX=1 (ON) + LOWT_PROBE |
| 율 기계 PRRR 인증(4이온) | 밴드 PASS + 계층분해 + 재-bake 비트재현 | validation/.../rates_certification/ |
| Γ(Co III) 34-57× | V1-V5(PRRR 앵커=CMFGEN 자기인증) | .../gamma_coiii_alllevel/ |
| POPCOB 규약 | 소스 + 왕복 1.00000000 | 동상 |
| GPH 3게이트 회귀 귀속 | 양방향 A/B + V1 + V3 + 45런 타임라인 | verdict_gate + 원장 |
| parity42 판정 | V0-V5 1호(10C/6R) | .../parity42/VERDICT_*.md |
| S IV/V·Ca IV/V 임포트 | 데이터 자체검증(준위 194/200/378/200·경계앵커 972/18790/967 정확폐합·왕복항등 ≤3.3e-16·χ NIST<0.1%[Ca IV −0.26%=CMFGEN 빈티지]) **[2회차 리뷰 생존-정정: "D1-D6" 라벨 실체 없음→본 실측으로 대체; ⚠기존이온 line_id 전면 재번호(디렉토리 경계 넘는 line_id 조인 무효); 임포트=사다리/MA/σ_bf까지, NLTE 행렬 승격 아님(census 21038 불변)]** | 리뷰 2회차 산출물 07-29 |
| 파이프라인 결정론 | 호스트·OMP 교차 0행 + 반복 0행 | 원장 |
| **[신규] FORMAL_FIX 수리(R1-R3)** | 수용시험(영점 1.000000028) + OFF 실증(OMP 스윕 md5) + 스모크 등록 3/3 적중(r3_unresolved=0) | withParityV·스모크 로그 |
| **[신규] 실Υ 40이온 추출** | 기존 bin 5종 bit-exact 재현 + 왕복 + 배너 수 앵커 + CMFGEN 리더 축자 이식 + **R1 끝-끝 사슬(원본 ASCII→bin→보간→런타임 출력, 3앵커·손산술 6/6)** | scripts/build_cmfgen_coldata_all.py + parity43 R1 |
| **[신규] 클램프 무죄 8건** | 발화 0 재확인 + **카운터 실배선 소스 확인**(적대 재검증) | clamp_census/{INNOCENT,ADVERSARIAL}.md |

## B. 검증 대상 — 정합성(코드 옳음)만. **국면 규칙(user 2026-07-29): fine tuning 관련 신규 항목은 B에 넣지 않고 D로 이월** — 10건
0a. **▲EPAY-REPLAY-001 — LCMFLP01-v1 EPAY 정규화 재생 불능(계기 부채, 08-02 등재):** payload는 per-cell disposition은 보존하나 rate-shape/scalar EPAY의 셸별 정규화 `wn`과 그 입력 `acc_abs`·`acc_dep`·`acc_w`를 직렬화하지 않는다. `acc_abs`가 assemble 당시 lagged J로 계산되고 dump 전 solve/damping 뒤 J와 다르므로, 같은 세대 payload만으로 population-native opacity 반사실의 EPAY scale을 exact 재조립할 수 없다. 근거=`n9_summary.json: epay_scale_not_reproducible=true`, disposition `5000/10696/34304/0`, clamp·fallback·nonfinite `0/0/0`. 재현=`/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/uv_t2n9_offline/n9_summary.json`. 처분=scale과 세 누산량의 per-shell 직렬화 필요(수리안 별도).
0b. **▲LINEPOP-POPAUTH-001 — NLTE 이온 창 밖 선의 population 권위 결손(08-02 등재, 08-02 정정):** ⚠**초판 이온 라벨 오독 정정** — 초판이 "Fe III 5,924행"으로 기재했으나 실체는 **Fe IV**다. 포렌식 JSON의 `ion` 필드는 **0-기반 `ion_number` 원값**(0=I)이며 writer가 `atom->line_ion_number[l]`을 그대로 기록한다(`lumina_cmfgen.c:847`). 근거 4중: `NLTE_TARGET_ION[]`에 값 0 존재(`lumina_plasma.c:7677-7684`의 O I/II/III 항), 피영향 집합의 ion 값이 {3,0,4}뿐이고 **1·2는 0건**(=창 {1,2}의 정확한 여집합), 코드 주석이 `ion_number 3 = spectroscopic IV` 명시(`lumina_plasma.c:14231`), Fe III 진단은 `(26,2)` 조회(`lumina_cuda.cu:2007`). Codex·fable 독립 조사 2건이 서로 다른 증거로 동일 결론(운전석이 틀린 전제로 발주했음에도 양쪽이 뒤집음). **실체**: 선택 1,169,145행 중 **28,949행**이 population-native 요구량 미정의(전량 `-1` sentinel, `raw_zero=0`, **`actual_solver_negative_population_rows=0`** ⟹ 양수성 계약 무죄). 매핑 술어는 순수 (Z, ion_number) 소속 판정(`nlte_build_projection`, `lumina_plasma.c:14161-14170`), 기본 창=14원소 II·III + O I/II/III이므로 미매핑 = **중성 + IV단 + V단**의 정확한 여집합. 미매핑 선은 bulk/nebular Sobolev τ(`lumina_plasma.c:2582-2681`)를 유지한 채 `line_source_S` 미기록 → formal 솔버에서 **S_l = B(T_e) 폴백**(`lumina_cmfgen.c:787-789`). **규모(정정)**: "46.7%"는 **피영향 집합 내부**의 몫이고 전체 BALL χ_line 대비 Fe IV 절대 몫은 **s0 0.664%·s8 0.160%**. 주목 대상은 Fe IV가 아니라 **Co IV 5,571행** — 확정 인과 사슬의 주범 이온이 통째로 창 밖이며 전역 최대 τ=4.34e6 @606.8Å(바닥준위 공명선, s0)이 여기 속한다. τ>1 피영향 셀 564건={Fe IV 270, Co IV 182, S IV 42, Ni IV 41, Si IV 16, Ca IV 6, S V 4, Ca V 3}. **잣대 결함 동반**: 상설 `[TAU-BY-ION]` 감시 기본 파장창이 **[1700,10000]Å**(`lumina_plasma.c:6440`)이라 600–1700Å 두꺼운 꼬리를 구조적으로 못 본다 — stdout "τ_sum≈0, N(τ>1)=0"과 덤프 실측 불일치의 원인. **부수 항등식 주의**: `outside_BALL_rows=0`은 발견이 아니라 BALL-only writer 술어의 항등식이다. **미결**: 스펙트럼 영향 크기 UNRESOLVED. 갈림 계측=`LUMINA_NLTE_STAGE4=1`(Fe/Co/Ni/Si IV 승격, 38-슬롯 테이블 `lumina_plasma.c:7693-7700`에 이미 존재) A/B. Fe III level-17 metastable trap과는 **별개**(피영향에 Fe III 0건; 간접 FUV 결합 가능성 미계측). 처분=커버리지 결손 등재(검증불가 고아 금지 규약: 매핑 확장으로 해소), 수리안 별도 과업.
0c. **▲CLAIM-INFLATION-001 — UV 캠페인 결론 11건 과장 판정(08-03 등재):** 독립 해석 감사(`docs/CODEX_CLAIM_AUDIT.md`)가 23개 주장을 **INFLATED 11 / SUPPORTED 9 / UNVERIFIABLE 3**으로 분류했다. 결정 영향 큰 순: ①"T1이 재분배 노선 전체를 닫았다·ALI가 유일 본진" — T5 자신은 `route_disposition=UNRESOLVED`, 반증된 것은 shell-8 균일 사전주입 대리모형 하나 ②"E8이 UV 11.977×를 완전히 설명" — 두 5247배가 **같은 `S_fixed` 분모**를 공유하고 필요이득에서 CMFGEN 항이 소거되는 **동일 시점 대수 항등식**(운전석이 독립 확인: `11.9770975 × 1.90567286e-4 = 2.28245e-3` ≈ 보고값) ③**Stage 3.1 "MC 수송 무죄"가 순환** — `J_MC`가 live estimator가 아니라 sidecar의 결정론 `J_producer`이고 비교 대상 `eta_total`에 `chi_es·J_producer`가 이미 포함(`stage31_cmf_field_bench.py:613`) ⟹ **MC 수송은 아직 무죄 아님**(원장 사례 7-8 생산자 왕복검증 재발) ④N9 "99.56% η_line 폐기" — 분모가 post-EPAY `eta_fixed`이고 B1–B4의 `1.0`은 mask 항등식 ⑤T4 "누락채널 0.24% < 1%" — 분모가 **비무작위 4억 prefix**(저장률 41.2%, `TRUNCATED_PREFIX-not-an-unbiased-random-sample`) ⑥T3 T_rad 기각 — pumping proxy 민감도만 측정, UV field 반사실 미측정. **처분**: 정본 `docs/UV_CENSUS_CONSOLIDATION.md`에 §0 정정 고지 + 각 지점 인라인 표지 기재 완료(08-03). 본문은 역사 보존을 위해 원문 유지. **근원 진단**: 산출물은 정직했고 **요약층이 단서를 벗겨냈다**(`"ion":3`→"Fe III", `UNRESOLVED`→"완전히 설명", `eta_fixed_post_EPAY`→"조립된 η"). 신설 규칙=헤드라인 수치는 **산출물 자신의 정의 문자열과 함께** 인용.

0d. **▲S32R1-EPSILON-001 — rung 1의 ε·λ*가 실제 반복 연산자와 짝이 안 맞음(08-03 등재, 유도 판정 완료):** v2/v3 패치의 1차 측정량 `rho=(1−eps0)(1−beta_Sobolev)`는 **코드가 수행하는 반복의 spectral radius가 아니다.** ①production의 `eps'=C/(C+A·beta)`(`lumina_plasma.c:8471`)는 Sobolev 사다리 합산의 닫힌 형태 `P_dest = 1 − Aβ/(C+Aβ)`이며 **옳다**; 패치의 `eps0=C/(C+A)`가 이 연산자에 대해 틀린 ε다(팽창 불투명도 `w=τβ·ν/(ctΔν)`가 이미 β를 품어 bin 상호작용은 사다리를 마친 사건). ②더 크게, 반복 대상은 Sobolev 2준위가 아니라 `S=S_fixed+(chi_es/chi_tot)·Λ_formal[S]`(`lumina_cmfgen.c:1641`)이고 그 대각은 production이 `cs->lambda_star`에 이미 보유(`:2147`,`:2179`) ⟹ **`rho_local=(chi_es/chi_tot)·lambda_star`**. **파급**: 구 사전등록의 `[0.99,1.0)` 구간은 `eps0` 정의에서만 도달 가능하고 `eps'` 정의에선 구조적 도달 불가 — **잣대가 답을 미리 정한 형태**. **처분**: 구 사전등록(`stage32_rung1_expected_changes.txt`, sha `e3c5c186…`) **은퇴**(수정 금지, 계보 보존), 정본=`_v2.txt`. 은퇴는 **production 측정 이전**에 잣대가 다른 연산자를 잰다는 것이 드러나서이지 결과를 본 뒤 골대 이동이 아니다. 전문=`docs/AUDIT_STAGE32_RUNG1_EPSILON_DISCREPANCY.md`.

0e. **▲S32R1-PATCH-CHAIN — v1→v4 수리 계보(08-03):** v1 REJECT(F1 disposition 재구성이 `acc_w>0` 누락·F2 불투명도 몫 배분·F3 클램프성 가드 3건·F4 세대 규율) → v2 REJECT(**운전석 감사 F5**: 계측 누적이 `w*Sl`인데 production은 `eta_l=w*el*Sl`, 게다가 fixture 스텁이 `-1.0`→`el=1.0`으로 **불일치를 구조적으로 은폐** — v1이 이미 규탄한 은폐 구조의 자리 이동. **근원=발주서 문안 오류**, `CHARTER_STAGE32_RUNG1_REPAIR.md` §2에 정정 기재) → v3(F5 수리: production `eta_l` **재사용**, 음성 대조 잔차 ±6.45e-07) → v4(측정 대상 교체 + ε 4열 `beta/eps0_raw/eps_prime/eps_applied` + 세대 정합). **v4 감사 통과**(운전석): `lambda_generation`은 복제가 아니라 `cmfgen_solve_J` 말미의 **인증 스탬프**라 조립-단독·대체솔버·조기반환을 전부 FAIL시킴; `chi_tot=0`은 대체값 없이 NaN+상태코드; 신규 clamp 0건; 음성 대조 3건 실물 FAIL. sha: v3 `b41e991c…`, v4 `7cba8539…`. **판정런 189065**(h200/syn104, bin `lumina_cuda.withParityAI` sha `aeb2a3cc…`) 진행 중. **부수 사고 2건**: Codex가 격리 밖 cwd에서 CPU 빌드를 실행해 untracked `lumina` 바이너리를 **2회** 덮어씀(v2·v4; 소스는 매번 byte-identical 확인). 처분=차기 발주부터 운전석이 격리 경로를 지정해 넘김.

0f. **▲UV-MAPSPLIT-001 — UV 선방출의 95.3%가 NLTE 매핑 선(08-03 등재, 측정 완료):** 포착 5셸(0/8/16/20/45) BALL 600–3000Å에서 매핑 선 방출 에너지 몫 **0.95317**, 미매핑 **0.04683**. 미매핑 **불투명도** 몫은 1.14%뿐 ⟹ 미매핑 선은 단위 불투명도당 ~4배 방출. 미매핑 방출체는 사실상 셋: **Co IV 45.2% · Fe IV 33.7% · Ni IV 19.6%(합 98.5%)** — 0b가 지목한 IV단 이온들. 에너지의 98%가 s0. **잣대 사고 1건(운전석 적발)**: 초판 스크립트가 production 조립식의 `eps_l` 인자를 누락 — 수정 시 답이 **15배** 변동(미매핑 0.31%→4.68%). production은 `eps_phys=1`이므로 **4.68%가 참값**. 자기시험 crash·정의 문자열 미갱신 2건도 수리 후 **주입 결함 2종 FAIL 시연**으로 게이트화. **판정**: ALI가 매핑 집합에만 작용해도 UV 선방출의 95.3%에 도달 ⟹ **rung 2 보류 해제**. 남는 4.68%는 population 권위가 없어 ALI 불가촉이나 UV 초과 11.98× 대비 진폭 주범 불가. **⚠정정 (08-03, Codex 반증 + 운전석 실측 확인)**: 초판이 "미매핑 선은 `S=B(T_e)` 고정"이라 적어 매핑/미매핑이 소스 종류로 갈리는 것처럼 읽히게 했으나 **틀렸다**. 이 캡처는 payload 헤더 `src_nlte=0`이고 조립식이 `Sl = src_nlte ? Sl_pop : 0.0`(`lumina_cmfgen.c:787`)이므로 **매핑 여부와 무관하게 전 선이 `B(T_e)`를 쓴다**. 행 실측: 매핑 1,140,196행 전부 `S_l_pop>0`으로 기록은 되어 있으나 **`S_l_used==S_l_pop`인 행은 0**(단 한 행도 population 소스를 소비하지 않음), 미매핑 28,949행은 `S_l_pop>0`이 0건. ⟹ 커버리지 결손은 실재하되 `SRC_NLTE=0` 아래서 **잠재**이며 이 런의 소스에 영향 없음. "미매핑이 단위 불투명도당 4배 방출"도 B 폴백 기전으로 설명되지 않는다(양쪽 다 B) — 잔차 원인은 `eps_l`·`B(nu)`·`dnu` 쪽이며 **미규명**. 매핑분해의 95.3%/4.68% 수치 자체는 유효(에너지 몫 측정이지 소스 종류 판정이 아님). 산출물=`/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/uv_mapsplit/`.

0g. **▲▲TWIN-FE-CONTRADICTION — "twin reproduces Fe fully"이 같은 twin의 최종 대장과 모순(08-03 등재):** `criminal_record/CRIMINAL_RECORD.md:63`(C10)이 *"twin reproduces Fe fully but Co still low ⟹ Co rate-side … survives correct field"*로 기재했으나, **같은 twin의 `co3_closure_trace/final_ledger.csv`가 이를 반박한다**. `deficit_cmf/twin` 실측(운전석 직접 재계산): s2 Fe **1.377** / s6 Fe **0.715** / **s8 Fe 4.249** / s8 Co 17.255. **Fe도 s8에서 4.25× 틀린다.** Co 잔차가 더 크다는 것은 사실이나 "Fe 완전재현"은 성립하지 않으며, C10의 추론이 **깨끗한 Fe 음성대조를 전제**하는데 그 음성대조가 존재하지 않는다. ⟹ "Co만의 10× 결손"과 "Co rate-side 확정"은 근거가 무너진다. **등급 주의**: 이것은 과장(0c 계열)이 아니라 **자기 산출물 간 불일치**다 — 요약이 아티팩트를 넘은 게 아니라 두 아티팩트가 서로 다른 말을 하고 있고 상위 문서가 한쪽만 인용했다. **부수 정정**: 그 twin의 핀 범위도 "T+J 전면 핀"이 아니라 **T_e 전 셸 whole-state pin + CMFGEN J-table을 Gph 광이온 적분에서만 대체**다(`scripts/sbatch_te_jtable.sh:11`, `lumina_plasma.c:9760`; 런 로그 50셸 pin·Gph table 소비 214,208,908회). 원 판정문(`rates_certification/VERDICT.md:6`)은 이미 조건부이고 Lumina J estimator·자체 population·recombination/Milne·fallback 경로를 **명시적으로 인증 범위 밖**에 둔다(`:246`). **"율 무죄 최종"이라는 문구는 원 산출물 어디에도 없다** — 운전석 메모리 인덱스의 요약이었고 08-03 정정함. **처분**: C10 결론 보류, Fe 음성대조 재수립 필요. 고정점 시험이 못 잡는 것=`dR/dJ` 오류·loop 증폭·`J=J*`에서 소거되는 오류.

0h. **▲CLAMP-C08b — `radeq_line_eps_phys`의 `denom<=0 → 1.0` 미등록 대체(08-03 등재):** `src/lumina_plasma.c:8472` `return (denom > 0.0) ? C_ul / denom : 1.0;`. 정의되지 않은 `0/(0+0)`에 **ε=1(완전 열화)** 이라는 최대로 결과를 바꾸는 값을 대입한다. 클램프 census의 C08(`eps_floor=1e-5`, 미등록 ε→1)은 **다른 분기**(미등록 line → `-1` → caller가 1)를 적고 있으므로 이 항목은 **별도 미등록**이다(Codex 독립 확인). 판별식 적용: 정확해가 위반 가능한 가드인가 → `C_ul=A_ul=0`은 정의되지 않은 것이지 `ε=1`이 아니다 ⟹ **대체(substitution) 확정**. **동반 계기 결손**: 기존 산출물에는 raw ε와 clamp reason이 없고 post-clamp `el`만 있어, 경계값과 같은 행을 세어도 "원래부터 1e-5/1"인지 "floor/cap 발화"인지 **구별 불가**. 사유별 카운터가 필요하다. rung 1 v4가 `eps_applied_diff`를 처음 도입했으나 E8 epoch의 발화 횟수는 소급 불가 ⟹ **UNRESOLVED**. **부수**: 대역 평균 `ε_eff=1.90567e-4 > floor 1e-5`는 개별 선 미발화를 **함의하지 않는다**(일부가 floor에 붙어도 나머지+continuum 가중합이 floor를 넘길 수 있다).

0i. **○DISSENT-ADJUDICATION-001 — 운전석 이의 3건의 독립 재판(08-03):** 운전석이 일주일치 결론 중 3건에 이의를 제기했고, **Codex(3건 전부)와 fable(1번, 상호 열람 금지)**에 "반증하라"로 병렬 발주했다. 결과: **1번 기전(커버리지 과방출) = 반증**(독립 2경로 — Codex: 이 캡처는 `src_nlte=0`이라 매핑 무관하게 전 선이 `B(T_e)` / fable: `noeps` 열 분해 시 소스항 자체는 단위 불투명도당 **0.24배 과소**이고 15× 점프는 전부 `eps_l`이 나르며 이는 두꺼운 IV단 공명선의 정상 물리; 게다가 s0 욕은 **초열적**(u 400.2 vs aT_e⁴ 224)이라 폴백은 오히려 **과소방출** — 운전석 기전의 **부호가 반대**). **1번 "확정이 과하다" = 양쪽 다 지지**(원 아티팩트는 "top suspect + 미실행 판별계측"이며 `reddening_localization/VERDICT.md:134-151`이 A/B 3종을 "design only — no runs"로 남김). **2번 = 지지**(0h 참조). **3번 = 대체로 오발**(운전석이 자기 메모리 요약을 원문인 양 공격) — 단 그 과정에서 **0g를 노출**. **fable 고유 발견(양쪽 중 유일)**: ①커버리지는 깔때기의 **원인으로 구조적 배제** — 깔때기 증거가 전부 MC 채널인데 MC 선수송은 `line_source_S`를 **소비하지 않는다**(운전석 코드 확인: `d_line_scatter_event` 시그니처에 S_l 없음; `nlte_line_map`의 cuda.cu 소비처는 `:1860` 호스트 τ/S writer **단 1곳**; macro-atom은 창과 무관하게 2,565,342선 전부 커버). 심부 방출 평균 λ=1553Å로 B(13120) 색 정합 몫 **<2%**(비열적), 진폭도 4.68%가 u의 42%를 감당 불가. ⟹ **사슬의 비어 있는 대안은 커버리지가 아니라 Co 이온화 균형**(reddening probe 1, Co IV→III 재결합 게이트, 미실행). ②**신규 이상 신호**: s0 B3(2000-2500Å)·B4(2500-3000Å)에서 미매핑이 χ의 ~0.01%만 쥐고 ε-가중 열채널 에너지의 **63%·14%**를 나름 — 귀속 미정. ③**잣대 오염 경로 확인**: formal observer march에도 `B(T_e)` 폴백이 있다(`lumina_cmfgen.c:3134-3135`). **교훈**: 운전석 이의는 기전이 틀렸으나 그 재판이 0g·0h·신규 이상 신호 1건을 산출했다. **에이전트를 "반증하라"로 걸어야 값이 나온다**(수렴시키면 메아리).

0j. **▲▲▲CHAIN-EPOCH-STALE — 확정 인과 사슬은 19일 전·바이너리 13세대 이전 상태의 진술이고, 종점값의 부호가 현행에서 역전(08-03 등재):** 사슬 3대 산출물(`trapping_audit`·`radeq_ledger_audit`·`reddening_localization`)과 `mastermind_test`가 **전부 같은 런 하나**에서 나왔다 — `logs/coevolve_consume_a10_kx_gphall`, **2026-07-15**(분석문서 07-19~20). `radeq_ledger.py:14`에 `Te=13119.874754`가 **하드코딩**돼 있다. 이후 배포 바이너리 13세대(W/X/Y→AA…AJ)가 지났고 최근 전 런이 **parity59**(`docs/PARITY59_INSTR.env`)다: chieta_188605→emiss_ab_188747→emiss_ab2_188766→fluormat_188828→instr_188901→instr_188932→s32_rung1_189065/189116. ⟹ **현행 기준선=parity59 확정**(운전석 계보 판정). **실측 역전**: s0 T_e가 07-15 런 **13,120 K**(진리 18,760 대비 −5,640 부족) vs 캡처 188932 **21,228 K**(진리 대비 **+2,468 초과**). 캡처 값은 진짜 해다 — `[TEHOLD] s0` 11183→…→21226→21228 단조수렴, **600/600 `radeq_root=root-found`, no-root·클램프 발화 0건**; T_e 게이트도 양쪽 동일(`RADEQ_TE=1 RADEQ_SIMUL=1 DAMP=0.5 FB_RATE=1 VR_STD=1`)이라 "두 T_e 방법" 설명 **기각**. 장 전체도 역전: 진리 열욕 `aT⁴(18760)=937` 대비 캡처 **u_mc=1.87× · u_cs=2.86× 과충전**인데 07-15는 **0.43× 기근**이었다. **파급**: 사슬의 출발점("EUV/FUV 기근")과 종점("zero-pump 냉각 root")이 둘 다 현행에서 성립하지 않는다. 사슬이 틀렸다기보다 **더 이상 현 상태를 서술하지 않는다**. **★재생 결과 (08-03, `validation/chain_replay_parity59/comparison_summary.csv`)**: 3대 분석을 **파라미터화 사본으로 양쪽 런 모두 재실행**(같은 코드로 07-15 baseline까지 재생 → 코드 변경이 차이를 만들 가능성 제거). 사슬의 **중심 기전이 소멸이 아니라 역전**됐다.

| 양 | 07-15 | 캡처 188932 | 판정 |
|---|---|---|---|
| s0 T_e | 13,119.87 K | 21,227.64 K | 역전 |
| 진리(18,760) 대비 | −5,640 K | **+2,468 K** | 역전 |
| s0 u_mc/CMFGEN | 0.576 | **2.518** | 역전 |
| **FUV 918–1290Å u_mc/CMFGEN** | **0.0230** | **2.526** | **역전** |
| **EUV 450–918Å u_mc/CMFGEN** | **0.0144** | **5.132** | **역전** |
| τ_es/CMFGEN | 1.182 | **1.020** | 유지·개선 |
| 1508Å 빈 u 몫 | 0.4161 | 0.1586 | 변화 |
| 1526Å mc/cs | 39.04× | **3.93×** | 변화 |

⟹ ①사슬의 심장인 **"EUV/FUV 기근"**이 43배·69배 **결손**에서 2.5배·5.1배 **초과**로 돌아섰다. "기근→bf가열·선펌핑 고사→냉각 root" 인과가 현행에서 전제부터 반대 부호다. ②`reddening_localization`이 진범 현장으로 지목한 두 정량 서명(1526Å mc/cs 39×, 1508Å 집중 41.6%)이 **3.93×·15.9%로 붕괴** — 10배·2.6배 완화. ③τ_es는 오히려 CMFGEN과 2% 이내로 정합(07-15엔 18% 초과). **캡처 쪽 UNRESOLVED 3건의 처분(08-03, 운전석 추적 완료)**:

- **τ_FUV·Rosseland line depth → `UNVERIFIABLE`(노후 아님, 재현 불능).** 근거: `lumina_line.csv`가 **어디에도 존재한 적이 없다** — 저장소 전체에서 이 파일을 **읽는 코드 2개**(`trapping_audit/audit_t_expop.py`와 그 재생 사본), **쓰는 코드 0개**; `logs/`·`/gpfs` scratch 어디에도 실물 없음; **07-15 런 디렉터리(파일 11개 전수)에도 없음**. ⟹ `τ_FUV≈70`을 냈다는 스크립트가 **07-15 당시에도 이 입력 없이는 돌 수 없었다**(미실행이었거나 소실된 임시입력으로 1회 실행 후 산출물만 잔존). 사슬의 *"FUV(τ70) 냉열화"* 고리는 **07-15 상태에서조차 재현 경로가 없다.** 발주 불가 — 발주해도 산출 불가.
- **심부 방출 평균 λ·Co IV 비중 → 해소됨.** 운전석이 `scripts/pile_ion_attribution.py`로 **이벤트 8억건 전량**(활성화 199,491,999 + 종단 200,010,004) 통과해 이미 측정했다(0f 계열 산출물 `pile_ion_attribution/`). prefix 4분위 편향 선행감사 통과(몫 이동 최대 4.9%p, 대부분 <2%p, `status=MEASURED-NO-PREDECLARED-STABILITY-THRESHOLD`). 1290–2000Å 심부(s0–2): **방출선 이온축 Co IV 21.0%**, **활성화선 이온축 Co IV 0.37%**, 활성화 지배 이온은 **Co III**(심부 B1 72.1%·B2 65.8%·B4 80.7%). 사슬 재생이 이를 UNRESOLVED로 둔 것은 운전석이 "8GB 전량 통과 금지"를 발주 제약으로 걸었기 때문이며 실측은 존재한다.
- **parity59 식 coupled root·5,640K lever 분해 → 미착수(발주 대기).** 현 Codex 조사 종료 후 착수(user 승인 08-03; 동시수정 회피 — 조사 1이 같은 `radeq_lines` 테이블을 읽는 중).

**처분**: ①정본 인용 시 **"parity59 이전 상태"** 단서 강제 ②재생 완료 — 사슬 기반 신규 추론은 위 대조표를 근거로만 ③**신규 조사 표적(user 승인 08-03)**: *07-15와 parity59 사이 바이너리 13세대 중 무엇이 기근을 고쳤고 과잉까지 밀어냈는가.* 판정런 착지 후 착수. ④오늘 밤 운전석·Codex·fable의 "Co IV 형광 깔때기 진범" 논쟁은 **이미 10배 완화된 현장** 위에서 이뤄졌음을 명기. **교훈**: 오늘 밤 운전석·Codex·fable 셋이 이 사슬을 **현재형으로 놓고** 논쟁했다. 산출물에 epoch 단서가 없으면 19일 지난 숫자가 현재로 읽힌다.

0k. **▲▲FALLBACK-OVEREMISSION — fable 자기 반증 2건 번복, 새 후보 기전(08-03 등재):** 0j의 온도 정정으로 fable이 이전 감사(0i)의 열역학 논거 **2개를 스스로 번복**했다. **①"폴백 과소방출" → 번복**: 옳은 비교는 `B(T_e)` vs 참 `S_l`이고 심부 진리는 `S≈B(18760)`인데 캡처 T_e=21,228 > 18,760이므로 `B(21228)/B(18760)` = **912Å 2.66× · 1200Å 2.11× · 1553Å 1.79× · 5000Å 1.26×** ⟹ **전 파장 과방출, EUV/FUV에서 최대**. **②색 논거 → 사망**: *"심부 방출 평균 1553Å이 B(13120) 색 정합 몫 <2% ⟹ 비열적"*이었으나 B(21228)은 **B_λ 피크 1365Å**, 에너지평균 1766Å, `frac(<1553Å)=34%`(13120K에선 1.8-2.8%). **진리 18,760K조차 B_λ 피크=1545Å≈1553Å** ⟹ 1553Å은 비열 서명이 아니라 그 온도의 **열적 색 한복판**. "파일은 비열적"은 틀린 온도의 아티팩트였다. **③채널 분리 → 유지**(온도 무관): `d_line_scatter_event`(`lumina_cuda.cu:5303-5325`)에 S_l 인자 없음, `nlte_line_map` 소비처 `:1860` 단 1곳 ⟹ MC는 폴백 소스를 소비 불가, **과방출은 결정론 채널로 작동**(실제로 `u_cs`가 가장 부푼 욕). **운전석 기전의 처분(정정)**: **방향은 되살아났고 귀속은 여전히 기각**이다 — `src_nlte=0`이라 매핑·미매핑 무관하게 **전 선이 `B(T_e)`**를 쓰므로 이는 *커버리지* 결함이 아니라 **결정론 레인 전체의 선원함수 결함**이다. **새 후보 기전**: 과방출→욕 가열→radeq 근 상승→`B(T_e)` 증가→과방출의 **양성 되먹임**. `[TEHOLD]` 11,183→21,228 단조상승과 정합(정합 관찰이지 증명 아님). **과열이 s0 국소**(s1=18,425≈진리−335K, s2=16,458)이고 **미매핑 에너지 98%가 앉은 셸과 일치**. **결판 요건(fable 제시, 둘 중 하나)**: `eta_line`을 매핑/폴백 분리 집계한 덤프 1회, 또는 폴백을 `S_l=B(진리 T)`로 핀한 대조런 1회. 전자가 저렴. **처분**: 사슬 재생(0j) 결과를 먼저 보고 중복 계측 회피 후 착수(운전석 판단, user 승인).

0l. **★★★S32-RUNG1-VERDICT — 반복 연산자 spectral radius 최초 실측; E8의 5247은 반복 증폭이 아님(08-03 확정):** 판정런 **189116**(h200/syn104, bin `lumina_cuda.withParityAJ` sha `aeb2a3cc…`→수리본 `e610345a…`, 2h19m) 착지. 페이로드 `stage32_rung1_iter10.iter010` 4.51GB, sha `e2a81ade…`, **37,586,850행**. 사전등록 정본=`patches/stage32_rung1_expected_changes_v2.txt`.

**계기 건전성 (전부 통과)**: `closure_residual=2.69e-19`(authoritative 9.8458e-4 = selected 9.8432e-4 + boundary 2.590e-7), `rho_undefined_chi_tot_zero_rows=0`, `field_generation==lambda_generation=11`(세대 계약), `primary_clamp/floor/cap/fallback=0`, **1차 항등식 독립 재계산 최대 상대오차 = 0.0**(`rho_local == (chi_es/chi_tot)*lambda_star` 정확 일치).

**사전등록 판정 = (c) 발견.** 두 구간 다 **상한을 좁게 잡아 반증**됐다(운전석 앵커 오류).

| 사전등록 | 구간 | 실측(s8) | 판정 |
|---|---|---|---|
| 예측 1 — 에너지가중 ρ 중앙값 | [0.90, 0.98) | **0.98332** | 반증(0.0033 초과) |
| 예측 2 — `1/(1−ρ)` | [10, 50] | **59.97** | 반증(20% 초과) |

앵커였던 E8의 `chi_coherent/chi_total=0.9777`이 `ρ<0.9777`을 강제한다고 봤으나 **실측 ρ=0.98332가 그 앵커를 넘는다** — 이 epoch의 `chi_es/chi_tot`가 E8 때보다 높다(또 하나의 epoch 차이, 0j 계열).

**★그러나 예측 2의 실질 주장은 87배 여유로 확증:** E8의 `gain=5247.49`가 요구하는 `ρ=0.99981` 대 실측 `ρ=0.98332` ⟹ 실측 증폭 **59.97 vs 5247 = 87배 불일치**. **⟹ E8의 5247은 iteration amplification이 아니라 post-EPAY source 조성비다.** 두 양은 서로소 배열에서 나왔다(`chi_es`·`lambda_star` vs `eta_fixed`·`eta_coherent`)므로 **독립 2번째 경로**다. 처분 (a) 성립: `UV_CENSUS_CONSOLIDATION`·논문의 **"recycle gain" 표현 철회 대상**.

**부수 실측**: 셸 의존 강함 — s8 `1/(1−ρ)=59.97`, `τ≥100` 전체 **452.9**, 전 행 **517.4**. `τ≥100` 에너지의 **93.66%**가 `ρ∈[0.99,1.0)`(구 사전등록 판독 ≥90%을 우연히 만족하나 **정의가 다르므로 통과로 읽지 않는다**).

**★branch-site 처분 실측(재구성 아님, `acc_w` evidence bit 포함)**: 행 기준 `rate_shape_replaced` **33,756,653(89.8%)** · `legacy_source` 3,758,685(10.0%) · `thick_exempt` 71,512(0.19%) · **`scalar_rescaled` 0**. **에너지 기준은 정반대** — `legacy_source` **92.2%**(9.07e-4) vs `rate_shape_replaced` **7.8%**(7.71e-5). ⟹ **N9의 "99.56%"는 행 축과 에너지 축을 혼동한 결과**이며, 논문 §6.3의 68.6%/76.2%도 이 실측으로 교체 가능(정의 대조 선행 필요).

**★클램프 대장 공백 해소**: `eps_applied_diff_rows = 31,353,733 / 37,586,850 = **83.4%**`. C08/C08b가 "위험·발화 미측정"으로 남아 있던 자리의 실측치다 — **드문 가드가 아니라 다수 경로**. `eps_floor=1e-5`, `eps_cap=1.0`.

**신규 결함(β 다중주인)**: 체커가 `secondary eps_prime identity failure`로 fail-closed. 원인=production `radeq_beta_esc`가 **`tau<=1e-6`에서 β를 정확히 1로 절단**(`lumina_plasma.c`)하는데 페이로드 `beta` 열은 정확값 `-expm1(-tau)/tau`(체커가 rtol 1e-12로 검증·통과). **위반 5,826,525행(15.5%), 최대 상대오차 5.0e-07**, 위반자 β 중앙값 1.0·`eps0>0.99`인 위반자 0.03%(운전석 초기 가설 "eps0→1 상쇄"는 **반증**). 크기는 무시 가능하나 **종류는 다중주인 전수조사가 겨눈 실물**. 1차 측정량은 ε를 소비하지 않으므로 판정 무영향.

1. ~~parity43 판정~~ **완결(07-29) → C로**: V0-V5 전 스탬프(V3 CONFIRM-WITH-CORRECTIONS 11건 반영 + R1 SATISFIED). 파생: R2 잔차 3건(SAME_N 0.7/linear-vs-loglog 보간/O II·III Ω_SET 0.01)은 D 이월; 비표재 미세구조 M1선의 vR E1 취급(×13170 과소 사례)은 후속 후보
2. RATES_FIX 생산 A/B(**F5 포함** — 버그 수리의 생산 검증)
3. ~~제2 영점 수리~~ **구현·오프라인 인증 완료(07-29 밤)**: CONSWIN 게이트(withParityX, f_win 17자리 3중 일치, OFF-동일성 18/18, 맨 백라이트 끝-끝 1.000) → **생산 인증 폐합(07-30 parity46 ride-along)**: 판정4종+c1_bins+s45 byte-ident + CONSWIN 3종 세트 生(구 잣대 62.37 불변 유지 + windowed 63.2835 신설, f_win=0.985622) — **FORMAL-CONS 진리비 판정은 이제 windowed 값 사용**
4. R2 비보존 대체(연속 산란원 ~27% 순 싱크 — 광자 보존 위반)
5. ~~이산화 오차 검증~~ **결착(07-29 밤)**: 둘 다 실물이나 front-B 몫 1-3%(최악 5-15%) — 정답 아님; Courant 순부호는 등록과 반대(방출 미달이 투과 과대를 이김 → J̄ 과소); 사다리 잣대가 3-8× 과소평가(comb 대비 불변군 결여) → **후속(신규): 선담요 천공(셸당 20% 투과 vs 진리 0.015%)의 청→적 플럭스 재분배 전용 프로브 설계** — front-B/C 직결 후보
6. ~~베이커 실버그 2건~~ **결착(PASS)** → ~~후속 type-2/3/8 평가기~~ **결착(07-29 밤): 오라클 ≤1.7e-14 인증·10,864 정확 교체·빈티지-일치 이온 전원 중립~개선(Fe II +11%)**; Co III FAIL은 평가기 무죄(19apr23 vs 18oct00 데이터 개정 2.6×, D/C 양쪽-정확 실증) → ~~bakefix3~~ **결착(사전등록 전부 적중: Co III 0.993=대조군·+15% 해소·Co II 1.000 예측초과 — 율 기계 전 이온 무죄 확정; 단 구조변경 −44%라 진단 은행만)** → **후속: bakefix4 제한 정합(구조-중립 13이온, 발주됨)** = 생산 bin 교체 후보; 19apr23 채택 여부(CMFGEN 재런)는 D; [정정] BAKEFIX "S II 278"→S I·§5 천장 주장 반박; 잔차 공시: Si II D/C 1.079(베이커측)·Fe II A/PRRR 0.860(진리측)
7. 스펙트럼-잣대 의존 판정 재검토(기록의 정합성; FORMAL_FIX 상대 잣대로)
8. GPH_ALLLEVEL_NLTE 하니스 커버리지 갭
9. 계측 부채 잔여(legacy 1e-30 카운터·캡 절단 에너지 분율)
10. ▲H1 fb 채널 A/B — **계량 결과 강등: 정확성 수리 전용**(과소 7.5× 확정이나 채널 ~1e-6; 스펙트럼 <0.1% 불변이 사전등록)
11. ~~H2 럼핑 계량~~ **결착** → C로: SUPER_CUTOFF=100 로더 덮어쓰기가 광구 차단 78.4%를 지배(출하 f_to_s는 24%)
12. ▲~~H3 배너 오표기 수리~~ **구현완료(07-30 withParityZ: banner_gate_off 4상태 헬퍼, LTHERM/BSRC/KPR 3곳, active 배너 불변 — 운전석 검수 통과; 생산 인증=parity48 byte-identity)** + census CSV 로그 복사 관행(미완 잔여)
13. ▲배선감사 신규: ~~N1 설계~~ **결착(07-29)** → ~~Y2 계기 인증~~ **예비 완료(07-30 02:40, killed-partial iter0-10: 위생 전축 PASS·bfr=raw장 ν-가중 쌍둥이 증명·경로비 심부 NULL/외곽 Si II ×2.8·Fe II ×3.1 — 기전=TEPIN 빈14 청색부 문턱 정좌·범위=하위 16이온만(III/IV는 matrix R_bf 부재)·R1은 이미 C2 소비 ⟹ Y3=내부 불일치 제거; Y3 천장=empty-fill 조작 flux 25-76% 별도[조용히 기재]; P1-P8/HS1-HS5 등록 → parity52 큐)** → **정식 인증 완료(07-30 clean parity46 덤프): 계기 확증**(위생 0결함/왕복 5.6e-07/정규화 384,841빈 전원 한계 내/T_e 복원 정확/결정론: c1_bins byte-ident·c2 1행/55만 말자리) — **그러나 Y3 사전등록 붕괴 → 56_parity52 회수(hold/)**: P5 반박(s8 S II 5.85%가 HS1 5%를 도착 즉시 발동)·P2-P4 공허(**행렬 16이온 중 10개 전 셸 n_ion=0** — toy06 층화, PRELIM 헤드라인 다수가 빈 이온 율-계수 비)·P1 밴드 초과(실현 가중 s20-35=1.88-4.06 중앙 3.11, 44%>천장 3.5, **12이터 미수렴 단조상승**)·P6 반박(Ca II 25.6%)·Ca II 갭=코어스빈12 'fit' 별도 채널(TEPIN은 Si II 전용 99-101%로 정밀화). **재등록 요건 4**: 실현-가중 P밴드/HS1 재설계(기준선 자체 5.9%)/미수렴 대책(NI 연장 or 수렴 게이트)/공허 이온 제거·Ca II 분리. **[이중 인증 13시대]**: 세션 이중화 사고(원장 참조)로 쌍둥이·운전석이 각각 독립 실행 — 운전석 발주분(`y2_c2bfr_certification_formal/`)이 위생 전축 동일 수치 확증(왕복 5.619e-07·384,841빈 0 이탈·T_e 35행 max 0.46K·구조항등 0불일치)+수렴 부재 동일(Si II s30 it11=3.38 단조↑·s20 1이터 ×2.00·셸≥20 장 19-31%/iter)+**개정 괄호 제안 실재**(규칙 선고정: P1 pop-wtd ×2.1-7.2·ground ×2.6-8.3·P2 ×1.7-4.0·P4 ×0.7-0.8·P5 ≤0.11·P6 ×0.92-1.30; P3/P7/P8 존치)+**PRELIM 괄호주장 반증**(실현 가중 raw 1.621-3.492가 지상/Boltzmann 괄호 밖 13/16셸=proxy 과대)+조성 정밀화(등록 셸에 원자 실재=Si II·Ca II뿐; Fe II s20-35 X=0·Mg/Al/Sc/Ti/V/Cr/Mn 파일 부재·O/C 전역 0 ⟹ P2-P4는 HS5·용액측 관측으로 반증 불능). **양 인증 소차 2건[재등록 시 결착]**: P5 백분율(쌍둥이 S II s8 5.85%·Si II 4.64% vs 운전석분 10.47%·3.58% — 가중/필터 차 용의)·실현 P1 밴드(1.88-4.06[44%>3.5] vs raw 1.621-3.492[천장 내] — ema/raw 선택 차 용의). R_bf_table 덤프 계기 후속 유지 / ~~N7 A/B~~ **결착(07-29: 트랩 인과 반증 — lev17=M7 허용·게이트는 ×8.87 역방향·Fe III는 행렬 무공급; 실물=S3 캐스케이드 88.7% 스핀금지 착지+stage-IV 지뢰 → 수리 `LUMINA_REC_SPINGATE` 스펙 박제, 88.7%는 B6 σ 재정량 조건부)** → **Y4 판정런 결착(07-30 parity47 V0-V3 완주): 등록 미달성 = 선택규칙-표적 불일치**(lev17=M7 septet=spin-ALLOWED ⟹ 어떤 재결합 spin-gate로도 트랩 접근 원리적 불가; **초안 가설 "S3 미커버" 반박 — S3 착지는 이미 게이트됨**(:2648→:4345→cuda:4392, 운전석 실물 재확인), 실효 사이트=S3+S1(下이온), S2=TOPSTAGE_IV 사문+배너 과대) · **Si II quartet 26준위 −1.6% 직접 억제(5.8σ, p≈0.001) = 유일 양성 지표(회귀 스위트 고정)** · **formal FUV 1000-2000Å −10.0%(기근 방향 악화) ⟹ 기본 OFF**(= C 21호); 부산물: levelpop **writer 의존 10×**(committed +1.6% vs resolve_raw +16.4%, 부호 robust)→writer 명시 의무·프로방넌스 구멍(빌드 19:31<src 00:24, 드리프트 N3 국한 확인[잠정])→verdict_gate mtime 관문(B19) / ~~N2~~ **Y6 결착(07-30 parity49 V0-V3 완주): 게이트 生 + 구조 차폐 = 코드 필연**(jbar_count·jblue_line 동일 crossing·동일 피가산식 cuda.cu:3576-3587[운전석 재확인] ⟹ J_blue==0⟹count==0 ⟹ IUP-JBLUE=ON에서 rate 무효과; 최강 증거=it10/11 앵커 발산의 thin-out 단일버킷 국소화+thin-in clamp 정수 동일=MC 수송 불변), **판정 4종 byte-ident ⟹ 채택=배선 위생 폐합**(= C 20호; 구성-의존 명기: 실효 조건=IUP_TRAD=0∧use_j_nu=1∧BINFIELD=0∧JBLUE=0∧det창 외부, or THEN-MC 단독 경로 cuda.cu:8921[jblue 짝 없음—차폐 논증 붕괴 구멍, B19]; 커버 83.5-90.8%; [3,10) 8.8%는 Si II/III 덤프·iter2-11 한정; 잔여[UNVERIFIABLE]=진단 6값 말자리 플립≤4e-7[교차바이너리 W/Y 명기, A/A 대조런 1회로 해소 가능]·it10/11만 플립 이유) · ~~N3 설계~~ **결착(07-30 dig: ARTIS=단일 공유 함수+감쇠 부재로 분열 구조적 불가(radfield.cc:819-826 순수 덮어쓰기·macroatom.cc:571-604 공유)·titer 0.5는 봉인 — ARTIS-충실=raw 통일; 게이트 스펙 LUMINA_JBAR_DAMP_UNIFY, 정본 docs/N3_JBAR_DAMP_UNIFY_DESIGN.md; **구현완료 07-30 01시대: withParityAA(2d0cfec5), 단일파일 3헝크 +107−1, OFF=블록-수준 branch 하니스 md5 동일 실증, 운전석 검수 통과 — 인증=parity50(OFF-ident)·판정=parity51(=1, §5 사전등록) 큐** → **판정런 V0-V3 완주(07-30) [잠정: parity50-조건부]**: W PASS(ema≡raw byte-ident)+**I-1 PASS 역전**(clamp-제거 판독 시 unclamped 평균 A −0.076→B 0±0.001 3버킷 붕괴=소비지점 항등 제2증명 — 운전석 초판 "잣대 오류" 분류를 V3가 반박; clamp-보정 판독=앵커 표준 승격)+HS-1 미발동(11계량 전부 <2, 실질 flicker +7.7%, runaway 재발 없음)+D-1 부분(Si III만 강건 67:13·점등 36; S III 역방향 절반=이온총량 보존 잣대 아티팩트); 효과=Fe IV ×0.870·formal 청+4.6~8.9%/적NIR −22%(too-red 반대)·FORMAL-CONS 진리비 12.92→13.05× 소악화; **채택 방향 지지 — 확정=parity50 byte-ident 후**(V3 N-1: W→AA 3세대 경유·Z 마커無=단독 귀속 미인증이 최대 유보); 오프라인 진폭 median 0.007dex나 |Δlog n|>1 항목(p44 기준 126)·Si III/S III 들뜬준위 1e-30 플로어 플립 최대 ~37dex[조용히 기재]; PASS 시 Y6+N3 동시-ON 폐합런(§7) 등록) → **★확정 승격(07-30 저녁 = C 22호): parity50(AA OFF-항등) PASS** — V3 직접 대조 **50-vs-44(=51의 A arm) 총 잔여 10바이트 전부 1-ulp·정렬 jbar 0/293.5만행·수송 정수 동일·FORMAL-CONS 문자열 동일** ⟹ N-1 해소가 3세대 사슬 아닌 직접 항등으로 완성, 판정 계량 여유 사실상 ∞(N3 효과: jbar ~99% 이동·수송 −1.98%·FORMAL 62.37→63.68 vs 잔여 1-ulp 10바이트; D-1도 7자릿수 여유). **[잠정] 해제 = N3 채택 확정**; §7 폐합런 = 58_parity54(y6n3: 같은 AA 바이너리로 vs parity51 항등 사전등록 — 코드젠 변명 불가) 큐** · **신규 N10(dig bycatch): 문턱 미달 폴백=당-이터 raw binned 장(cuda.cu:7093/7866) — 한 소비자 안 문턱 경계 세대 혼식** / ~~N9 카운터~~ **구현완료(07-30 withParityZ: 호스트+디바이스 4사이트, 계수만·RNG 무소비; [FB-EDGE] 첫 실측=parity48)** / ~~K6 모순~~ **dig 결착(07-29) → 쌍둥이 V0-V5 결착(07-30) = C 19호(생산 기준선 전-DR-OFF 채택·런처 기본값 전환)** |
15. ~~▲verdict_gate.py c4 문자열 버그~~ **결착(07-29)**: note4 분기 수리 + 양 분기 검증(p43 "binary differs (declared)"·p42 "identical", 공히 PASS exit 0, --no-write로 기존 판정 산출물 불변). 기존 preflight의 :16 문구는 ADVERSARIAL에 버그로 주석됨
14. 미인증 계층: **Lumina J̄ 구성(3단계 본선)**·자체 인구·재결합/Milne 측·Kramers 폴백·fine-ν 서브빈 + Si II/III τ 재판정
16. ~~인증 커버리지 전이온 확장~~ **확정(07-29 밤, fable CONFIRM 8/8 — 3단 관문 완주)**: 커버리지 9→21, 3분류 결함(준위절단[데이터]/심부-Wien[기계, 실무 0]/fit-type[수리됨]) → C로. **후속 B**: ①~~베이커 준위캡+부재 6이온~~ **결착(07-29 밤: bakefix5 — 절단 D/C→1.000 완전 폐합·부재 6이온 첫 인증·기존 비트동일; [정정] 4b 전제 오류 에이전트 반증; ★Co IV 빈티지 1.5× 신규 노출·no-VM 지뢰(S III 3항) 문서화·CAP_TOPION 사다리 1단 상승 물리변경; 생산=앵커 B 후 bakefix6)** ②~~NORAD (28,1) 수리~~ **구현완료(07-30 withParityZ: 라벨 진실화 (28,1)→(28,2) shadow, 채택 판단 없음; 생성기 왕복 1필드 diff 증명; 현 기본 config 영향 0 — ★단 FROZENIN_DR=1 계보(parity42/43/45) 재현 시 실변수 플래그)** ③Fe III Milne합 vs NORAD 7× 갭 감사 ④~~Ni II NB01 σ·TOPbase Ca 다운로드~~ **결착(07-29 밤, fable CONFIRMED 9/CORRECTED 1/BLOCKING 0 — 3단 관문 완주)**: 피크 원본 실재(원본 지상 최대 1.03e-11 cm²)·**CMFGEN=2점 수기 클립**(÷100·÷10, 가수보존; 2점=지상 ∫σdν 78.8%, α_gnd ×4.73)·237엔트리 전수 217 identical/19 꼬리절단(규칙 19/19)/1 변조·TOPbase Ca 3이온 완비(단위 Mb 왕복 확립; [C8 정정] Ca V f3 상수는 싱글릿 164블록만)·rrc 왕복 byte-identical → C로; **신설 후속=Milne/rrc 자기일관 판별기**(NORAD σ와 α가 동일 계산 산출물 — raw vs 클립 중 NB01 자기 α 재현측 판별, 오프라인, 파일 전부 로컬) ⑤s8 PRRR/POP 스냅 불일치(계기) ⑥**bakefix6**(no-VM+FULL_LEVELS+S III/Co II phot 수리 동반 — 앵커 B 착지 후, 생산 후보)
17. **s12-s30 S-대역 요동 포켓 + 미수렴 종료(신규, K6 쌍둥이 V3 발견)**: 양 arm 각자 포켓(p44 s26-29·p45 s14-22, S II 진폭 ~0.3)이고 s20-s30은 3런 공히 T_e 상승 중 종료(+46~73K/최종반복) — "국소 쌍안정" vs "반복 미수렴" 미판별. 이 대역 절대수치는 판정 인용 금지(방향성만). 후속: 반복수 연장 A/B 또는 수렴 진단 계측(front-B; 조용히 기재, 튜닝 금지)

18. **▲REC_SPINGATE 내부 비일관 3건(parity47 V3 신규, 조용히 기재)**: ⓐ CMF `eta_bf` 준위별 bf 방출률(plasma.c:6213-, CMF_BF_MILNE=2로 **현 config 활성**)이 spin 술어 미적용 — 개체수/캐스케이드에서 잘라낸 재결합이 판정 잣대 스펙트럼 연속체에 잔존 ⓑ FB-MILNE exact(plasma.c:4713-, 현재 OFF) 미게이트 — 켜는 순간 구멍 ⓒ S2(TOPSTAGE_IV) 사문 + 배너 "S3/S1/S2" 과대진술. spin 규칙이 4-5 사이트 중 2곳만 적용 = 물리 모드 단일성 위반 소지
19. **▲판정 계기 보강 2건 + 잔여 미결(07-30)**: ①verdict_gate.py에 **바이너리 mtime vs 인용 소스 mtime 프로방넌스 관문** 추가(구현 대기) ②THEN-MC 단독 경로(cuda.cu:8921)의 jbar-단독 무장 — Y6 차폐 논증이 붕괴하는 구성, 해당 구성 판정런 설계 시 단일변수 목록 의무 ③~~parity49 진단 6값 플립 원인~~ **결착(07-30 parity46이 A/A 겸함, V3 교정판): 시그니처 6값 행·값·방향 완전 재현(p46≡p49: s8/s49 byte-ident·정렬 jbar 0差·raw 9146/421208 동일) = W→Y 인터리빙-불변 코드젠 差 확정, 게이트 무관. 단 '결정론' 전역 표현은 철회 — 파이프라인은 비트-재현 아님(CE 잔차 stdout이 p44/p46/p49/h100probe 4런 전부 상이·ALPHA-SPINGATE static int 경합 배너 4/1/1/2회·jbar 행순서 전면 재배열); 다른 인터리빙에서도 6값이 정확 재현된 것이 되레 코드젠 귀속의 최강 증거** ④신규[정정 — V3 전수 cmp -l, 07-30]: p46 특이 플립 = **6값**(resolve_raw 3행 436738/659344/659565 + resolve_ema 3행 418259/427961/430161; 전체 diff는 12값, max|rel|=3.0e-7) — 시그니처와 **반대 부류(비재현=자발 지터)**. 계측 게이트 무죄 3중 방증: h100probe가 게이트 0差로도 CE 잔차 발산·C2 writer 순수 관측·CONSWIN 불활성은 spectrum_formal 항등이 입증(+Y2 인증의 killed-vs-clean 1행/55만 말자리 동류). 처분=fail-and-waive(등록 2분할 처방은 비재현 현상에 귀속 불능=falsifier 없는 루프라 집행 면제). **판별기(무비용) 등록: gpuprobe 보존 목록에 진단 덤프 5종 추가→자발 플립 base rate 실측.** 진단 전용·판정 4종 불변, 조용히 기재 ⑤**Z-외톨이(parity48, 07-30 저녁)**: 등록 기대 문면 FAIL — it-11 패킷 재배선(수송 정수 터미널 −1534·이벤트 −3068; 통제 4런(W/H200·W/H100·Y·Y+Y6)+parity50(AA) 전부 불변, Z만 이동). 귀속 미결, 생존 후보=**빌드-코드젠 vs 확률 추첨** 둘뿐(의미론 변형·계보 유전 변형은 parity50이 기각 — AA는 Z1-3 의미론 포함 재빌드인데 수송 정수 정확 재현). 판별 **결착(07-30 20:18 parity53 분기 (a) 7/7, V3-3 PASS): H-det 확정 — Z 결정론 빌드-코드젠 캐리어**(같은 Z 2런이 it-11 정수·c1 0B·resolve_raw 0B·정렬 jbar 0으로 재배선 비트-재현; stale-input 6중 배제; 바이너리별 수송 원장 W=Y=AA=...959/Z=...425 — 4빌드 2상태). **Z 인증 FAIL·은퇴**(수송 발자국 3사이트: it-11 재배선+jbar it-8+s49; levelpop 플립은 지터로 재분류=판정 표면 영향 nil 측정 승격). N9 계기는 FB-EDGE 수리와 함께 차기 배치 재상륙. **프레임 교훈(정본): 교차-바이너리 byte-항등은 보장 아닌 경험적 결과 — 재컴파일마다 코드젠 초안(패킷-결정 문턱 초과 가능) ⟹ 바이너리별 인증 필수**(AA는 50에서 직접 인증=프레임 정상 작동). 51 오염 상한은 AA-계보 실측 0으로 교체(Z쌍 수치 은퇴). **잡음 바닥 원장(같은-바이너리 실측, 07-30 밤 갱신)**: 수송 정수 0·c1 0·정렬 jbar 0·s45/s49 0·s8 ≤1셀·levelpop ≤2행·c2_bfr ≤3셀·resolve raw/ema 3+3행; 클래스① 상한=비-resolve 7셀. parity51 오염 상한: b4 자리 1.87e-5(50-vs-44 직접 잔여 10바이트로 실효 ∞ 여유). ⑥**N9 [FB-EDGE] 인쇄 도달 불능(계측 결함)**: 인쇄가 MC 에필로그(cuda.cu:9782)에 있는데 pure-CMFGEN 경로는 :8791→:8799 return으로 종료 — 카운터는 가동, 첫 실측 유실. 수리=인쇄 위치 이동, 다음 바이너리 배치 편입(AA 계보 봉인 유지). ⑦**계측 규칙(3번째 재발로 승격)**: 러너/런 생존 판정은 산출물 성장(mtime/크기)으로만 — ps/pgrep은 계산 노드 상이 시 맹목(로그인 노드에서 match 0인데 런 진행 실증 2회).
20. **▲함수-감사 결함 대장(07-31, 외부 감사+Opus V3 확증; 정본 docs/V3_FUNCTION_AUDIT_VERIFICATION_2026-07-31.md + 원장 해당 절)**: D-1 bf continuum CDF 폐기+nu_edge/nu 분기 전무 / D-2 D6 배너 허위(배너 과대 3호) / D-3 chi_bf stim-recomb 부재 / **D-4 MA_LINE_DESTRUCT 이중 계상(**이중 확정 07-31: Opus+Codex 독립 합치** — 산술 2p−p²·실측 가중 0.5298%[이터 최대 1.0177%] 일치; 전 parity 런 ON·지배 열화 채널; A/B 상대판정 생존·절대상태만 오염; **처분 확정=ARTIS lane 기본 OFF+재기저 A/B — 실행은 parity55 후 수리 시간표 D-4 슬롯**; 사후 수치 보정 불가[재분배 연쇄]·정본 docs/CODEX_P06_DOUBLE_DRAW_VERDICT.md; **★재기저 완료(07-31 parity57, V2): OFF=수리-등가[(1−p)p만 제거·p 잔존, V3-A1/A2 코드 확정] — FORMAL-CONS −21.90%·T_e(s0) −2897K[진리 방향]·Δb_4 −0.83%·근자외 2192-3235Å가 감소 63% 운반; 정본 logs/coevolve_consume_parity57/VERDICT_DRAFT.md + docs/CODEX_PARITY57_V3_ADVERSARIAL.md; 생산 기본값 채택=user 대기**)** / D-5 공유 lo-ion save/restore+쌍 이온화 블록의 hi-ion drain 부재 / G-1 per-ion pin(선언된 residual) / G-2 IUP_JBLUE≠현 ARTIS 소비자(arm 서사 재기저) / G-3 24-bin 기하 비충실(superbin·핀 개수·범위) / M-1 writeback 불활성. 수리=함수 내부 검증 국면 V0, 튜닝 금지.
21. **b4 계보 감사(07-31 신설, parity57 V3-C3 파생)**: 운전석 라벨 사고 실증 — "b4자리 3.908e4"는 b_k(col9)가 아니라 n_k(col7)였음(Codex C 적발·운전석 스팟체크 확정; 정본 CSV 헤더 shell,Z,ion,level_num,E_eV,g,n_k,n_ground,b_k,...). 감사 대상: ①캠페인 원장의 "재솔브 b4=3.08(≥2.5)"(parity26-diag)가 현행 s8 b_9=3.0779와 수치 일치 — 구 "b4"가 실은 b_9였는지(라벨 사고 계보) vs 시대·구성 차이의 우연인지 ②"b4 문턱 2.5" 판정의 유효성(문턱 자체는 사례18 감사에서 무죄 판정났으나 지표 식별이 흔들리면 재검) ③lev4 기준전이=1206.5Å(0→4)·1113Å=3→9/10/11 — 1113Å 형광 서사의 지표 준위 재확인. 처분: 감사 전까지 "b4" 인용 시 col·row·정의 명기 의무.

## D. 이월 (fine tuning 국면으로 — 지금 검증하지 않음)
▲N6 심판-피고 장 분리(잣대 정책) / ▲N5 IUP-JBLUE stim(ARTIS 대조 후) / ▲H2 SUPER_CUTOFF 레시피 변경(SL 컷 조정) / ▲JBAR-POPS 트립와이어 문턱 재설계 / det-jbar 수리안 A 비용 실측·채택(오염 자체는 C에 확정) / METACOLL 40이온 확장 / SAME_N gbar 0.7(유계 소항) / OMEGA_SET vs O II/III 0.01(0.056%, override 有) / Ti/Cr/Mn II 승급 A/B / PRRR 3× 기전(CMFGEN 내부 — 진리 사용 시 34/90깊이 제외 규칙만 C에 유지)
**상설 규칙(B에서 이동)**: fine 해상도 노브(ppd·vdop·n_ali) 변경 시 det-jbar 수렴 재검 의무(절벽 끝).

## C. 확정 — 21건 (▲=오늘 신규)
| 결론 | 근거 |
|---|---|
| 게이트 격자에 해 없음 | N2 +45%·N3 45런 0 |
| 2-20× 이온화 오차 = 장 귀속 | 2단계 인증 |
| Γ(Co III) FUV 지배 | Γ_all 분해 |
| mode-2 파국적 상쇄 기각 | 산술 |
| 상쇄 항등식 S·τ ∝ n_u | 6자리 → **2회차 리뷰 강화: parity43 s8/s45/s49 전수 14,567그룹 전부 ≤1.9e-6**(원본 4프로브의 ~3000×) |
| SKIP_Z 명분 화석(**Si II·Si III**) | p35(Si III)+p36a(Si II: thick 250선 붕괴 0건 재도출)+승격근거는 parity38 기준 재기술(metering_batch1) **[2회차 정정: 주석 :15406-15413의 피해자=Si II·반박된 명분 문구 잔존(위생 플래그)]** |
| LTE floor 혼합쌍=1000-1300Å 병리 | p37/38 |
| 장 되먹임 교락 | N5 |
| B/C 경로 Saha 비대칭 구조 | RATES_FIX 분석 |
| CAP_TOPION 에너지 장부 파괴 | dig_F18 |
| ▲**det-jbar: 광구 CLEAN·외곽 s≥30 CONTAMINATED(−55%)**; 실척도=표류량 D | 하니스 상호검증 + 운전석 재계산 자릿수 일치 |
| ▲OMEGA_FLOOR 정체=실데이터 덮어쓰기(85.9%를 중앙 18.7×) + 88.16%/253× 발화 | 적대 재검증 계량 |
| ▲CMFGEN 실처방 = OMEGA_GEN_**V3** 3단(실표/vR gbar 0.2·0.7/금지선 0.1, f_lu≤1e-5) | 소스 + 47.972 항등식 |
| ▲formal 구조 결함 확정(짝 비보장·영점 1.068·W·B 폴백) — 단 25× 귀속은 미확정(초열 S_l 개연) | ADVERSARIAL 13C |
| ▲Co III 105=full("105→full"은 신화)·Ca III CMFGEN 전이 0(구 SKIP 사유 오기록) | col_data 직독 |
| ▲조기정지 오판 실재(ρ≥3에서 43% 지점 "수렴") — parity42 기전은 아님 | 하니스 로그 |
| ▲S_l 3원화(binned=B(T_e) 100%·fine 폴백 20.9-99%·formal=W·B핀) | 소스+실측 |
| ▲**K6 실체: DR 감사=per-ion 분열(S/Si/Fe→OFF 정합·Co IV→III만 마스크-ON 정합)인데 배포=일괄 ON(parity1-43 전 체인, α 1.15-3.94×@10kK 이중계상 + Γ 자동이온화 짝 부재)** | dig(하니스 A/F·config echo 전수·60144ad) + 운전석 자릿수 재검증 2건·소스 3건 |
| ▲**parity43 판정: Ω 3단 처방 채택(provenance 수리, adopted-with-residuals)** — 값 사슬 CMFGEN ASCII→bin→보간→출력 끝-끝 재현·기전 ≥4 결합(귀속 UNRESOLVED)·43/50셸 T_e 발산 소악화는 밴드 내 | V0-V5 전 스탬프(VERDICT_* 4종) |
| ▲**이산화 2후보 실물이나 front-B 정답 아님(몫 1-3%)**; Courant 순효과=J̄ 과소(등록 부호 기각 — 방출 미달 승); 1000-1120Å 창결함 좁고 깊음(광구 무죄); **유사성 사다리는 comb 대비 축에 맹목(3-8× 과소평가)·fine refine은 악화** | EXACT 경로 하니스(게이트 4종) + 운전석 CSV 대조 |
| ▲**인증 커버리지 결함 지도(fable CONFIRM 8/8)**: 한계=추출측(참조 27종 전부 데이터 보유)·s0 지배이온 전원 기존 미인증·**LUMINA 원자 준위절단 6이온(인구 ~0인데 Γ 20-95% 운반 — 인구잣대 맹목 부류)**·Ni II 빈티지 730×(+구베이커 ×3.235)·NORAD (28,1) 단일행 오배정·ARTIS Fe-피크 지상=Verner 재구현(비독립) | coverage_extension/REPORT.md + fable 재검증 |
| ▲**K6 결착(쌍둥이 parity44/45, V0-V5 완주): 생산 기준선 = 전-DR-OFF(parity44 config)** — 근거 3축: 잣대-충실(조성 내 DIE 전부 F,F; DIE_CoIII=F,F VADAT:703)·DIECoIII 실측=Co IV 전셸 진리 역방향(#33 결손의 반대부호 보정)·배선 단일변수(DIECoIII=코드 유일 생존 DR 채널, frozen-in 게이트 뒤; 단 STAGE4 ON 시 소실). K6 오염 실증: 일괄 DR=S/Si/Fe 이중계상 α(제거=노출, 등록 방향 전부 적중·Si II 정정 ÷3.41로 강화). DIECoIII 단독≈p43 Co 효과(s≥2; s0은 마스크 44엔트리 60%). 수리=run_coevolve_s01.sh 기본값 전환(오버라이드 보존) | 쌍둥이 V1×2 PASS + V3(CONFIRMED 37/CORRECTED 13/REFUTED 3/BLOCKING 0) + 운전석 스팟체크 3/3 |
| ▲**[20호] Y6/N2 결착: MA 문턱 분열=실재, rate-무효과=IUP-JBLUE 구조 차폐(코드 필연)** — 동일-tally(cuda.cu:3576-3587)·thin-out 국소화·판정 4종 byte-ident; 채택=위생 폐합; **게이트 생사 3종 세트 규칙(사례 13)의 실물 표본** | parity49 V0-V3 완주(VERDICT_DRAFT V2-final + V3_ADVERSARIAL) + 운전석 소스 재확인 |
| ▲**[21호] Y4/N7 판정런 결착: 선택규칙-표적 불일치**(Fe III lev17=M7 허용=게이트 원리적 무감·S3는 기게이트·등록 미달성) + **Si II quartet −1.6%@5.8σ=유일 직접 신호** + **FUV −10% 실측 대가 ⟹ 기본 OFF** | parity47 V0-V3 완주(VERDICT_DRAFT V2-final + V3_ADVERSARIAL: 순열검정·술어 복제 764/1500·formal 대역합) + 운전석 급소 3건 재확인 |
| ▲**Ni II 19apr23 = NORAD 충실 임포트 + 2점 수기 클립**(1.36e-12 피크 원본 실재·원본 지상 최대 1.03e-11 cm²·클립 2점=÷100/÷10 가수보존=지상 ∫σdν 78.8%·전수 237: 217 identical/19 규칙적 꼬리절단/1 변조·문턱은 OP→NIST 재스케일 1.0465) — **"원본과 다름≠원본이 맞음"의 실물 표본 2종**(클립=큐레이션, 문턱=실험 보정). TOPbase Ca 3이온 로컬 앵커 완비(단위 Mb 왕복 1.0004) | external_download/REPORT.md + FABLE_VERIFICATION.md (3단 관문) |

**상설 규칙(07-30 추가, parity47 V3)**: levelpop 기반 판정은 **writer 명시 의무**(committed `lumina_levelpop.csv` vs `resolve_raw` vs `resolve_ema`가 준위별 10× 차 실증 — 부호만 writer-robust).

## ★3회차 회전 리뷰 (2026-08-18, 15일 정지 후 재가동)

08-03 이후 회전 리뷰가 멈춘 사이 08-08~08-18 에 새 주장이 쌓였다. 이번 라운드는
**그 신규분만** 등재한다(기존 A/B/C 표본 재도전은 4회차로 이월).

### 강등 3건 (08-08 폐합 → 부분 폐합)
| 단 | 강등 사유 | 미실행 게이트 |
|---|---|---|
| SH-GAMMA | 사전등록 게이트 미실행 상태로 "폐합" 기재 | NC2·NC4·Γ4 (**08-18 현재도 미회수**) |
| DET-R6 | "byte-parity" 요구에 필터된 로그 줄 비교로 답함 | R6-4(stdout/stderr 분리)·N6-2·N6-3 (**미회수**) |
| SH-GRID | "어디서 얼마나 움직였나"에 "표면화 안 됨" 유보로 답함 | B-6 — 단 08-08 이후 **MC-EVT census 가 단을 재개방**시켜 sigma fresh bake·photo-rate 재작업으로 진화(현 상태 `ASSET_COMPLETE_PHOTO_ACCURACY_OPEN`) |

### 신규 C 승격 후보 → **B 등재**(독립 경로 1)
- **K36 결정론 baseline**: R1 45회 residual 9.6662782724980344e-09 · R2 52회 8.1222406993212508e-09,
  **08-18 신규 바이너리(S_producer 캡처 추가)에서 자릿수까지 재현** ⟹ 결정론·무변조 독립 경로 2번째 확보.
  단 회전 리뷰 생존 이력이 없어 A 승격은 보류.
- **population 음수해 원인 = stage-total lock 이중 권위**: `ION_LOCK=0`·`PER_ION_RESCALE=0`
  single-total A/B 에서 double/80자리 모두 비음수 finite. 근거=인계서 §3.2.

### ▲정정 1건 (기존 주장 강등)
- **"Stage-4 선택 IV 1,282선 전부 J 후보"** → **기각**. 근거는 잣대 불일치였다
  (분자 Jbar=생산@T_e 10020.0 K, 분모 S=시행@19059.411 K; Wien 꼬리 444–1884Å 에서
  B 비 37.8~4.6e6). 정본 `docs/VERDICT_A210_STAGE4_JO_2026-08-18.md`(Codex 감리 반영본).
  ⟹ 「잣대부터 감사」 사례 13 등재.
- 동 판정의 **V2("생산 Jbar 가 폐합을 만족")는 [확정] 아님** — B(T_e) 대용 일치(중앙 0.9992)이며
  생산자 실제 S 미캡처. DET-SPROD 단(진행 중)이 확정/기각한다.

### 진행 중 (판정 대기)
- **DET-SPROD**: 생산자 두 가수 직접 캡처. S0 빌드 PASS · S1 음성대조 PASS
  (`=2` → `INVALID_SPRODUCER_CAPTURE_REQUEST` → R6 BLOCKED → FATAL) ·
  캡처 활성 상태에서 R1·R2 bit-identical(무변조 실측) · S2/S2b/S3 판정런 진행.
  판정기 `scripts/a210_sproducer_ledger_check.py`(음성대조 6종 selftest PASS).

### ★미실행 게이트 회수 (2026-08-18, user 지시 "미실행 게이트 회수해")

| 게이트 | 성격 | 상태 |
|---|---|---|
| **SH-GRID B-6** 값 이동 장부 | 오프라인 측정 | **회수 완료** — `validation/sh_grid_b6/SH_GRID_B6_LEDGER.md`. T1(단의 실제 변경): 포함 True·손실 0(1e-18)·이동은 **FUV 집중, 최대 2.6e-4**(찬 프로브)·생산 T_e 에서 4.5e-5 ⟹ **현 격자 재빈닝 오차 상한 ≈3e-4**. T2(sigma bake 전후): dlog·상단 경계 **bit-identical**, 이동 ≤1.3e-15 ⟹ **BF 격자를 1000→1234 로 갈아엎어도 canonical 불변** — 안 B 의 "파생하라" 설계가 실증됨. 한계: 해석적 Planck 프로브 기반이며 **선 국소 이동은 미측정**(구 격자 J 덤프 부재) |
| **SH-GAMMA NC2** · **NC4** | 주입 단위시험 | **회수 완료 PASS** — `GAMMA_PUBLICATION_STALE_EPOCH` / rc=4 `GAMMA_DOUBLE_PUBLISH` + 배열 byte 보존 + 양성 대조(다른 epoch 는 정당 재발행). 증거 `validation/gate_recovery/` |
| **DET-R6 N6-2** | 주입 단위시험 | **회수 완료 PASS** — 한 글자 근접 오탐도 `QHASH_MISMATCH` 로 거부. 기존 커버는 전혀 다른 해시를 넣는 조잡한 형태였음 |
| **DET-R6 N6-3** | 주입 단위시험 | **회수 완료 PASS** — `NEGATIVE_OR_NONFINITE_JBAR cell=1 validity=1 value=-1`, `rejected_at=commit`. ★검수가 v1 의 "무관한 실패도 PASS" 구조(08-08 강등 3건과 동급)를 잡아 단계 분류를 넣은 덕에 **의도한 가드가 막았음**이 증명됨 |
| **SH-GAMMA Γ4** M1/M2 | 런 의존 | **회수 보류 — 사유 있는 이월**. 08-08 `validation/gamma/GAMMA_VERDICT.md:130` 이 "M1/M2 는 A2-10 성공을 요구하는데 MC 전송이 죽어 도달 못 한다" 고 **명시하고** 이월했다 ⟹ 누락이 아니라 의존성 대기. MC 팔이 A2-10 에 도달해야 회수 가능 |
| **DET-R6 R6-4** MC 바이트-parity | 런 의존 | **회수 보류** — MC 팔 2회 런 필요. 현재 GPU 2쌍이 DET-SPROD 판정런 점유 중 |

## ★4회차 회전 리뷰 (2026-08-19~20) — 오라클 계보 정정이 지배

### ▲▲▲강등 — CMFGEN 자체런 기반 정량 결론 전부

`docs/RETRACTION_INELIGIBLE_ORACLE_CITATIONS_2026-08-20.md`.
프로젝트가 **2026-08-04** 에 `validation/a2_00_oracle/` 매니페스트로 이미 판정해 두었다 —
`CMFGEN_PHYSICAL_ORACLE=INELIGIBLE` · `NONLINEAR_CONVERGENCE=FAIL`([13800,8980,3460]%) ·
`SNAPSHOT_REPLAY` 는 *"cross-file physics gating 불허"*.
그런데 08-17 성분 대조가 `LINEHEAT`/`NETRATE` 를 물리 판정에 썼다
(CMFGEN 참조 144건, **적격성 인용 0건**).

**회수**: 흡수 1/172 · net 4.37e5 · 방출 1.41× · `oracle_vs_cmfgen` 커버리지 17.01%.
**정성만 유지**: "CMFGEN 은 양 부호를 갖는다"(수치 인용 금지).

### ▲NaN 함정 (신규 등재)

NaN 종료 CMFGEN 런 **8개**. `MAXCH = 0.0000000000000000` 은 **NaN 의 서명**이며
"MAXCH ≤ 1%" 형 게이트가 이들을 **완벽 수렴으로 인증**한다(이름이 `_conv` 인 것 포함).
⟹ 게이트 권고: `MAXCH == 0.0` 을 PASS 로 읽지 않는다. 근거 `ORACLE_NAN_AUDIT_2026-08-19.md`.

### ★C 승격 — 오라클 무의존 결론 (독립 경로: Lumina 내부 항등 + 소스)

| 결론 | 근거 |
|---|---|
| 순-흡수 선 **0/1282**, `cancellation_condition=1.0` 정확, `absolute_signed_sum==signed_rate` 17자리 | 봉인 stderr `180b23f1…` 전수 재계산 |
| `S_producer/B(T_e=10020.0) = 0.999993`, 1282/1282 행이 1±1% | 〃 (생산자 두 가수 캡처에서 역산) |
| `J_cont ≤ B`(max 1.00000024) · `Jbar ≤ S_p`(max 1.00000011) | 〃 |
| **`RADEQ_NO_BRACKET` = LTE 시드의 필연적 귀결** | 위 셋의 사슬 + `res_lo −2.578`/`res_hi −1.49e8`. `FINDING_NOBRACKET_LTE_SEED_2026-08-19.md` |
| 생산자/소비자 상태 불일치 τ≈2.02e5 · S≈1.9e5 · η≈4.7e-11 배 | 〃 |

★이 다섯이 **현 본선의 표적**이며 오라클 붕괴의 영향을 받지 않는다.

### B 등재 — DET-STAGE12(고정-T 레인)

게이트 L1·L3·NL PASS(selftest 3종, `det_stage12 NL1..NL5 production_loader positive=6`),
L2·L4·L5·L6 은 판정런 대기. 감사 2회전 8건 반영. 부수 관찰 10건은
`LEDGER_DET_STAGE12_OBSERVATIONS_2026-08-19.md` 에 조용히 기재
(레인-맹 계약 필드 2개가 최대 부채).

### ▲정정 3건 (운전석 자기 오독)

08-19 하루에 같은 계열로 세 번 틀렸다 — 모두 **클립/센티널 값을 실측으로 읽은** 것:
① `MAXCH=1e7` 을 발산으로 ② 그 정정에서 다시 수렴도로 ③ 게이트를 여는 MAXCH 대신
옆 채널(`Maximum % increase`)을 외삽해 "2-3회면 Newton 이 열린다".
셋째는 Fable 갈림길 평가가 `CORRECTION_SUM`(외곽 51 깊이가 1800 변수 중 1200-1700개
≥10% 보정 요구)으로 반증했다. **교훈: 클립 가능한 요약값을 잣대로 쓰지 않는다.**

## ★5회차 회전 리뷰 (2026-08-26) — 차단 계측의 완결

### ★C 승격 — A210-ZERO-OPACITY census (오라클 무의존 · 독립 경로 2)

정본 `docs/VERDICT_A210_ZERO_OPACITY_2026-08-26.md` · 계약·장부 `docs/RUNG_A210_ZERO_OPACITY_2026-08-19.md` §10 ·
봉인 런루트 `/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1`
(job 324388 · COMPLETED 02:42:56 · a100/syn102 · A100-SXM4-80GB×2 · `git_head=e500e29` · 봉인 지문 136파일).

| 결론 | 근거 | 독립 경로 |
|---|---|---|
| hazard 행 **221,382** (셸 0 · Fe/Co/Ni ion 1), 카운터 == ROW 줄 수 (양 반복) | 봉인 stderr 전수 | 판독기 rc=0 + 운전석 독립 grep + 감리-B 재현 |
| **fa = 1.0 정확** — 전건 `tau_validity=A208_EXACT_ZERO`, coverage 결손 (b) **0건** | 〃 | 판독기 + 판정자 봉인 재계산 + 감리-B |
| 전건 `source_defined=0`(χ_eff=0 ⟹ S 미정의) · `n_upper>0` · `tau_raw`·`effective_tau` 정확 0 | 〃 | 〃 |
| 결정론 앵커 `line=669992` 4필드 **bit-재현** (L6C 승계) | 〃 | 판독기 anchor + 판정자 hex 왕복 + 감리-B |
| ★**반복 간 hazard 집합 완전 동일**(line·준위쌍 대칭차 0)인데 `n_upper` **bit-동일 0/221,382**(상대변화 중앙값 1.442e-2) | 〃 | 판정자 stderr 재계산 + JSON 교차 + 감리-A·B 재현 |
| ★덱 **221,382쌍 전건 `g_l==g_u`** (매핑 불일치 0 · `f_lu==0` 0건) | 덱 전수 실측 | 판정자 + 감리-B |
| 「`g_l==g_u` ∧ τ 정확-영 ⟹ `n_l ≡ n_u`」가 **부동소수점 수준에서 성립** (언더플로 여유 ~287 자릿수) | `opacity_publication.c:26-29` 직독 + 산술 | 판정자 + **감리-A 독립 재유도** |

⟹ **차단 계측이 완결됐다.** S4-III·L6C 를 죽인 `INDEPENDENT_SPROBE_UNDEFINED` 가 **0회** 발화했고
(캡처 2종=0 설계), hazard 행 목록·총수·성질이 처음으로 전수로 잡혔다.

### ▲미결로 남긴 것 (확정 아님 — 과대주장 금지)

- ★**본선 물리 계약(A2-07 solve 의 J̄ 응답)은 이 단이 풀지 않았다.** 이 단은 그 **앞의 차단**을 계측했다.
  측정은 Z-2A 착지 후에야 가능하다.
- `fa=1.0` 은 출판·물질 정합의 표지이지 **물리 인증이 아니다**. `(b)=0` 은 **스캔 범위 한정**이다.
- Z2-III · Z3 · Z4 게이트는 수리 후 게이트로 사전등록이 명문으로 유보했다.
- bit-겹침의 **귀속**(무엇이 그렇게 만드나)은 열려 있다 — 관측 기술만 확정.
- 세대 전이 관측은 **1회**(iter0→iter1). "매 세대" 는 n=1 외삽이다(감리-A 지적).

### ▲등재 — 계량 혼동 함정 (신규)

같은 런의 `cmf_fine` `exact_zero_tau=81,265,890` 은 A2-10 `zero_opacity_emitting_rows` 와 **비교 불가**다.
스코프가 셸(전 셸 vs **셸 0 단독**)·이온(전 이온 vs Fe/Co/Ni ion 1) 두 축에서 다르고, 판별식도
플래그 대 값+발광 요구로 다르다 — **포함관계 불성립**. 또한 라벨 `line_shells=27,748,410` 은 총수가
아니라 `signed_cells` 이며 `exact_zero_tau` 와 **서로소**다(총 셀 = 109,014,300, 정확-영 74.55%).
분모를 라벨대로 잡으면 **293%** 라는 헛수가 나온다. 근거 `RUNG_A210_ZERO_OPACITY_2026-08-19.md` §10.2.
⟹ 「잣대부터 감사」 사례집에 편입.

### ▲방법론 실측 — 감리 두 겹(개정17)의 값

감리-A(Fable fresh, 물리·주장)와 감리-B(Codex read-only, 재현)의 **검출이 겹치지 않았다**:
A 는 대수 사슬의 빈칸·주장 강도·전수 주장 과대(5건), B 는 인용 경로 오류·명령 주석 불일치(4건).
⟹ 감리를 **교체가 아니라 추가**로 간 판단이 실측으로 지지됐다. 폐합 조건(둘 다 통과) 충족.

## ★6회차 회전 리뷰 (2026-08-27) — bit-lockstep 생산자 귀속

### ★C 승격 — DET-BITLOCK (오라클 무의존 · 독립 경로 ≥2 · **런 0회**)

정본 `docs/VERDICT_BIT_LOCKSTEP_PRODUCER_2026-08-27.md` · 계약·장부
`docs/RUNG_BIT_LOCKSTEP_PRODUCER_2026-08-26.md` §11 · 커밋 `4bf4a5b`.

| 결론 | 근거 | 독립 경로 |
|---|---|---|
| hazard 221,382행의 `n_upper` 가 **(Z,iter)별 정확 1값** (dmax=1), 다중도 {26:106502, 27:105685, 28:9195} | 봉인 stderr 전수 | 판독기 G-B2 + 검수자 자체 파서 + 운전석 자체 파서 + 감리-B 독립 파스 (**4벌**) |
| ★선정 1,586행 중 **hazard 밖 1,432행**(τ≠0·등g 아닌 1,061 포함)도 전건 lump 값과 bit-일치 | 〃 + 덱 조인 | 판독기 D4 + 감리-A 표본 실물 확인 + 감리-B 재현 |
| `nlte_level_populations` 기록자 **131 참조 전수**에 계열 밖 기록자 **0** | 소스 전수 | 판정자 + **감리-A 전수 재수행** |
| 클램프류 노브 16종이 **금지 레지스트리**(위반 시 TERMINATE)에 있고 봉인 exports 전건 부재 | 소스 + 봉인 | 판정자 + 감리-A |
| ★**귀속**: 생산자 = `LUMINA_SUPER_CUTOFF` 단독 설정의 **반상태** — lump 레이아웃 × `f≡1.0` × SL-차원 solve. 실물 자리 `lumina_plasma.c:19398` | 소스 사슬 6단 + 수치 논리곱 | 판정자 결합 + 감리-A 재유도(전제 4개 명시·충족 확인) |
| 봉인 `stdout:236` 자기모순 배너 `Super-levels: off (identity) (13015 FL -> 1173 SL)` | 봉인 | 운전석 + 판정자 + 감리-B |

⟹ Z-O §9 #5 **승계 미결 폐합**. **GPU 런 0회** — A2-07 배선 감사(08-22)에 이은 두 번째
런-없는 귀속 종결.

### ▲미결 (확정 아님)

- **본선 물리 계약(A2-07 solve 의 J̄ 응답)은 이 단도 풀지 않았다.**
- `n_lower` 는 어디에도 인쇄되지 않는다 ⟹ 「n_l≡n_u」 전체가 **출판 대수 경유 추론**이다.
  (감리-A 평가: D2 실측 + 소스 필연성만으로도 따라오므로 이 의존은 판정문 자인보다 가볍다.)
- ★**같은 반상태가 L6·L6C 봉인 exports 에도 실린다**(`LUMINA_SUPER_CUTOFF=100` 실측).
  그 단들의 전제 재감사는 **미결 인계** — 단정 금지.
- ★**기전의 사정거리는 NLTE 인벤토리 11개 이온 전부**(13,015 FL = 62+11×101)이나 **측정은
  커버 3이온뿐**이다. 재감사 단 개설 시 대상은 11이온 전부다.
- 수리 단 미개설 — 발견의 처분은 **조용한 대장 기재**다.

### ▲등재 — 게이트 한계 2건 (신규)

1. ★**인용 실존 게이트는 컴파일 여부를 보지 않는다.** G-B3 가 인용 행의 실존만 검사해,
   운전석 A1 이 `#if 0` 死코드 7행(`lumina_cuda.cu` 1083–2063)을 라이브로 인용한 것을
   통과시켰다. 판정자가 검출·정정, 감리-A 가 자체 전처리기-깊이 오토마톤으로 독립 확인.
   ⟹ 차기 게이트는 **전처리 분기 깊이**를 함께 검사할 것.
   **교훈**: `grep` 으로 기록자를 열거하고 **지역 문맥만** 읽으면 1,000줄 위의 `#if 0` 이 안 보인다.
2. **발주 프리플라이트는 「범위 축소」를 못 잡는다.** 발주서가 계약 값을 재서술하지 않아
   DP-1 을 통과하면서도 계약이 판독기에 실은 게이트를 금지할 수 있다(이 단 발주 중단 1건,
   주인=운전석). 재서술 검사와 **범위 포함 검사**는 다른 축이다.

### ▲방법론 실측 — 각 겹이 앞 겹이 못 본 것을 하나씩 잡았다

| 겹 | 검출 |
|---|---|
| Codex(구현) | 발주서↔계약 충돌 — 가짜 통과 거부하고 멈춰 보고 |
| 검수(Fable) | NC-K0 판별력 실증(변조 주입) · G-B3 커버리지 구멍 |
| **판정(Fable fresh)** | **운전석 A1 의 死코드 인용 7행** |
| 감리-A(Fable fresh) | A1 **별칭 포인터 사각**(스스로 폐합) · 함의 이온 일반성 과소 기재 |
| 감리-B(Codex read-only) | 좌표 설명 오류 · 명령 표기 · **자기 차단을 정적 오라클로 재검증해 해소** |

⟹ 개정17(감리 두 겹)과 개정16 ②(검수 선행)의 값이 두 단 연속 실측으로 지지됐다.
★감리-B 가 운전석 대리 실행을 **그냥 받지 않고** 픽스처 sha256 을 손계산해 교차 검증한 것이
대리 실행을 수용하는 옳은 형식이다.

## ★7회차 회전 리뷰 (2026-08-27) — 진단 격리 in-vivo

### ★C 승격 — A210-Z2A (오라클 무의존 · 독립 경로 ≥2)

정본 `docs/VERDICT_A210_Z2A_DIAG_ISOLATION_2026-08-27.md` · 장부 사전등록 §10 · 커밋 `4fc84fc`.
봉인 R-II `/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/rii_20260826T134337Z_ion1`
(job 324557 · 04:34:36 · a100/syn101 · 지문 135파일).

| 결론 | 근거 | 독립 경로 |
|---|---|---|
| **격리 Class-1 in-vivo 작동** — hazard 442,764회 발생 중 물리 완주, `te_generation=2->3` 커밋 | 봉인 R-II | 판정자 + 감리-A + 감리-B |
| `INDEPENDENT_SPROBE_UNDEFINED` **0회**(L6C 를 죽인 그 차단) · ISOLATED 0 · BLOCKED 0 | 〃 | 3자 |
| **f_bit = 1.0** — ROW 442,764행 census 와 byte 동일 | 〃 | 판정자 + 감리-A `cmp` + 감리-B |
| env delta 0 · 사망 기전 접촉은 `406712d` 단독 | exports·git 계보 | 판정자 + 감리-B |
| hazard **442,764/442,764 전건 `source_defined=0`** ⟹ 가드 보유+capture=1 은 통과 불능 | 봉인 | 판정자 + 운전석 |
| ★`e500e29` 는 **실런 실적 보유** — census 런이 정확히 그 커밋, **L6C 사망 노드 syn102 에서 완주** | 봉인 provenance | 감리-A 발견 + 운전석 |

### ▲미결 (확정 아님)

- ★**본선 물리 계약(A2-07 solve 의 J̄ 응답)은 이 단도 측정하지 않았다.**
- **Class-2 격리는 발동 실적 0** — selftest 계급만. 실증 아님.
- **E-I1 문면 FAIL** — `sacct FAILED 70:0`. 모델이 아니라 **수용 블록**이 죽였다.
- **R-III·R-IV 제출 불가** — §5.5 가 「런처 수용 PASS 후」를 요구하는데 실물이 없다.
- 전역 물리 무접촉은 미주장(등록 bit-대조 범위 내만).

### ▲등재 — 계측 결함 2건 (신규)

1. ★**앵커 검사의 판별력 0**: `analyze_a210_z2a_probe.py` 의 `ANCHOR_BITS` 가 **iter0 값 고정**
   인데 `anchor_matches()` 가 **양 반복에 적용**한다. 앵커의 `n_upper`·`emission_per_sr` 는
   반복 간 정당하게 변하므로 **iter1 이 필연 불일치** ⟹ **봉인 census 자신을 넣어도 FAIL** 한다.
   Z-O 판독기는 앵커를 iter0 에만 적용해(`public_blocks[0]`) E-Z3 가 통과했다.
   **selftest 가 못 잡은 이유**: NC-I0 픽스처가 양 반복에 동일 행을 재사용 — 원리적 검출 불능.
2. ★**fail-open 벡터**: 후속 단 PR-0 이 **판정문 문면을 리터럴 매칭**하므로, 판정문에
   「감리 A·B 통과」를 조건문으로 두면 **감리 미통과 상태에서도 오충족**된다.
   ⟹ 앵커는 **판정문이 아니라 장부**에서 찾아야 한다. 본문에서 리터럴 제거로 1차 차단.

### ▲등재 — 잣대 함정 (신규)

`sacct State=FAILED 70:0` 만 보면 **런이 죽은 것으로 오독**한다. 실제로는 모델이 04:34:36
완주해 `model.rc=0` 을 내고 R7 커밋까지 갔으며, 죽은 것은 **수용 블록**이다.
⟹ 「클립 가능한 요약값을 잣대로 쓰지 않는다」의 네 번째 사례.

### ▲방법론 실측 — 세 단 연속, 두 겹의 검출이 겹치지 않는다

| 단 | 감리-A 가 잡은 것 | 감리-B 가 잡은 것 |
|---|---|---|
| Z-O | 대수 사슬 빈칸·인용 기전 오류 | 재현 명령 실패·경로 오류 |
| BIT-LOCKSTEP | A1 별칭 사각·함의 과소 기재 | 좌표 설명 오류·명령 표기 |
| **Z-2A** | 반례 수색·전제 의존성 분석·**저자의 과소 주장** | ★**provenance 전제 거짓**(핵심 논증) |

★**차단이 결론이 아니라 전제를 쳤다** — 저자가 전제를 사실로 교체하고 논증을 보강했으며
결론은 불변이었다. 「결론을 지키려 전제를 꾸미지 마라」의 준수 여부는 감리-A(재)가 **골대 이동
여부로 별도 판정**했다(이상없음).
★**감리가 과소 주장도 잡는다** — 「`e500e29` 실런 부재」 한정이 과도하게 약했다.
**낮춰 쓰는 것도 부정확이다.**

## ★★8회차 회전 리뷰 (2026-08-28) — 본선 물리 계약 첫 측정

정본 `docs/VERDICT_L6C_REDISPATCH_2026-08-28.md`(+`.rev1`) · 계약·장부
`docs/RUNG_L6C_REDISPATCH_2026-08-27.md` §13 · 커밋 `e11de26`.
봉인 R-II `…/rii_20260826T134337Z_ion1`(135) · census `…/zo_20260826T050021Z_census_ion1`(136).
**GPU 런 0회** — 이미 봉인된 데이터를 판독했다.

### ★C 승격 — DET-L6C-COVER-R (F1b) · 오라클 무의존 · 독립 경로 **4**

| 결론 | 근거 | 독립 경로 |
|---|---|---|
| **ff = 95/885 = 0.10734463276836158** — C-R 경계(≤0.10) **강부등 위**(동결 95 vs 상한 88) | 봉인 R-II·census | 판독기 + 판정자 + 감리-A + 감리-B (**판독기 import 0 의 독립 파서 3벌**) |
| **C-M-S 술어 성립** — 동결 집합 ≡ BL∩FPC, **line-id 대칭차 0** | 〃 | 4자 |
| ★**`ff_responsive = 0.0`** — 응답 가능 계급(EXP∪MIX **790행**) 동결 **0**, 최소 d `2.928690524872657e-06`(동결 문턱의 2.9e7배) | 〃 | 4자 |
| `f_mapped = 2193/2193 = 1.0`, 비표적 0 | 〃 | 3자 |
| **RD-4 재구성 항등** 상대편차 표본 3행×3축 **전부 정확히 `0.0`** | 〃 | 판정자 + 감리-B |
| 「커버 이온 시드 = LTE」 **반증** — `\|S/B−1\|≤1e-3 = 0/1,053`, med `28.186` | 〃 | 판정자 + 감리-A + 감리-B |

⟹ **동결은 전량 반상태 구조 계급의 대수 필연에 귀속되고, 응답이 정의되는 790행은 전건
동결 이탈**이다. 계약 등록 문언 «**반상태 solve 의 SL-단위 응답**» 이 **관측 등급**으로 착지했다.

### ▲★주장 상한 (확정과 혼동 금지)

**측정되지 않은 것**: 준위별(**FL**) 응답 · **정량 옳음** · **반상태의 정당성** ·
**비커버 8이온**(측정은 커버 3이온, 기전은 11이온에 미친다 — BIT-LOCKSTEP).
C-R 급 「실증」 라벨을 쓰지 않았고 「물리가 옳다」·「CMFGEN 수준 재현」은 **명시적 미확정**이다.
감리-A 가 **어휘 스캔 전수**로 상한 초과 문장 **0** 을 확인했다.

### ▲★규율이 지켜진 자리 (실측)

- **골대 불변**: `ff=0.10734` 가 경계 `0.10` **바로 위**다. 사전등록이 **문턱을 알고도 불변을
  선언**하고 «BL 제거 시 0/790 → C-R» 을 공개하며 거부했다. **움직였으면 C-R 이 나왔다.**
  부등호 방향 **3자 일치**(계약 §9-2 · 기계 선언 §12 · 판독기 `:994-996`).
- **사전 등록된 반증 가능 술어**: `C-M-S`(F ≡ BL∩FPC)를 **데이터 존재 전에** 등록했고 실패 가지
  `C-M-U` 도 함께 등록했다. F1b 규율(사전등록을 데이터 열람 전 커밋)의 증거가 커밋 `2e7e41c`
  시점 R-II `PENDING` 으로 박혀 있다.

### ▲등재 — 계측 결함 3건 (신규, 수리 0 — 계약이 이 단의 수리를 금지)

1. ★**F-1 `sacct` 교차-클러스터 provenance**: 판독기 `:1136` 이 `sacct` 를 **클러스터 무지정**으로
   호출하고 운전석이 **grammar-debug 에서 집행**해, 보고서의 `324557|COMPLETED|0:0` 이
   **grammar 의 무관 작업 `UPP_QE68_SCF`** 였다. syn 정본은 `LUMINA_Z2A_II FAILED 70:0`.
   ★**게이트 무영향 실증** — 비종단 집합에 `FAILED` 부재로 정본으로도 통과하고, 감리-B 가
   상태 주입으로 `FAILED→PASS`·`RUNNING→RII_NOT_TERMINAL` 재현.
   ⚠**운전석 오류**: `sacct` 는 연산이 아니라 **클러스터 귀속 상태**인데 연산 규칙으로 보냈다.
   ★기존 규약 「`squeue -u kjhan` 은 계정 범위」의 **클러스터 판**이고, `NC-R14/R15` 는 텍스트
   주입 픽스처라 이 계급을 **원리적으로 못 잡는다**.
2. **F-2 σ 반복 스코프**: 판독기 iter0(`:896`) vs 계약 술어 J̄₁(iter1). 양 반복 q10
   (`0.7275646582`/`0.7215715484`) 둘 다 SIGNAL — 분기 무영향.
3. **F-3 proxy ≢ capture 전건**: `4,078/4,078` bit 불일치(비율 ~0.95·0.69 계통차).
   E-5① `[미확인]` 을 **부정으로 해소**. 분기량은 전부 캡처 잣대라 무영향, **정량 그림만
   프록시-한정 강등**.

### ▲선행 사슬에 대한 답

- **L6 분기 B** — **ion IV 계급 한정으로 유지**(배선 부재 계급과 배선 있는 계급은 모순 아님).
- **「iter0 = LTE」** — **커버 이온 선정 행 한정으로 폐기**. 비커버 몫 **PC-2 는 존속** — 과잉 철회 없음.
- **`NO_BRACKET`** — 공유 전제의 사망만 기재하고 **재판정하지 않았다**(§10 경계 준수).

### ▲방법론 실측 — 네 단 연속

감리 두 겹의 검출이 **네 단 연속 겹치지 않았다**. 이번엔 감리-B 가 **환경 제약**(Slurm DB 차단·
`/tmp` EROFS)을 **「아마 맞을 것」으로 넘기지 않고** 재현 불능으로 적은 뒤, report 쓰기와
live `sacct` 만 제외한 **우회 경로**로 `PR-3=PASS`·본체 `PASS C-M-S`·`RD-1~7` 전건 PASS 를 확인했다.
⟹ 재현 불능의 정직한 기재와 우회 재현의 병행이 **대리 실행을 수용하는 옳은 형식**이다.

### ★A210-Z2A R-III/R-IV 폐합 (2026-08-29) — 분기 `ZI-O-G6-DECLARED-PATH-ABSENT`

**등급 A (경로 ≥2)**
- **R-IV 값 파일 byte 불변**: 봉인 L6 대비 **9/9 byte-동일**(등록 3쌍 + 실물 6쌍
  `work/physics_comparison/physics_DET_iterNNNN.{manifest.json,shell.csv,spectral.csv}`).
  경로 = 운전석 독립 대조 + 감리-B 독립 재현.
- **G5 S2 항등**: `Jbar == continuum + local_emission`, `max_relative_deviation=0.0`(tol 4e-16),
  `producer_terms_undefined_rows=0`. NC-G5 음성대조 시연(`UNDECIDABLE_PRODUCER_TERMS_MISSING`).
  ⚠**캐비앳 존속**: 이 항등은 **구성상 참일 수 있어 물리 검증이 아니다.**
- **두 런의 전역 상태 결정론**: 4.5시간 GPU 런 2개가 `LINE-ION-OWNER-SUMMARY` 8행
  **전행 byte-동일**(T_e·n_e·rate 21자릿수).

**등급 B (검증 대상 — 관측만)**
- 반복 0→1 물질 응답: T_e 고정(10,020 K)·n_e +0.196% 인데 signed rate **−8.83~−10.54%**,
  셸 바깥으로 **단조 증가**. 잡음 바닥 `signed_grouping_delta ≤ 2.03e-16`.
  ⚠**J̄ 구동 귀속 아님** — A2-10 은 `interpretation=DIAGNOSTIC_ONLY`. d-proxy 분리가 남았다.
- ion III vs IV: 방출 비 **2.7e-9**(53.2 대 1.44e-7) · ion IV 선택 **불변**(297=156+20+121) ·
  hazard/sprobe 가 **ion III 전용**(182,244/반복 대 0).
- ⚠**「297 = DET-A207-WIRING 의 297」은 미확인** — 동일 집합인지 확인되지 않았다. 인용 금지.

**미착지 / 결함**
- **G6·E-I8 미착지 — 계약 결함**: 등록 9경로 중 6개가 **경로 형태만 다른 실물**로 존재.
  판정자가 경로 대체를 거부(계약 재작성이므로). 라벨은 「등록 경로 부재」이지 실물 부재가 아니다.
- **S2b FAIL** 존속(`missing_j_cont_rows=389`, tol 1e-12) — 계약 밖, DET-BETAPROD 미구현.

**★절차 기록 (운전석 이탈 4건 — 전건 감리 검출)**
1. **제출 자격 이탈**: 개정 2가 **결과 확인 뒤 골대 이동**(축자 `OLD:`/`NEW:` 없이 ~90행 신설)
   으로 무효 판정 ⟹ user 지시로 철회, 구 조건 「R-II 런처 수용 PASS 후」 복귀 ⟹ **미충족**.
   ★**산출물은 유효**하며 물리는 손상되지 않았다.
2. **1차 판정 발주서의 정본 역지정**(폐기본 `.rev1` 을 정본으로) ⟹ 판정 폐기·재발주.
3. **이온 대조축 설계 오류** — 전역 잣대가 `TARGET_ION` 을 소비하지 않는다.
4. **NC-R14/R15 제출 전 기재 누락**(사후 집행 rc=0).
⟹ 이 넷이 **개정18 ⑤ + 운전석 게이트 3종**(`check_doc_canonical`·`check_driver_hygiene`·
`check_gate_roster`)을 낳았다. 셋 다 음성대조 내장.

**감리**: 감리-B 통과(«폐합 차단» 0) · 감리-A 차단 2 → 철회·(d) 정정 → **확인 감리 통과**.

---

### ★DET-SUPERLEVEL-B 폐합 (2026-08-30) — 분기 `SLB-F` · d-proxy `DPX-X`

**등급 A (독립 경로 ≥2)**
- ★**B-2 사망 행의 정체와 수치**: Ca II H `λ=3969.591342 Å`(Z=20 ion=II, 0→3, shell 33,
  line 1370680). `S/J̄−1 = −1.03736542245e−7` 대 `σJ̄/J̄ = 1.45761196807e−7`, 여유 **1.405×**.
  `srce_chk=clamp=floor=0`. 경로 = Fable 감리-A 도출 + Codex 감리-B 독립 재현.
- **Ca II 무럼프**: 봉인 덱 62준위(0–61) 전부 singleton (`super=min(level_num,K)`, K=100).
- **대조**: `SIGN_MISMATCH/UNRESOLVED/R7 FATAL` = R-II 0/0/0 · R-III 0/0/0 · **B-2 1/2/1**.
- **승격 전 자유도 축약**: 짝 2,420 · ΔJ̄ 상이값 2,420 · Δn_upper 상이값 **73** · 준위 1,599 ·
  **축약비 21.904109589:1**. 표준 Spearman 선-수준 **0.4588** · 군-수준 **0.6286**.

**등급 B (관측 — 귀속 아님)**
- ★**「정확해가 위반 가능한 가드」 계급 발동**: 산란 지배 공명선의 깊은 셸 열화 극한에서
  `η−χJ̄ → 0` 이 정확해다. 반상태 시대에는 S 가 g-아티팩트라 은폐됐다. **버그도 물리도 아닌
  게이트-대-정확물리 충돌**이 선도 가설(감리-A). **인과 단정은 하지 않는다** — 단일 노드
  반복이 없다(판정).
- ⚠**「73 ≈ SL 수」는 틀렸다.** 대수가 주는 것은 `73 ≤ 커버 SL 수` 한쪽 부등식뿐이고,
  BITLOCK 의 교차-lump bit-일치는 `73 < SL 수` 쪽을 시사한다. **인용 금지.**

**미착지**
- G-2~G-7 (런 산출 부재) · E-4/E-5/E-6/E-7 · d-proxy P-1~P-4 전부(짝 0).
- B-1: 계약 §8+§7-1 대 런처 `:223-231` **계약 내부 비호환** ⟹ 실행 불가.

**★철회 (운전석 사전등록 결함 — 커밋 `a2e8fe7`)**
d-proxy P-1·P-2·P-3·§2-1·§2-2 **무효**. ① `f=f(T_e)` 라 셸-T 동결 시 반복비에서 소거 ⟹
P-1「등급 1 대수 필연」이 **대수 오류**(SLB E-5 함의가 같은 대수를 이미 옳게 등록했는데 반대로
씀). 완주했다면 `DPX-B` 거짓 양성. ② 군내 셔플은 짝 다중집합 보존 ⟹ **귀무가 공허**,
보고된 95% CI 는 **순위함수 동률 미처리 아티팩트**. ③ ρ 등록값 둘 다 오류(정답 +0.4588).

**★감리 두 겹의 값 — 계열 분리의 실측 재확인**
Codex 검수 3회 + 판정 2회가 통과시킨 것을 **Fable 감리 1회가 차단 4건**으로 잡았다(대수 오류·
공허 검정·분기 오판 X→F·귀속 역전). 그중 둘이 **분기 라벨 자체**였다. 대장의
「Fable=물리·대수 / Codex=재현, 교집합 0」이 다시 확인됐다 — 개정18 ②의 근거.

**후속 (SLB-F 처분 = 사유 특정 후 재작업, 런 소모 없는 것 우선)**
① 인증 게이트의 열화-극한 거동 처분(오프라인) ② B-1 계약 내부 비호환 해소
③ d-proxy 재설계(「반복-Δ 상이값 수」 아닌 조작화)

---

### 9회차 예정

기존 A 표본 재도전 · 런 의존 게이트(Γ4·R6-4·NE2·ME2·E4) · `toy06_19.48d` 계열 146건 인용의
성격 분류(미완).

**관리**: 이 파일이 정본. 강등·승격은 날짜·사유 기록. 1회차(07-29 낮, B/C 적대 재검증)·**2회차(07-29 밤, A×2+C×2 재도전: 강등 0·정정 3·강화 1)** 반영 완료 — 3회차는 나머지 A 표본 + C 신규(K6·parity43) 재도전 예정. 예약: A3 프로브 로그 보관 재주행(withParityY 착지 직후, Y4 리팩터 회귀검증 겸용).

---

## 2026-09-05 — d-proxy SL 수준 공변동 판독 (런 0회, 판정 아님) — `docs/READOUT_DPROXY_SL_COVARIATION_2026-09-05.md`

**등급 C (확정 사실 — 독립 경로 1, A 승격 대기)**
- **플랫폼·패치 등가성**: 331050(H200, provenance 바이너리 `a4990a59`) 대 328645(A100, 봉인 `39d452e4`) ROW **685/685 행 9필드 bit-일치**. ⟹ A100 재런·332010 재실행은 새 정보 없음(둘 다 취소, user 09-05).
- **승격 모델 대수 정합**: runtime SL 안 `Δlog(n/frac)` 산포 최대 1.8e-15 ⟹ `n_FL = N_SL·f(T_e)` 성립. `within_sl_frac` 반복 간 변화 0.
- **runtime SL 구조**(shell 0, Z=26/27/28 ion 1, 98 SL): 다멤버 SL 은 Fe 465(66준위)·Co 869(21준위) 둘뿐, 443행 단독 anchor.

**등급 B (관측 — 귀속 아님)**
- 반복 0→1 SL population 변화는 **이온 공통모드 지배**: 승격 Fe +3.3%·Co +0.7%·Ni +4.3% (SL sd ≤1.1%), 반상태 R-II +10.7%·+28.3%·+9.5%.
- SL 수준 `rho_SL`(midrank stratified): 승격 **+0.008**, 반상태 −0.129. 블록별 Fe +0.50/+0.79(양, 두 모델 반복), Co −0.01/−0.29, Ni −0.23/−0.33. permutation null 미계산 ⟹ 유의성 무주장.
- 공통 231 짝에서 `rho(Δlog n_prom, Δlog n_semi) = −0.44` — 승격이 SL 별 변화 방향을 뒤집음. ⚠승격 대 반상태는 단일 변수 대조 아님(소스 판본·K36/42·capture 도 다름).
- 짝 수준 ρ +0.61 은 91멤버 럼프의 유사복제 아티팩트(철회 P-3 재확인).

**후속**: 분기 라벨이 필요하면 사전등록(D4 귀무)→판정 Fable fresh→감리 두 겹. 물리 다음 후보 = 공통모드의 출처(이온 분율 반복 변화)를 기존 산출로 분해.

### 09-05 후속 — 공통모드 출처 분해 (런 0회, 배선) — 판독 §8
- **등급 C**: 이온 공통모드는 `scale_ion` 이 **아니다**. 이온 분율 rate-SE 비의 입력이 Γ_phot(정준 J ≥4 eV 반복 간 동결, LTE 가중)·T_e 고정·n_e +0.1956% 뿐 ⟹ `|Δlog scale_ion| ≲ 0.2%`. 준위해는 `Σ준위 = n_ion` 으로 정규화(`nlte_solve_ion_shell:19823`) ⟹ 공통모드 = 이온 안 여기 준위의 공통 departure 이동.
- **등급 B**: 그 이동의 가용 구동자는 선 J̄(<4 eV) 뿐 ⟹ population 의 J̄ 응답은 **집단적(행렬 결합)**이지 SL 별이 아니다.
- 대장 후보(조용히 기재): `nlte_writeback_ion_stage` 의 `1e-300` floor 2건(이 런들에선 비활성). 이온 밀도 writer 는 있으나 어느 런도 안 켬(`LUMINA_ION_POP_DUMP_ITER`).

### 09-05 후속 2 — 집단 응답 기전 (런 0회) — 판독 §9 · ★09-05 후속 1 의 등급 C 정정
- **[정정]** 「공통모드 ≠ scale_ion」은 **Saha `ion_number_density` 에만** 맞다. `ION_LOCK=0`·`PER_ION_RESCALE=0` 이라 준위해는 쌍 총량만 보존하고 단계 분배는 행렬이 정함 ⟹ 준위해의 **이온 II 총량은 움직였다**: 바닥 준위 S-역산 Fe II +3.62%·Co II +0.49%·Ni II +4.09%(승격), +12.5/+30.5/+9.2%(반상태). 역산 정확도: 같은 하준위 공유 선 간 산포 8e-11.
- **등급 C**: 공통모드 = NLTE 행렬이 정한 이온 II 단계 총량의 이동. 한 런에 이온 II 밀도가 둘 공존(Saha 대 NLTE, writeback 꺼짐) — 이중권위 계급.
- **등급 C**: 관측 338선의 외부 구동 0 (전부 ≥4 eV, shell 0 bin J 변화 <1e-4; `rho(Δlog J̄,Δlog S)=+0.99` 로 J̄ 는 S 의 그림자). 전 선목록 펌프 구동 `Δlog P_u` 중앙 1e-6.
- **등급 B**: 구동자는 <4 eV bin J(최대 +7.8%, IR 0.5~0.75 eV; `Δlog χ_bb` 와 +0.77) 가 bb 율로 준위 분포를 바꾸고 준위별 Γ_l 차이로 단계 분배가 따라가는 여기–이온화 결합 반복. UV 핀은 shell 0 만.

### 09-05 후속 3 — 이온 단계 밀도 이중 권위 배선 감사 (런 0회) — `docs/AUDIT_DUAL_ION_DENSITY_WIRING_2026-09-05.md`
- **등급 C (배선 사실)**: 한 런에 이온 II/III 밀도 둘 — Saha `ion_number_density`(n_e·쌍 총량·η_bf 재결합 항·비사상 선·Γ_phot) 대 NLTE Σ준위(사상 선 τ·S·η, χ_bf 준위 항, ROW, CMF 방출). `ION_LOCK=0`·`PER_ION_RESCALE=0`·`OPACITY_IONSTAGE` 꺼짐 ⟹ 합쳐질 기제 없음. 어긋남 = Fe II +3.6%/반복(승격)·+12.5%(반상태).
- **등급 B**: 전하 보존이 권위를 가로지름(n_e 는 Saha, 수송은 NLTE); bf 한 과정 안 χ/η 권위 혼합; CMFGEN(단일 행렬·단일 n_e)과의 구조적 차이 ⟹ 재현 목표의 귀속 후보.
- 대장 후보(조용히): bf `n_ion<1e-30` skip · radeq_simul `mixed>nel2` cap(비활성) · writeback `1e-300` floor ×2(비활성).

---

## 2026-09-06 — DET-DUAL-AUTH-2 폐합: 분기 **DA-R** (판정 Fable fresh ×2 · 감리 A Fable fresh 통과 · 감리 B Codex 통과) — `docs/PREREG_DUAL_AUTHORITY_2026-09-05.md`

**등급 C (경로 2)**
- `LUMINA_NLTE_PER_ION_RESCALE=1` in-vivo 작동: gen-1 exact epoch base·arm bit-동일(`45 / 9.6662782724980344e-09`), gen-2 상이, 공통모드(base Fe +3.62%·Co +0.49%·Ni +4.09%/반복) 가 핀 아래에서 소실(arm −0.06%·−0.02%·+0.32%).
- S-역산 `n_l = n_u(g_l/g_u)(1+2hν³/(c²S))` 는 ROW η/χ 항등식의 정확 역함수(감리-A 재검, Einstein 관계 덱 편차 ≤1e-6).

**등급 B (경로 1)**
- ★**이중 권위 크기의 첫 직접 측정** `n_II^Saha/n_II^NLTE`(shell 0, iter 0): **Fe II 0.884 ± 0.004 · Co II 1.0231 ± 0.0002 · Ni II 0.9549 ± 0.0001** (population 가중 평균 항등, ± = 준위 간 산포). NLTE 분배가 Saha 보다 Fe II 를 13%, Ni II 를 5% 더 준다. iter 1: 0.852/1.018/0.920.
- formal 스펙트럼 불일치(arm/base, 1,234 bin): [0,2) −0.17% · [2,4) +0.53% · **[4,6) eV +1.37%** · [6,12) +0.02% · [12,∞) ~0 (단일 bin 최대 +49% @523 Å). **불일치 크기이지 오차 크기 아님**(D-D 고정점은 제3 상태).
- Ni II 바닥 +0.321% ∈ (0.3%, 1%] ⟹ DA-R(이온 내 재배분 구간 — 구간 이름이지 시연 기전 아님; Saha 총량 이동 ≤0.195% 이면 재배분 ≥0.126%). Fe 럼프 SL 465 arm −6.05%.
- 사정거리: 반복 0→1·shell 0·Fe/Co/Ni II. 「Saha 옳다」·「CMFGEN 근접」 불성립.

**운전석 이탈(주인 = 사전등록·판독 저자 Fable 운전석)**: G-4 「338 전건」 상태 의존 선택 미검증(폐합 차단 → 개정 3) · E-1 「코드 귀결」 과대 · E-4 「단일 비」 전제 미검증·판독 과대 결론 · 집행 기록 역순. 개정 3 = 정당한 정정(판정 2·감리-A 독립 판단 일치: ≤337 어떤 문턱도 같은 라벨).

**«기재로 족함» 15건**(판정 4·감리-A 6·감리-B 5): E-4 가중 평균 항등 미기재 · E-3 문면 338→337 · §8-2 Γ_phot「<1e-4」 bin 수준 미재현(최대 3.8e-3, 배선 추론) · pub-0 J_ν 1069/1234 bin 상이 · §5 규칙 ln 전용 · 330 문턱 원리 · G-0 「8쌍」 오기(7) · sacct 샌드박스 불가 · selftest 미재실행 등.

**§5 규칙 발동**: [4,6) eV 중앙 ≥1% ⟹ **D-C/D-D 사전등록을 L6C 재감사보다 앞세움**(`docs/DESIGN_DC_DD_CMFGEN_CONTRAST_2026-09-05.md`), 결정 user. GPU 소모 = 이 단 1런(2h59m).

### 09-06 [정정] DET-DUAL-AUTH-2 G-1 의 `resolved_lumina.sha256 cmp` 는 공허했다
astra 사전감사(D-C-1, `docs/ASTRA_PREAUDIT_DC1_R1_2026-09-06.md`)가 발견: 그 파일은 스테이징 **전** exports 해시(`stage_a210_z2a_probe.sh:252`)로 재작성 뒤 exports 와 불일치(`29a91bf4` 대 `cc3155e6`), base·arm 이 같은 stale 파일을 복사해 `cmp` 가 항상 참. G-1 의 실효 부분은 registry·binary sha 였다 ⟹ 게이트 판정치 불변(«기재로 족함»), 잣대 결함으로 기재. 판정 2회·감리 2겹이 못 잡음.

### 09-08 배분 개정21 (user) — 판정도 astra ⟹ 계열 분리 소멸 (대장 기재)
**사유 = Fable 가용 불가**(user 09-08, 선호 아님). 개정18 ②의 「Fable 절약형」 예비 조항의 한계 사례이며, 그 조항이 명시한 대가 = **물리·대수 렌즈 상실**(Fable 이 잡은 실사고: 08-29 P-1 대수 오류, 09-05 E-1 동어반복). Fable 복구 시 감리-A 부터 환원.
astra 가 계획·코딩·사전감사·종료감사·판정 전부를 맡는다(운전석 = 집행·재현, Fable fresh 는 user 지시 시만). 개정18 ①은 「운전석 금지」로만 충족되고 **자기 산출 자기 판정**이 구조적으로 발생한다. 개정17·18 ②(감리 두 겹 계열 분리)는 이 시점부터 적용되지 않는다.
실측 근거로 남기는 것: 계열이 갈렸을 때 두 방향 모두에서 차단이 나왔다 — 09-06 Codex 검수3+판정2 통과분을 Fable 감리가 차단 4(둘이 분기 라벨), 09-08 Fable 판정 1회차를 astra 마감감사가 차단 4. 개정21 이후 이 축은 없다.
운전석 보완 의무: 발주서 저작자 명시·자기 저작분 표기 요구, 판정문 실측 주장 전건 재현 후 불일치 대장 기재.
