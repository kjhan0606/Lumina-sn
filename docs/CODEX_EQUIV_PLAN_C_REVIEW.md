# 독립 교차 리뷰 결과

**최종 판정: 반대(현 버전).**

노선 (c)의 방향 자체는 재검토 가치가 있지만, 현재 문서는 Stage 0의 핵심 전제인 “현 fixed-T 완주본을 Jν/rate/population oracle로 인증”이 실물에 의해 반박된다. 이 상태에서는 이후 비용·일정·acceptance가 모두 잘못된 앵커 위에 선다.

## 1. CMFGEN 포트란 인용 전수 대조

로드맵의 CMFGEN 포트란 locator 24개를 전부 확인했다.

| 주장 | 판정 | 실물 대조 |
|---|---|---|
| 모든 종·이온 STEQ/BA와 전하식을 조립 | **CONFIRMED/보강** | [cmfgen_sub.f:1588](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:1588)–1680에서 전 이온 STEQ/BA 초기화, `STEQ_MULTI_V10` 순회, `STEQNE_V4` 전하식 추가가 확인된다. 다만 “전 준위”는 fine level 전부가 아니라 **전 model-atom SL 미지수**라는 표현이 정확하다. |
| SE·RE를 한 선형계로 풂 | **CONFIRMED** | [solve_for_pops.f:1](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/solve_for_pops.f:1)–8이 목적을 명시하고, [solveba_v13.f:90](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:90)–104가 block-banded solve, [solveba_v13.f:311](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:311)–322가 인구와 T를 같은 correction vector로 갱신함을 보인다. |
| CMF formal ray + moment/VEF 반복 | **CONFIRMED** | [comp_j_blank.f:603](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/comp_j_blank.f:603)–768에서 formal 계열과 moment 계열을 모두 호출하며, [comp_j_blank.f:779](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/comp_j_blank.f:779)–808에서 Eddington factor 수렴을 검사한다. |
| line formal의 approximate diagonal Λ | **주장 CONFIRMED, 두 번째 인용 REFUTED** | [formsol.f:1](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/formsol.f:1)–16은 `LAMLINE` approximate line-diagonal operator를 명시한다. 그러나 [formsol.f:522](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/formsol.f:522)–533은 line `LAMLINE` 계산이 아니라 전자산란용 `JINT` 갱신이다. 후자는 오인용이다. |
| `χ_noscat J−η_noscat` RE 적분 | **CONFIRMED** | [cmfgen_sub.f:2305](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:2305)–2321에 그대로 구현돼 있다. 비간섭 전자산란일 때 `RJ−RJ_ES` 보정도 포함한다. |
| 선율·J·인구·T 변분이 BA에 들어감 | **보강—주장은 맞지만 locator 불충분** | [cmfgen_sub.f:2437](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:2437)–2502는 line net rate와 `DO_VAR_CONT` 호출까지만 보인다. 실제 BA line update는 [cmfgen_sub.f:2516](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:2516)–2525 및 `update_ba_for_line.f`의 population/T/J derivative 부분이다. |
| 조립된 SE·RE를 함께 갱신 | **CONFIRMED** | [cmfgen_sub.f:4001](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:4001)–4005에서 `SOLVE_FOR_POPS`를 호출한다. |
| CMFGEN super-level 구조 | **CONFIRMED/보강** | [steq_multi_v10.f:1](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/steq_multi_v10.f:1)–18은 full atom process를 작은 SL population으로 기술함을 명시한다. |
| full collision rate를 SL로 투영 | **보강—간접 인용** | [steq_multi_v10.f:113](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/steq_multi_v10.f:113)–181은 `SUBCOL_MULTI_V6` 호출부다. 실제 full→SL 합산은 [subcol_multi_v6.f:5](/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/subcol_multi_v6.f:5)–6, 177–230이다. |
| SL→full 복원 및 총량 재정규화 | **CONFIRMED** | [sup_to_full_v3.f:140](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/sup_to_full_v3.f:140)–175가 LTE 비율/보간 복원, [sup_to_full_v3.f:236](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/sup_to_full_v3.f:236)–250이 각 SL 총량의 정확한 재정규화를 수행한다. |
| ion별 복수 photoionization target | **CONFIRMED** | [comp_opac.f:86](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/comp_opac.f:86)–101은 모든 `N_XzV_PHOT`를 순회하고 `XzV_ION_LEV_ID(J)`를 넘긴다. [mod_cmfgen.f:160](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/mod_cmfgen.f:160)–184가 자료 구조를 확인한다. |
| target population 기반 stimulated recombination | **CONFIRMED** | [genopaeta_v10.f:175](/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/genopaeta_v10.f:175)–205에 식이 설명되고, [genopaeta_v10.f:252](/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/genopaeta_v10.f:252)–269에서 `CHI += σ(HN−T1)`, `ETA += ...T1`로 실현된다. |
| observer formal에 동일 χ,η 전달 | **CONFIRMED** | [cmf_flux_sub_v5.f:2053](/gpfs/kjhan/cmfgen_src/cur_cmf/obs/cmf_flux_sub_v5.f:2053)–2057에서 전자산란 emissivity를 합하고, [cmf_flux_sub_v5.f:2133](/gpfs/kjhan/cmfgen_src/cur_cmf/obs/cmf_flux_sub_v5.f:2133)–2166에서 같은 배열을 `OBS_FRAME_SUB_V9`에 전달한다. |
| Doppler/상대론 변환과 ray 적분 | **CONFIRMED** | [obs_frame_sub_v9.f:683](/gpfs/kjhan/cmfgen_src/cur_cmf/obs/obs_frame_sub_v9.f:683)–750에서 주파수·η·χ 변환, [obs_frame_sub_v9.f:942](/gpfs/kjhan/cmfgen_src/cur_cmf/obs/obs_frame_sub_v9.f:942)–1018에서 ray intensity와 flux 적분이 확인된다. |
| collisional ionization은 ground target만 | **CONFIRMED** | [steq_multi_v10.f:21](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/steq_multi_v10.f:21)–30에 명시돼 있다. |

핵심 구조에 대한 큰 오독은 없지만, “full linearization=모든 fine level”처럼 읽히는 제목과 line-BA/Λ 인용은 수정이 필요하다.

## 2. Lumina source 인용 전수 대조

Lumina source locator 21개 중 다수가 실제 주장 위치와 어긋난다.

| 축 | 판정 | 대조 결과 |
|---|---|---|
| pair별 셸 Gauss solve | **CONFIRMED/보강** | [lumina_plasma.c:15746](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15746)–15785가 실제 근거다. 로드맵의 시작점 15723은 진단 CSV 코드다. |
| 기본 5회·0.5 damping, element-wide residual | **REFUTED(인용)** | 인용된 16214–16240이 아니라 [lumina_plasma.c:16252](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16252)–16263에 있다. |
| shared lo-ion 저장·복원 | **CONFIRMED/보강** | 저장은 [lumina_plasma.c:16291](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16291)–16320, 실제 restore는 16331–16335다. 기존 인용은 restore 직전에 끝난다. |
| bin별 SC+ALI | **CONFIRMED** | [lumina_cmfgen.c:539](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:539)–680이 정확하다. |
| 주파수결합 CMF gate | **CONFIRMED** | [lumina_cmfgen.c:1535](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1535)–1546이 정확하다. |
| GPU `O(Nbin)` 반복 경고 | **CONFIRMED** | [lumina_cmfgen.c:1636](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1636)–1649가 정확하다. |
| 열식 `H−C` | **CONFIRMED/보강** | 함수는 [lumina_plasma.c:10479](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:10479)–10501이다. 기존 인용은 냉각 합산 전에 끊긴다. |
| coupled Newton은 셸 독립 | **REFUTED(인용)** | 실제 선언은 [lumina_plasma.c:12352](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:12352)–12355다. 인용된 12336–12339는 time-dependent charge gate다. |
| global A4는 조건부 gate | **REFUTED(인용)** | 실제는 [lumina_plasma.c:12417](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:12417)–12429다. 인용된 12401–12410은 shell 범위 설정이다. |
| SL 공간 solve와 Boltzmann 복원 | **REFUTED(인용)** | 인용 15730–15739는 진단 매크로다. SL solve 설명은 15755–15758, fraction 계산은 [lumina_plasma.c:16205](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16205)–16234다. |
| pair 보존·핀 구조 | **REFUTED(인용)** | 인용 15809–15820은 inversion ceiling 검사다. pair 보존행은 15686–15701, per-ion rescale/pin은 15832–15843이다. |
| 첫 mapped bf target collapse | **CONFIRMED/보강** | 실제 첫 target 선택은 [lumina_plasma.c:6541](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6541)–6544다. 기존 인용은 설명과 ground-target 구축만 포함한다. |
| neutral bf skip | **REFUTED(인용), 주장 CONFIRMED** | 실제 skip은 [lumina_plasma.c:6551](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6551)–6553이다. 인용된 6532–6537은 환경 gate다. |
| `χ=n_lσ`, stimulated recombination drop | **CONFIRMED/보강** | χ는 [lumina_plasma.c:6694](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6694)–6696, drop 자백은 6704–6714다. 기존 인용은 6703에서 끊긴다. |
| Gaussian line-overlap formal | **REFUTED(인용), 주장 CONFIRMED** | 실제 함수와 식은 [lumina_plasma.c:17230](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17230)–17245다. 인용 17207–17222는 앞 formal-integral의 에너지 출력이다. |
| e-scatter fallback와 bf `Bν(T_e)` | **REFUTED(인용), 주장 CONFIRMED** | 실제는 [lumina_plasma.c:17330](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17330)–17351이다. |
| field producer 분열 | **CONFIRMED** | [lumina.h:242](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:242)–265와 [lumina_cuda.cu:7153](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7153)–7208이 이를 직접 보인다. |
| 결정론 루프와 fine J̄ 보유 | **CONFIRMED이나 과잉해석** | [lumina_cmfgen.c:3167](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:3167)–3223에서 기본은 binned solver이며 fine J̄는 별도 gate다. “권위 코어로 승격만 하면 됨”은 현 성숙도를 과대평가한다. |
| EventRec 인프라 | **CONFIRMED** | [lumina.h:66](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:66)–104가 정확하다. |
| terminal 재추첨 | **CONFIRMED/보강** | [lumina_cuda.cu:4365](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4365)–4388에 두 번째 ε draw가 있다. 단 이중계상을 증명하려면 같은 `C_down`이 기존 `kp_deact`에도 들어가는 [lumina_plasma.c:4425](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4425)–4445도 함께 인용해야 한다. |

따라서 Lumina 인용은 주장 방향은 대체로 맞지만 file:line 정확성은 보고서 수준으로 불충분하다.

## 3. 노선 (c) 판정

### (a) 기각

**보강.** pair-SE·upper-target collapse가 packet 수 증가로 고쳐지지 않는다는 논거는 공정하다. 그러나 “MC J는 noisy하므로 전역 Jacobian을 닫기 어렵다”는 일반론만으로 MC 기반 비선형 방법 전체를 기각하는 것은 과도하다. variance reduction, deterministic response operator, common-random-number Jacobian 등 가능한 중간 설계를 비용 비교에서 누락했다.

### (b) 기각

**REFUTED/불공정.** (b)를 “CMFGEN 포트란을 C/CUDA로 거의 복제”하는 노선으로 정의해 비용을 키운 반면, (c)의 권위 코어도 frequency-coupled CMF, 전 원소 SE, charge, RE, `δJ`와 observer formal을 모두 새로 요구한다. 두 노선의 결정론 부분은 사실상 대부분 겹친다. MC 재결합까지 포함하는 (c)가 근거 없이 (b)보다 싸다고 단정됐다.

### 비용·일정

**REFUTED.**

- 단계 PM을 합산하면 29–51 PM이 맞지만 근거 WBS, 함수 수, 테스트 자산, FTE 가정이 없다.
- 선행 조건대로 직렬 합산한 Stage 0–6 기간은 **56–98주**다. Stage 7 계산 1–2개월까지 포함하면 약 **14–25개월**로, 제시한 12–18개월과 모순된다.
- 현재 A4 global은 전 준위 coupled solve가 아니라 셸당 `T_e,n_e` 두 변수의 2×2 block이다([lumina_plasma.c:12035](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:12035)–12069). Stage 4는 증분 개선이 아니라 사실상 새 solver다.
- 현 frequency-coupled 경로도 Courant 문제를 capped operator split으로 우회한다고 스스로 적는다([lumina_cmfgen.c:1554](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1554)–1564). Stage 3의 5–9 PM 역시 낙관적이다.

### Stage 4 위험

**최고 위험이라는 판정은 CONFIRMED, 평가 깊이는 부족.**

빠진 것은 Jacobian directional-derivative 검증, preconditioner 성능 문턱, 메모리·iteration 예산, trace population scaling, line-search 실패율, fallback 전환 시점이다. “마지막 3회 단조 감소”도 Newton/Krylov 수렴의 적절한 필요조건이 아니다.

따라서 (c)는 후보 아키텍처이지 현재 자료만으로 추천 확정할 수 없다.

## 4. 단계 계획의 주요 결손

1. **REFUTED — 시간의존성 범위가 없다.** 저장소 README는 자체런이 `DO_DDT=F` steady snapshot이고 공개 CMFGEN은 time-dependent sequence라고 명시한다([README.md:21](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/README.md:21)–29). CMFGEN 실물에는 SE comoving derivative와 time-dependent energy/adiabatic 항이 존재한다([cmfgen_sub.f:2920](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:2920)–2958). steady parity인지 published time-dependent parity인지 먼저 봉인해야 한다.

2. **보강 — model-atom projection 단계가 독립 관문이어야 한다.** 공통 level/SL/line/continuum ID, energy, g, stage coverage, collision·photoionization 데이터 버전, target map을 양쪽에 투영한 뒤 행 수·transition 수·checksum이 일치해야 한다. 현재 Stage 1은 continuum graph에 치우쳐 있다.

3. **보강 — 선흡수 위상과 bound-bound E-동등이 빠졌다.** `line_id→lower/upper full level→SL→흡수 activation→stimulated emission→net rate→macro-atom fate`의 전 사슬과 per-line LTE detailed-balance KA가 없다.

4. **보강 — 전자산란 재분배가 빠졌다.** coherent pure-scattering KA만으로는 CMFGEN의 `RJ_ES`, redistribution convolution, 전자 열교환을 검증하지 못한다. coherent/incoherent/Compton 범위를 명시하고 moment·RE·formal 모두에서 같은 kernel을 검사해야 한다.

5. **보강 — 과정 inventory가 불완전하다.** charge exchange, dielectronic/autoionization, non-thermal ionization·excitation, Auger/X-ray, two-photon, level dissolution, Rayleigh/dust/clumping, free-free 및 radioactive/time-energy 항이 어느 stage에서 흡수되는지 없다.

6. **REFUTED — Stage 2의 원소별 행렬에 전하/`n_e` 행을 각각 넣는 서술은 불명확하다.** 하나의 `n_e`는 모든 원소를 결합한다. frozen-`n_e` 원소 파일럿과 전 원소 global charge solve를 구분해야 CMFGEN식 전역계와 비교할 수 있다.

7. **보강 — acceptance 계산 규약이 검증 불가능하다.** 다음이 정의되지 않았다.

   - median/p95의 depth·frequency·rate weighting과 near-zero 처리
   - active set을 CMFGEN, Lumina, 양쪽 합집합 중 무엇으로 정하는지
   - SE/RE “정규화 잔차”의 행 scaling
   - 공간/주파수 보간과 shell mapping
   - EW·특징속도의 line list와 blending 규약
   - MC 다중-bin 신뢰구간의 family-wise coverage
   - 2배 해상도 한 번이 아니라 3단 grid/Richardson 검증
   - E-동등 판정을 위한 residual vector 및 Jacobian-vector 비교

8. **보강 — 결함 흡수표는 V3 D/G 목록만 폐합한다.** 위 시간의존성, model-atom projection, line topology, incoherent e-scattering, 비열 과정과 경계조건은 별도 defect/coverage ledger로 추가돼야 한다.

## 5. FIX_T와 released-T 요건

### FIX_T 탐지

**CONFIRMED.**

- [RVTJ:11](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/RVTJ:11): `Was T fixed?: T`
- `VADAT:622`: `T [FIX_T]`
- `MODEL:306`: `T [FIX_T]`
- `OUTGEN`에서도 반복마다 “Temperature held fixed at all depths.”

다만 README 인용은 틀렸다. `FIX_T=F` stint 2의 실물 설명은 README 135–142가 아니라 [README.md:76](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/README.md:76)–81이다.

### 현 fixed-T run을 J/rate/pop oracle로 쓸 수 있는가

**REFUTED — 가장 중대한 결함.**

마지막 iteration의 실물은 수렴이 아니다.

- [OUTGEN:3089](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/OUTGEN:3089): 외곽 luminosity `5.80519×10^10 Lsun`, 목표 `2.60000×10^7 Lsun`
- [OUTGEN:3094](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/OUTGEN:3094): 최대 증가 `3.46×10^3%`
- [OUTGEN:3096](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/OUTGEN:3096): correction `1.0×10^7`
- 직전 반복에도 `MOM_J_REL_V9: excessive iteration count`가 반복된다.
- [CORRECTION_SUM:6](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/CORRECTION_SUM:6) 등에서 많은 active variables가 10–100% 이상 변한다.

즉 RVTJ의 “Completion of Model”은 정상 종료/출력 생성 표시일 뿐 수렴 인증이 아니다. 로드맵 14, 91, 239행의 “frozen oracle 인증”은 철회해야 한다. 이 상태의 `EDDFACTOR/PRRR/GENCOOL/pop`도 최종 oracle로 사용할 수 없다.

### 올바른 rerun 요건

**released-T rerun 필요성은 CONFIRMED하나 요건은 보강해야 한다.**

1. 먼저 `FIX_T=T` 상태에서 active-population·Jν·moment solve를 실제 수렴시킨다.
2. 이어 `FIX_T=F`로 release하고 RE/SE/charge를 재수렴한다.
3. RVTJ flag뿐 아니라 active-variable correction, SE/charge/RE residual, successive-iteration stability, moment solver 오류 0건, luminosity/energy ledger를 동시에 요구한다.
4. 공개 time-dependent CMFGEN을 acceptance target으로 삼으려면 `DO_DDT=F` released-T snapshot만으로 부족하다. time-dependent epoch sequence를 재현하거나, “steady CMFGEN self-run parity”로 목표를 명시적으로 낮춰야 한다.
5. 공개 CMFGEN과의 `T_e/n_e` 문턱은 matched-physics anchor와 scientific cross-check로 분리해야 한다.

현재 Stage 0은 “앵커 봉인”이 아니라 **기존 앵커 실격 및 재생성**부터 시작해야 한다.

---

**최종 판정: 반대.** 노선 (c)은 재작성 후 다시 심사할 수 있으나, 현 문서는 미수렴 CMFGEN run을 oracle로 인증한 치명적 오류, 다수의 Lumina file:line 오인용, steady/time-dependent 목표 혼재, 검증 불가능한 acceptance 규약 때문에 채택할 수 없다. 파일 수정, solver 실행, 테스트 실행, git 작업은 하지 않았다.