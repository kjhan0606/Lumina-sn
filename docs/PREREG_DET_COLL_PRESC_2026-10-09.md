# 사전등록 — DET-COLL-PRESC: 충돌 처방 교체에 따른 공개 CMFGEN 어긋남 비교

2026-10-09 · 읽기 전용 기안  
운전자 파일화 대상: `docs/PREREG_DET_COLL_PRESC_2026-10-09.md` — 신설  
상태: **기안. 이 계약의 기계 게이트·Fable 사전감사·구현·시험·실행·종료감사는 미집행.**

## §0. Q-GOAL·Q-LEAN 자기답과 과학 질문

**Q-GOAL — 최종목적에 직접 기여하는가? 자기답: PASS.**

최종목적은 CMFGEN과 비교 가능한 SN Ia 이온화 상태와 스펙트럼을 재현하는 것이다. 이번 단은 기준선 N과 같은 바이너리·입력·12회 반복에서 충돌 처방만 바꾸어, **공개 CMFGEN 대비 S II/III·Si II/III 분율과 UV·광학 스펙트럼의 어긋남이 어떻게 움직이는지** 표 하나로 답한다. 이를 생략하면 충돌율 차이가 관측 가능한 비교량에 미치는 영향을 알 수 없다. 결정론적 pure-CMFGEN 레인의 결과이며 최종 MC 수송의 재현 성공은 아니다.

**Q-LEAN — 각 측정은 어디에 소비되는가? 자기답: APPROPRIATE.**

| 유지하는 측정·검사 | 소비하는 표의 칸·결정 |
|---|---|
| NLTE 이온합·원소 총량·등록 셸 대응 | 분율 20행의 CMFGEN·N·새 런 칸과 종별 개선 셸 수 |
| 여덟 대역 적분과 OPT 적분 | 색 잔차 8행, R₈, L_OPT/공개 및 이동 방향 |
| 입력·환경·반복·관측 세대 신원 | 세 열이 같은 조건의 비교인지 확인 |
| 기존 Ω 활성화 로그 한 줄 | 새 런 열을 처방 교체 결과로 해석할 자격 |
| 실제 자원 사용·원본 보존 | 등록 실행 범위 준수 여부 |

새 충돌율 전수 재계산, NRB 원천 재검증, 수렴 측도, 준위 역산, 저장 산출 재검 장치, 쓰기 결함 모사, 결정표 재계산은 추가하지 않는다. 직접시험은 §8의 음성 대조로 한정한다.

사용자 결정 근거는 `/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/evidence/USER_DECISION_2026-10-09.txt`의 addendum이다. 충돌 처방 선택과 「12회, N 대비 (추천)」 결정을 적용한다. 선행 문서의 「NRB 정상 상태 확인 뒤」를 추가 실행 조건으로 유지하지 않는다.

기준선은 **N 12회**, 새 런 O도 **12회**이며 양쪽 **관측 물질 세대 13**을 비교한다. N은 NRB가 켜져 있다. R′는 바이너리·입력 계보로만 사용한다. §9의 42 GPU·h는 N 실측에 근거한 개정 상한이며 사용자 원문의 정확한 숫자 승인으로 표시하지 않는다.

C2의 황 분율 변화 ≤0.01 dex는 다른 구성에서 Ω와 충돌 이온화를 함께 바꾼 관측이다. 작은 응답도 가능하다는 참고로만 사용하며, 이번 효과의 부호·크기·상한이나 무효과 기대치로 이전하지 않는다.

## §1. 경로·기준선·승계 사실

| 식별자 | 고정 대상 |
|---|---|
| N — 비교 기준선 | `/gpfs/kjhan/lumina/cmfgen_map/nrb_judge_20261008/NRB/run1` |
| R′ — 바이너리·입력 계보 | `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1` |
| 봉인 소스 | `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src` |
| 작업 기반 B | `/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009` |
| 새 런 O | `/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1` — 신설 |
| 공개 자료 | `data/standart_data1/toy06` |

N과 그 계보 R′의 STAGE_IDENTITY.txt에 기록된 바이너리 SHA-256은 `8ef03fb37b3e0de986f9c9ba15f0b7bbf8f98800d62ab18fb666ec518914f5d8`이다. 바이너리·빌드 신원·원자 자료·덱·기하·온도 프로파일은 그대로 사용하며 재빌드하지 않는다. NRB 판정에 기록된 N 입력 65개는 R′와 SHA-256이 같고, exports는 §2의 경로 키 7개 재설정과 반복 수 6→12만 다르다.

COLL-OMEGA 판정은 선행 오프라인 저장 표본에서 하향 충돌율 비가 2배 밖인 키를 Si III 74/86, S III 311/424로 확인했다. 그 표본은 이번 N·O의 관측 물질 세대 13 비교와 구분한다. 이 사실과 봉인 C의 선별·표 순회 재현 결과를 승계한다. 새 런의 이온화·스펙트럼 결과를 승계하는 것은 아니다.

공개 파일은 spectra_toy06_cmfgen.txt, ionfrac_si_toy06_cmfgen.txt, ionfrac_s_toy06_cmfgen.txt이며 시각은 **19.480 d**로 고정한다.

## §2. 생산 개입의 정의와 고정 범위

봉인 lumina_plasma.c 약 19264행에서 Ω 분기가 COLL_FIX 분기보다 먼저 선택된다. 따라서 **COLL_FIX를 끄지 않고 Ω 변수 하나만 추가**한다. 이는 선행 오프라인 C 대조의 Ω 설정과도 같다.

| 환경변수 | N | O |
|---|---|---|
| `LUMINA_OMEGA_CMFGEN` | 미설정, 유효값 0 | `1` |
| `LUMINA_NLTE_COLL_FIX` | `1` | `1` 유지 |
| `LUMINA_NLTE_COLL_ION` | 미설정, 유효값 0 | 미설정 유지 |
| `LUMINA_ARTIS_PARITY` | 미설정, 유효값 0 | 미설정 유지 |
| `LUMINA_OMEGA_CMFGEN_SET` | 미설정, 기본값 0.1 | 미설정 유지 |
| `LUMINA_NLTE_ASSEMBLE_GPU` | `0` | `0` 유지 |
| `LUMINA_BB_NRB`, `LUMINA_PURE_CMFGEN_ITER` | `1`, `12` | 그대로 |

허용하는 비물리적 경로 이식은 NRB의 PATH_KEYS와 같은 다음 키뿐이다.

`LUMINA_MODEL_DIR`, `LUMINA_CMFGEN_SIGMA_BF`, `LUMINA_DEPOSITION_FILE`, `LUMINA_FIXED_TE_PROFILE`, `LUMINA_IONIZATION_REFERENCE_FILE`, `LUMINA_TOPION_LEVELS_FILE`, `LUMINA_PHYSICS_COMPARISON_DIR`.

각 값에서 N 런 루트 접두만 O로 바꾸며 상대 경로는 유지한다. O의 경로 키를 N 루트로 역변환한 뒤 **N의 input/resolved_lumina.exports 대비** 환경변수의 추가·삭제·값 차이는 **Ω 하나뿐**이어야 한다. 미설정을 임의로 명시적 0으로 바꾸지 않는다. 실행 전 상속된 `LUMINA_*`를 제거하고 등록 exports만 읽는다.

개입은 **봉인 구현의 전역 CMFGEN Ω 3단 처방**이다. 표 수록 쌍, 미수록 vR·OMEGA_SET 근사, 선 없는 표 연결 추가, 중복 억제를 함께 포함한다. 다른 이온과 복사평형 준비 경로에 미치는 같은 스위치의 효과도 포함한다. Si·S 표만의 효과로 귀속하지 않는다.

봉인 구현에는 온도 선형 보간, SAME_N 미적용, 기본 OMEGA_SET=0.1 등 CMFGEN 원본과의 차이가 남는다. 이 단에서 수정하지 않는다. 충돌 이온화·추가 clamp/floor·수치 보정·원자 자료 변경은 넣지 않는다.

## §3. 판정런 전 오프라인 검증

운전자는 기존 NRB 판독기를 import하여 입력 복사·경로 이식·공개 비교 함수를 재사용한다. 선행 COLL-OMEGA의 C 대조를 다시 실행하지 않는다.

- `det_nrb_judge_stage(..., outer=12, prereg=이번 계약)`를 준비 함수로 호출한다. N과 같은 방식으로 R′ 바이너리·입력 계보에서 O를 만들어도 된다.
- 새 래퍼는 O의 exports에 Ω만 추가한 뒤 경로 키를 역변환하여 **N의 input/resolved_lumina.exports 대비** §2의 단일 변경을 검사한다. stage의 계보 exports를 비교 기준으로 삼지 않는다.
- O의 신원·exports 해시·OWNER.txt·제출 명령·반복 12회·자원 상한을 이번 계약 값으로 확정한다. NRB의 시간 상한·작업 목적은 승계하지 않는다.
- 바이너리와 입력 사본의 내용 신원은 N 및 그 계보 R′와 일치해야 한다. 출력 디렉터리는 비어 있어야 한다.
- GPU 실행 전에 §5 표의 CMFGEN·N 칸을 기존 공개 비교 함수로 채워 시각·셸·분모·대역 대응을 확인한다.

기존 판독기의 전체 `analyze`·회귀·저장 재검을 호출하지 않는다. NRB의 COLL_FIX 전용 율 재구성이나 기존 NRB 과학 판정을 Ω 런에 적용하지 않는다.

## §4. 판정런 1회의 설계

생산 모델 진입은 **한 번**, 외부 반복은 **12회**다. N과 같은 초기 입력에서 시작하며 계보 R′·기준선 N의 종단 상태를 재시작 입력으로 사용하지 않는다.

실행 인자는 등록 바이너리에 `MODEL_DIR 1 2 spectrum nlte`다. 온도 고정, 씨앗, NRB, 복사 격자, 내부 반복, 감쇠, 저장 설정은 N 그대로다.

주 비교는 양쪽 모두 **관측 물질 세대 13**이다.

- 이온 분율: physics_DET_iter0010.ion.csv와 대응 manifest의 population_generation=13.
- 해당 물질 덤프의 radiation_generation=12는 정상적인 세대 관계다.
- 스펙트럼: lumina_spectrum_det_obs.csv와 관측 캡처 manifest의 generations.material=13, generations.radiation=13, generations.iter=11.
- 마지막 추가 물질 갱신인 세대 14를 관측 스펙트럼과 결합하지 않는다.

**Ω 적용 증거는 O의 stdout.log에 있는 기존 `[OMEGA-CMFGEN] ON:` 한 줄**로 정한다. 표 수·전체 선 수·세 tier 계수를 읽어 표 수와 각 tier 계수가 양수이고 tier 합이 전체 선 수와 같으며 OMEGA_SET=0.1인지 확인한다. N에는 이 활성화 줄이 없어야 한다.

이 표지는 봉인 lumina_plasma.c 약 1097행에서 실제 Ω 준비·tier 순회 뒤 출력된다. 봉인 바이너리 신원, Ω 우선 분기, CPU 조립 유지 및 정상 완료와 결합해 적용을 확인한다. 10400 K 준비 계수이며 실제 모든 행렬 셀의 율을 추적한 증거로 확대하지 않는다.

## §5. 기대치·주 산출·공개 비교

율 비교의 정의는 \(d_C=\log_{10}(C_{\rm FIX}/C_\Omega)\)다. 같은 저장 상태에서 표 수록 쌍의 중앙값 −0.32·+0.19는 Ω 하향 율이 Si III에서 약 2.1배, S III에서 약 0.65배임을 뜻한다. 미수록 쌍의 +2.56·+1.73은 약 1/363·1/54배에 해당한다.

이는 준위쌍의 조건부 율 비교다. 추가 연결과 결합계의 응답이 있으므로 **Si II/III·S II/III 분율, 색, R₈, OPT 광도의 이동 부호와 크기는 예측하지 못한다.** 공개값 접근은 시험할 질문이며, 전부 개선하거나 일정 크기 이상 변한다는 기대를 등록하지 않는다.

등록 공간 대응은 다음과 같다. 결과를 보고 보간·최근접 방법을 바꾸지 않는다.

| 셸 | Lumina vel_mid, km/s | 공개 vel_mid, km/s |
|---:|---:|---:|
| 10 | 11544 | 11534.59 |
| 15 | 15184 | 15320.31 |
| 19 | 18096 | 18264.10 |
| 24 | 21736 | 21629.71 |
| 30 | 26104 | 25881.39 |

분율은 \(f_{Z,q}=n^{\rm NLTE}_{Z,q}/n^{\rm element}_Z\), \(q=1,2\)다. 분자는 Z14·Z16의 nlte_q1_cm3·nlte_q2_cm3, 분모는 같은 행의 element_total_cm3를 사용한다. SAHA 열이나 II+III 재정규화를 사용하지 않는다.

공개 II·III는 si1·si2, s1·s2다. 이 비교는 공개 헤더 si0…si4·s0…s4를 I…V로 읽는 기존 명명 가정에 조건부다.

스펙트럼은 기존 함수의 경계 선형 보간·사다리꼴 적분을 사용한다. 여덟 대역은 **2500–3000, 3000–3500, 3500–4000, 4000–5000, 5000–6000, 6000–7000, 7000–8000, 8000–10000 Å**, OPT는 **4000–6000 Å**다. 외삽·시각 대체·중복 파장 허용은 하지 않는다.

\[
d^f=\log_{10}(f/f_{\rm public}),\quad
C_b=L_b/L_{\rm OPT},\quad d_b=\log_{10}(C_b/C_b^{\rm public}),\quad
R_8=\sqrt{\tfrac18\sum_b d_b^2},\quad A_{\rm OPT}=L_{\rm OPT}/L_{\rm OPT}^{\rm public}.
\]

**주 산출은 아래 형식의 CMFGEN 어긋남 표 하나**다. 항목 식별 열 뒤의 자료 열은 CMFGEN·N·O로 고정한다. 분율 20행·색 8행·요약 2행을 빠짐없이 낸다.

| 항목 | CMFGEN | N, 12회·세대 13 | O, 12회·세대 13 |
|---|---|---|---|
| S II·S III·Si II·Si III 각각 × 등록 다섯 셸 | 공개 분율 | 분율, \(d^f\) | 분율, \(d^f\), N 대비 이동 |
| 여덟 대역 각각 | \(C_b^{public}\), d=0 | \(C_b,d_b\) | \(C_b,d_b\), N 대비 이동 |
| R₈ | 0 | R₈ | R₈, N 대비 차이 |
| L_OPT/공개 | 1 | \(A_{\rm OPT}\) | \(A_{\rm OPT}\), N 대비 이동 |

OPT 행에는 세 열의 L_OPT 절대값도 병기한다. 기계 표에는 기존 함수가 반환한 대역 광도와 수치 구간을 함께 보존하여 색 정규화의 분모 효과를 확인할 수 있게 한다. 추가 과학 표나 III/II 판정은 만들지 않는다.

## §6. 수치 기준·과학 코드·실행 분기

각 이온·셸과 색 대역의 질문은 **\(|d_O|<|d_N|\)인가**다. R₈는 직접 크기를 비교한다. OPT는 \(|\log_{10}A_{\rm OPT}|\)가 작아졌는지로 공개값 접근을 표시한다.

기존 함수의 표시 정밀도·부동소수 오차 구간을 사용한다. 이는 관측 오차나 물리 모형 오차가 아니다. 중앙값 차이도 표에 남기되, 구간이 겹치면 확정 개선으로 세지 않는다.

| 코드 | 의미 |
|---|---|
| 0 | 새 오차 구간의 상한 < N 오차 구간의 하한: 개선 |
| 1 | 새 오차 구간의 하한 > N 오차 구간의 상한: 악화; 또는 비교 원량이 정확히 같음: 동일 |
| 4 | 비교는 정의되지만 개선·악화·정확한 동일을 분리하지 못함 |
| 5 | 결손·영 분모·비유한값·비양의 로그 인수 등으로 비교 미정의 |
| X | 환경·신원·세대·셸 대응이 계약과 달라 비교 부적격 |

정확한 동일은 동일 분율 또는 동일한 비교 스펙트럼 원량으로 확인되는 경우에만 쓴다. 반올림된 표시값의 같음은 근거가 아니다. 영·결손을 작은 양수로 대체하지 않는다.

종별로 **개선·악화·동일·미분리·미정의 셸 수를 분모 5와 함께** 보고한다. 스펙트럼은 여덟 대역의 같은 계수와 R₈ 변화, OPT 접근 여부를 보고한다. 종·대역 사이의 개선과 악화를 가중 합산해 하나의 성공 점수로 만들지 않는다.

## §7. 최소 변경집합·출력

### 변경집합 — DET-COLL-PRESC (표)

| 파일 | 허용 변경 |
|---|---|
| `scripts/det_coll_presc_readout.py` | 신설: 환경 차이 검사·Ω 증거·어긋남 표와 얇은 실행 연결 |
| `/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/tools/run_det_coll_presc_gpu.slurm` | 신설: NRB 러너 형태 승계, 반복·환경·자원·작업 신원만 이번 계약에 맞춤; 승계 selftest 블록(이번 반복·시간 상수와 Ω off·COLL_ION 추가 거부 사례 포함)은 「승계 확인」으로만 기록하고 등록 게이트·직접시험 수에 넣지 않는다. |
| `/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/tools/run_det_coll_presc_offline.slurm` | 신설: 준비·판독을 a10 CPU 작업으로 연결 |

### 변경집합 끝

생산 소스·바이너리·기존 판독기·기존 러너 변경은 **0**이다.

이온은 `det_nrb_judge_public_block`, `det_nrb_judge_ion_dump`, `det_nrb_judge_ions`를 재사용한다. 스펙트럼은 NRB가 사용하는 `det_siiii_repair_public`, `det_siiii_repair_spec`, `det_siiii_repair_spectrum_table`를 재사용한다. 기존 절대 목표의 통과·반증 반환값은 이번 판정에 사용하지 않는다.

준비·판독 CLI는 각각 `prepare`, `analyze`로 한다. prepare는 직접시험·기준선 판독·stage를 수행하고, analyze는 O를 더해 표를 완성한다.

과학 표의 기계 원본은 cmfgen_comparison.csv, 사용자 표시본은 같은 행의 cmfgen_comparison.md다. 실행 신원·시험·게이트·비용은 evidence.json에 모은다. 별도 decision.json이나 저장 표를 다시 읽어 결정을 재계산하는 장치는 만들지 않는다.

## §8. 게이트·직접시험·종료감사 범위

| 게이트 | 검사와 막는 실제 실패 |
|---|---|
| G-DOC | 기존 계약·참조·정본·발주·러너 위생 검사를 적용한다. 다른 계약·없는 참조·잘못된 러너로 집행하는 실패를 막는다. |
| G-LINK | §1–3의 입력 신원·정확한 환경 차이·빈 새 런을 확인한다. Ω 외 물리 변경, 잘못된 입력, 기존 런 재사용을 막는다. |
| G-RUN | §4·9의 반복·정상 완료·관측 세대·할당·비용을 확인한다. 미완료 상태나 세대 14를 12회 관측 결과로 보고하는 실패를 막는다. |
| G-OMEGA | §4의 활성화 줄을 확인한다. G-RUN과 별도로 유지한다. exports만 켜고 적용되지 않은 런을 처방 효과로 해석하는 실패를 막는다. |
| G-TABLE | §5의 행 키·세대·공개 시각·vel_mid·분모·전체 대역을 확인한다. 다른 셸·이온 열을 결합하거나 불리한 행을 누락하는 실패를 막는다. |

새 직접시험은 정상 소형 입력을 바탕으로 다음 음성 대조만 수행하고 실제 종료 코드와 거부 사유를 기록한다.

1. 허용 Ω 변경에 `LUMINA_NLTE_COLL_ION=1`을 추가하면 환경 차이 검사에서 거부한다.
2. Ω 활성화 줄을 제거하면 적용 증거 검사에서 거부한다.
3. 표 조립 입력의 셸–공개 vel_mid 대응을 바꾸면 셸 대응 검사에서 거부한다. 승계 함수의 vel_mid 거부와 달리 새 조립 검사는 반환된 표 행의 런×종×셸 키 전수·중복, 물질 세대 13과 §5의 등록 공개 vel_mid를 대조한다.

승계 함수의 시험을 복제하지 않는다. 쓰기 결함·저장 재검·결정 재계산 시험을 추가하지 않는다.

**종료감사의 1차 대상은 과학 표의 수치·셸 대응·주장 한정이다.** 판독기 내부 구조는 과학 수치·행 귀속·판정에 영향을 줄 수 있는 구체적 결함만 «폐합 차단»으로 삼는다. 추상적인 방어 강화나 내부 구조 선호는 관측으로 남긴다. 수리는 해당 결함과 영향 행에 한정한다.

## §9. 환경·비용·재현 명령

| 항목 | 등록 상한·규칙 |
|---|---|
| GPU 실행 | Slurm job-per-run, 모델 진입 1회, 외부 반복 12회 |
| GPU 배분 | h200 ×2 우선, 실행 전 h100 ×2 대체 가능; a40·a100 제외 |
| GPU 작업 CPU·메모리 | 16 CPU·64 GiB |
| 누적 GPU 할당 경과 | **75,600 s = 21:00:00** |
| 누적 GPU 비용 | **42 GPU·h** |
| 재실행·재시작·requeue | **0회** |
| 오프라인 | a10, 작업당 2 CPU·8 GiB·GPU 0, 동시 1작업 |
| 오프라인 누적 상한 | **600 s·1/3 core·h**, 시험·stage·게이트·판독·감사 재현 포함 |

N 실측은 Slurm 할당 경과 67,484 s × GPU 2장 = 37.4911 GPU·h이며, RUN_FOOTER.txt의 모델 경과는 67,474 s다. 비용 근거는 `/gpfs/kjhan/lumina/cmfgen_map/nrb_judge_20261008/evidence/sacct_gpu_416610.txt`다. 등록 상한 75,600 s·42 GPU·h는 이 실측보다 약 12.0% 여유를 둔 값이다. 누계는 이번 O의 할당에 적용하며 N의 기존 비용은 포함하지 않는다. Slurm 전체 할당 경과로 비용을 계산하며 h100에서도 상한을 늘리지 않는다. 대기시간은 별도 보고하고 할당 경과에 합산하지 않는다.

작업명은 `LUMINA_` 접두, 제외 노드는 `--exclude=syn102,syn03`, 런 루트에는 OWNER.txt를 둔다. module purge·load와 실제 버전을 기록한다. GPU 모듈은 N의 gnu13/13.2.0·cuda/13.0.2를 승계한다.

이번 O의 Slurm 시간은 **21:00:00(정수 1,260분)**으로 제출하고 실제 저장된 TimeLimit이 **75,600 s 이하**인지 확인한다. 할당 전 대기 작업의 취소·대체 제출만 허용하며 동시 대기 사본을 두지 않는다. 할당 후 실패하면 원인을 보고하고 추가 GPU 제출은 하지 않는다.

다음은 구현할 러너의 재현 인터페이스이며 기안 단계에서 실행하지 않았다. prepare는 h200x2 또는 h100x2를 받아 §3을 완료한다. 각 작업 완료와 게이트 통과 후 다음 명령을 실행한다.

```bash
COLL_BASE=/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009
sbatch --exclude=syn102,syn03 --time=00:05:00 "$COLL_BASE/tools/run_det_coll_presc_offline.slurm" prepare p1 h200x2
sbatch --exclude=syn102,syn03 --time=21:00:00 "$COLL_BASE/OMEGA/run1/job.slurm" "$COLL_BASE/OMEGA/run1"
sbatch --exclude=syn102,syn03 --time=00:03:00 "$COLL_BASE/tools/run_det_coll_presc_offline.slurm" analyze a1
```

위 CPU 요청 합계 외 남은 120 s는 문서 게이트·필요한 재현에 배정한다. 실제 소비를 누계하며 실패도 포함한다. 상태 조회는 60초보다 자주 하지 않는다. 원본은 읽기만 하고 새 경로에 일반 파일 사본을 만들며 기존 산출 덮어쓰기를 거부한다.

## §10. 처분·완료 조건·후속 결정

완료 보고의 첫 산출은 §5의 표다. 기준선 열(N)은 기존 자료의 같은 파이프라인 재산출이고 새 정보는 O 열과 이동 열이다. 판정 문장은 그 표에서 다음 순서로 작성한다.

1. S II·S III·Si II·Si III 각각에서 공개값에 가까워진 셸 수와 악화·동일·미분리·미정의 수.
2. R₈의 N→O 값과 변화 방향, 여덟 대역 중 개선·악화한 대역.
3. L_OPT/공개의 변화와 공개 절대 광도 접근 여부.
4. 혼합 결과 또는 변화 미분리라면 그 사실과 해당 행을 명시.

적격 실행과 표가 갖춰지면 개선 여부와 무관하게 **PASS_LIMITED**로 결과 보고를 완료할 수 있다. 이는 처방의 우월성 판정이 아니다. 실행 미완료는 `INCOMPLETE`, 비교 신원 오류는 `INADMISSIBLE`로 보고하며 유효하지 않은 열에서 효과 결론을 내리지 않는다.

**12회·세대 13의 유한 반복 상태 비교다.** 기준선 N은 선행 NRB 판정에서 `NOT_CONVERGED`였으며 O의 수렴 여부는 이번 단에서 판정하지 않는다. 생산 승격·기본값 변경·물리적 정답·수렴·씨앗 독립성·특정 이온 또는 경로의 인과 기여를 선취하지 않는다. 추가 반복·씨앗·GPU 실행과 물리 변경은 이 결과를 본 뒤 사용자가 결정한다.

## §11. 집행 기록·분장

| 업무 | 담당·현재 상태 |
|---|---|
| 기안·개정 | 발주상 astra; 이번 응답의 식별 상태는 첫 줄에 기재 |
| 경량 사전감사 | Fable fresh, Q-GOAL·Q-LEAN 전용 1회, 미집행 |
| 구현·직접시험·게이트·실행·판독·판정 | 운전자, 미집행 |
| 종료감사 | astra, 과학 표 중심, 미집행 |

GPU 판정런이므로 Fable 사전감사를 생략하지 않는다. 기안자 자기답을 감사 통과로 표시하지 않는다. 운전자 구현·판정 겸임과 계약 저자의 종료감사 이해관계를 최종 기록에 남긴다.

기안자는 공유 맥락·현재 작업 디렉터리·분장·지정 계약과 판정문·봉인 소스·R′/N 신원·재사용 함수를 읽었다. 파일 변경·계산·기계 게이트 실행은 하지 않았다. 운전자는 이 절에 실제 모델 ID, 파일 신원, 작업 ID, 게이트 결과, 비용, 이탈과 종료감사 결과를 이어 기록한다.

### 집행 기록 (운전자 추가 — 정본 장부, 개정16 ①; 위 기안 기록은 그대로 둔다)

증거 접두 `B=/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009`. 작업 번호·경과는 `B/JOBS.txt`·`B/evidence/COST_LEDGER.txt`.

| 항목 | 담당(실제) | 상태·증거 |
|---|---|---|
| 기안·개정 1 | astra `gpt-6-astra` | ef4c0f1 · 0ab3428(OLD/NEW 31쌍, 행 단위 의도 외 소실 0) |
| 기계 프리플라이트 | 운전자 | 기안·개정 1 모두 prereg·referents·canonical rc 0(a10 417556·417586), 개정 발주 프리플라이트 rc 0(417583) |
| Fable fresh 경량 사전감사 | Fable `claude-fable-5-1` | Q-GOAL 부합·Q-LEAN 적정·«폐합 차단» 0·«기재로 족함» 5(`B/fable_preaudit.md`) |
| user 결정(G-1) | user | 「12회, N 대비 (추천)」(`B/evidence/USER_DECISION_2026-10-09.txt` addendum) |
| 구현·직접시험 | 운전자(Opus 5.5) | 판독기 cbce1e0, 러너 둘(위생 rc 0, 417587); dev1 417557 실패(픽스처 TypeError) → prepare p1 417588 직접시험 4/4 |
| staging | 운전자 | p1 → `B/OMEGA/run1`(입력 65 파일, N 대비 exports 차이 = Ω 한 줄) |
| GPU 판정런 | 운전자 | 417590 h200x2 `--time=21:00:00`(저장 21:00:00), 10-09 15:29:19–10-10 10:15:43, SUCCESS, 37.55 GPU·h |
| 판독 | 운전자 | a1 a10 418282 VALID(`B/results/a1/`), 게이트 발견 0 |
| 판정 | 운전자(구현·판정 겸임) | `docs/VERDICT_DET_COLL_PRESC_2026-10-09.md` PASS_LIMITED |
| 게이트 roster | 운전자 | `B/evidence/GATE_RECORD_a1.json` |
| 종료감사 | astra(계약 저자 이해관계) | 대기 |
| 비용 | 운전자 | GPU 37.55 / 42 GPU·h, 모델 진입 1/1; CPU 전용 147 s / 600 s(a1 까지, 이후 게이트는 장부에 누적) |

## §12. 기계 선언

직전 계약의 changeset·branches·references 형식을 사용한다. 분기 코드는 §6의 이동 판정이며 기존 NRB 명제 코드를 승계하지 않는다. 참조 목록에는 실재 파일만 넣는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-COLL-PRESC (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_coll_presc_readout\\.py|/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/tools/run_det_coll_presc_(?:offline|gpu)\\.slurm)",
    "symbol": "det_coll_presc_",
    "roots": [
      "scripts/det_coll_presc_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/tools/run_det_coll_presc_gpu.slurm",
      "/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/tools/run_det_coll_presc_offline.slurm"
    ],
    "expected_extra": [
      "scripts/det_coll_presc_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/tools/run_det_coll_presc_gpu.slurm",
      "/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/tools/run_det_coll_presc_offline.slurm"
    ]
  },
  "branches": {
    "regimes": [[0,0],[1,1],[2,2],[4,4],[5,5],[9,9]],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,4,5) for x in v)",
      "c0": "sum(1 for x in v if x == 0)",
      "c1": "sum(1 for x in v if x == 1)",
      "c4": "sum(1 for x in v if x == 4)",
      "c5": "sum(1 for x in v if x == 5)"
    },
    "rules": [
      {"name":"X","predicate":"not valid"},
      {"name":"TOWARD","predicate":"valid and c0 == 1"},
      {"name":"NOT_TOWARD","predicate":"valid and c1 == 1"},
      {"name":"UNRESOLVED","predicate":"valid and c4 == 1"},
      {"name":"NOT_IDENTIFIABLE","predicate":"valid and c5 == 1"}
    ],
    "adversarial_fixtures": [
      {"name":"toward","mix":[[0,1]]},
      {"name":"not-toward","mix":[[1,1]]},
      {"name":"unresolved","mix":[[4,1]]},
      {"name":"undefined","mix":[[5,1]]},
      {"name":"invalid","mix":[[2,1]]},
      {"name":"missing","mix":[]},
      {"name":"multiple","mix":[[0,1],[1,1]]}
    ]
  },
  "references": [
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/evidence/USER_DECISION_2026-10-09.txt"},
    {"path":"docs/PREREG_DET_NRB_JUDGE_2026-10-08.md"},
    {"path":"docs/VERDICT_DET_NRB_JUDGE_2026-10-08.md"},
    {"path":"docs/VERDICT_DET_COLL_OMEGA_OFFLINE_2026-10-08.md"},
    {"path":"scripts/det_coll_omega_offline_readout.py"},
    {"path":"scripts/det_nrb_judge_readout.py","flags_existing":["det_nrb_judge_ions","det_nrb_judge_spectrum","det_nrb_judge_stage"]},
    {"path":"scripts/det_siiii_repair_readout.py","flags_existing":["det_siiii_repair_spectrum_table"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1/STAGE_IDENTITY.txt","flags_existing":["run_binary_sha256"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1/input/resolved_lumina.exports"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/nrb_judge_20261008/NRB/run1/STAGE_IDENTITY.txt","flags_existing":["outer_iterations=12","run_binary_sha256"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/nrb_judge_20261008/NRB/run1/RUN_FOOTER.txt","flags_existing":["outer_iterations=12","elapsed_seconds=67474","allocated_gpus=2","terminal_state=SUCCESS"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/nrb_judge_20261008/NRB/run1/input/resolved_lumina.exports"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/nrb_judge_20261008/NRB/run1/work/physics_comparison/physics_DET_iter0010.manifest.json"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/nrb_judge_20261008/NRB/run1/work/det_obs_capture/manifest.json"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/nrb_judge_20261008/NRB/run1/stdout.log"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/nrb_judge_20261008/evidence/sacct_gpu_416610.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/nrb_judge_20261008/tools/run_nrb_judge_gpu.slurm"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/nrb_judge_20261008/tools/run_nrb_judge_offline.slurm"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c","flags_existing":["LUMINA_OMEGA_CMFGEN","LUMINA_NLTE_COLL_FIX","LUMINA_NLTE_COLL_ION","[OMEGA-CMFGEN] ON:"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_nlte_assemble.cu","flags_existing":["LUMINA_OMEGA_CMFGEN"]},
    {"path":"data/standart_data1/toy06/spectra_toy06_cmfgen.txt"},
    {"path":"data/standart_data1/toy06/ionfrac_si_toy06_cmfgen.txt","flags_existing":["si1","si2","19.480"]},
    {"path":"data/standart_data1/toy06/ionfrac_s_toy06_cmfgen.txt","flags_existing":["s1","s2","19.480"]}
  ]
}
```
