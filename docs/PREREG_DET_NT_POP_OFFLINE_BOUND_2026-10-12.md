# 사전등록 - DET-NT-POP-OFFLINE-BOUND: 저장 상태에서 비열적 전리의 조건부 상한과 인구 계측 필요성

2026-10-12 · astra 읽기 전용 기안  
상태: 지정 원자료와 소스 조회 완료. 파일 수정·구현·시험·계산·제출·판정 미집행.

## §0. Q-GOAL·Q-LEAN 자기답

**Q-GOAL: PASS.** 최종목적은 공개 CMFGEN과 비교 가능한 SN Ia의 절대 광도와 스펙트럼을 얻는 것이다. 이번 단은 침적 소비와 인구 권위 연결 중 **현재 NT 전리 처방이 철족 II의 광전리에 비해 작다고 저장자료만으로 제한할 수 있는가**를 가른다. 이를 생략하면 이미 저장된 정보로 좁힐 수 있는 대상을 GPU로 다시 기록하거나, 원자당 율을 실제 소비 전력으로 오인할 수 있다.

종료 산출은 철족 II의 셸별 조건부 율 상한·전력 상한·미식별 소비량을 담은 표 하나다. 이 표로 NT 인구 가중을 직접 기록하는 후속 GPU 계측이 필요한지와 최소 기록 대상을 결정한다. 전체 침적 닫힘이나 청색 결손의 원인 판정은 이번 질문이 아니다.

**Q-LEAN: APPROPRIATE.** 물리 상태를 바꾸지 않고 저장 상태 하나를 판독한다. 기존 침적 발행 판독, 원자 단면적 판독, 구간 산술과 Si 소비 trace를 최대한 재사용한다. 신규 계산은 바닥준위 광전리 적분과 조건부 상한뿐이다.

생산 변경, GPU, Spencer-Fano 새 해법, 인구 재해, 원천 주입, 관측 재생, 새 광도 적분, 전 원소 순위표, 저장 산출 재검 장치, 쓰기 결함 모사, 별도 결정표 재계산 장치를 제외한다. 보조 산출은 주 표의 입력 신원·상한 전제·미정의·시험·비용만 뒷받침한다.

## §1. 권한·기반·보존

기안·개정·모든 감사는 astra, 구현·직접시험·게이트·실행·판정은 운전자다. GPU 0·생산 변경 0이므로 경량 사전감사를 생략한다(25-ab). 코드 감사는 없다(25-z). 종료감사는 astra이며, 자기 계약에 따른 판정을 감사한다는 이해관계를 명시한다. 현재 직접 발주가 과거 분장보다 우선한다.

| 약칭 | 위치 |
|---|---|
| P | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn` |
| B | `/gpfs/kjhan/lumina/cmfgen_map/ntcons_20261012` |
| F | `/gpfs/kjhan/lumina/cmfgen_map/ntcons_20261012/offline` |
| O | `/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1` |
| S | `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src` |
| D11 | `/gpfs/kjhan/lumina/cmfgen_map/diag11_bluecause_20261012` |
| T | `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010` |
| C | `/gpfs/kjhan/cmfgen_src/cur_cmf` |
| U | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06` |

권한 근거는 B의 evidence에 있는 `USER_DECISION_2026-10-12.txt`다. 선택은 계약화와 오프라인 선행이다. 선택지의 철족 바닥율·여기 규모 예시는 운전자 미감사 문구이며 요구나 확인 사실로 승계하지 않는다.

생산 소스, O와 T의 런, 원자료, 선행 산출은 수정하지 않는다. 본 단의 신규 파일은 F 아래에만 둔다. 명시 입력과 재사용 코드의 크기·SHA-256을 기록한다. 전체 트리 복사·해시, 하드링크, 파일시스템 고유 기능은 사용하지 않는다. 실행마다 새 출력 위치를 사용하고 실패 산출도 보존한다.

## §2. 저장자료로 가를 질문과 생산 기록이 필요한 질문

> O의 물질13·복사12·T12 저장 표현에서, 현재 기본 pair 처방의 철족 II NT 전리는 같은 바닥준위의 광전리보다 얼마나 크며, 비음성 준위 인구에 대해 전체 II 광전리 대비 NT 전리의 조건부 상한을 줄 수 있는가?

표적은 Fe·Co·Ni의 II에서 III로 가는 기본 pair, O의 셸 0-49다. 원소나 II가 없는 셸도 행을 유지한다. 대상 원소 확대, III 이상 사다리 전개, 시각 탐색은 하지 않는다.

| 저장자료만으로 가능한 것 | 생산 상태 기록이 있어야 가능한 것 |
|---|---|
| 침적 발행 배열과 입력의 연결, 원자당 NT 계수 복원 | 각 조립 호출에서 실제 소비한 NT 계수와 인구 |
| 저장 J·단면적의 바닥준위 광전리 계수 | 철족의 준위별 생산 광전리 소비와 유효 경로 |
| 기본 pair 가정 아래 전체 광전리 대비 NT 전리 상한 | 제약행 전후 NT 항 보존, 채택 후보의 실제 인구 가중 |
| 저장 NLTE II 총량을 사용한 전리 전력 상한 | 실제 전리·여기 전력과 자유전자 열 배분 |
| 공개 파일의 단위·시각·공개량 범위 | 공개 CMFGEN의 내부 J·준위 인구·채널 소비 전력 |
| 후속 계측에서 필요한 항목과 대상의 축소 | 같은 상태의 총 에너지 닫힘·고정 T 외부 교환 |

율 상한이 작아도 가열·여기·시간 항 후보를 기각하지 않는다. 여기 소비의 물리적 결손 규모는 저장자료로 식별되지 않는다. 이를 0이나 침적 잔액으로 채우는 계산은 등록하지 않는다.

## §3. 잣대·세대·인구 권위

주 상태는 O의 physics_comparison에 있는 `physics_DET_iter0010`의 manifest·shell·ion·spectral 자료다.

| 자료 | 확인한 세대와 의미 |
|---|---|
| iter0010 덤프 | 물질13·복사12·T12, COMMITTED·FIXED_T |
| 관측 포획 | 물질13·복사13·iter11 |
| iter0011 덤프 | 물질14·복사13·T13 |
| 최종 인구 | 물질14 |
| 침적 발행 | generation 1·epoch 1683072 s, EXTERNAL_FILE |
| 채널 장부의 gen | 단계 sweep 자체 counter이며 물질 세대가 아님 |

주 계산은 iter0010의 J를 사용한다. 관측 포획 J나 최종 인구로 바꾸지 않는다. 침적 세대 번호를 인구 세대 번호와 같게 만들지 않고 epoch와 발행 신원으로 연결한다.

단계 II 밀도는 단계 해의 권위이고, NLTE II 밀도는 해당 준위 합의 권위다. 둘은 같은 물질 세대에서도 다르다. 전력 상한에는 NLTE II 합만 사용하며 단계 합을 대신 넣지 않는다. 단계 밀도는 주 표에서 권위 구별용으로만 표시한다.

저장 spectral J는 `physics_comparison.c` 약 42행의 보존 재격자 결과다. 따라서 계산량의 기본 등급은 **STORED_REPRESENTATION_CONDITIONAL**이다. 같은 세대 태그만으로 내부 반복·조립 시점·생산 계수·수렴이 같다고 주장하지 않는다.

기안 조회에서 확인한 배선은 다음과 같다.

- `lumina_cuda.cu` 약 7930행의 외부 침적 등록은 약 8093행의 DET 진입보다 앞선다.
- `lumina_plasma.c` 약 23919행은 O 설정에서 \(0.05q_{\rm dep}/(35\,{\rm eV})\)를 발행한다. `NT_FULL=0`은 NT 꺼짐이 아니다.
- 같은 파일 약 3943행은 원자 총밀도로 나눈 NT 계수를 단계 전리에 넣고, 약 20423행은 기본 NLTE pair의 SL 0에서 상위 바닥으로 가산한다.
- 원자 덱의 철족 SL 0은 여러 미세구조 준위를 포함하지만, O의 `LUMINA_SUPER_CUTOFF=100`이 이를 덮어쓴다. `lumina_atomic.c` 약 1460행과 `lumina_plasma.c` 약 18030행의 투영에 따라 유효 SL 0은 준위 0 단독이다. 덱의 super_level 열만으로 소비 대상을 정하지 않는다.
- `radeq_publication.c` 약 210행은 침적 전량을 GAMMA에 기재하고 별도 NONTHERMAL 열 항을 EXACT_ZERO로 둔다. 이것은 NT 전리 소비 전력 0의 증거가 아니다.

위 소스·설정·덱 연결을 계산 시 확인한다. 확인되지 않는 원소의 상한은 미정의이며 다른 원소에 전파하지 않는다.

## §4. 비교 율과 조건부 전력

셸 \(s\), 원소 \(Z\), 바닥준위 \(g\)에 대해 다음을 정의한다.

\[
V_s=\frac{4\pi}{3}(r_{{out},s}^{3}-r_{{in},s}^{3}),\quad
D_s=V_sq_{{dep},s},\quad
R_{{NT},s}=\frac{0.05q_{{dep},s}}{35\,{\rm eV}},\quad
\gamma_{{NT},s}=\frac{R_{{NT},s}}{n_{{atom},s}}.
\]

\(q_{\rm dep}\)는 erg/s/cm3, \(R_{\rm NT}\)는 1/s/cm3, \(\gamma_{\rm NT}\)는 s^-1, \(D\)는 erg/s다. eV 변환은 봉인 상수를 사용한다. \(n_{\rm atom}\)은 shell 자료의 전체 원자 수밀도이며 전자밀도·원소밀도·II 밀도로 대체하지 않는다.

발행 복원은 diag11의 manifest 규약을 재사용한다. 0.05는 코드 계수이며, 닫힌 물리 에너지 분할의 5%로 해석하지 않는다. 35 eV도 실제 각 전이의 에너지로 대입하지 않는다.

바닥준위 광전리 계수는 저장 구간 표현에 대해
\[
\gamma_{{ph},g,s}^{*}
 =4\pi\sum_b J_{sb}
 \int_{\nu_{b,-}}^{\nu_{b,+}}
 \frac{\sigma_g^{*}(\nu)}{h\nu}\,d\nu
\]
로 정의한다. 단위는 s^-1이다. 별표는 저장 표현의 조건부 계산을 뜻한다.

단면적은 O 덱의 `cmfgen_sigma_bf.bin`, 준위 키와 문턱은 `levels.csv`·`ionization_energies.csv`에서 얻는다. 원소·이온·준위 키와 전역 행 대응을 확인한다. 없는 단면적을 Kramers나 다른 원소로 대체하지 않는다.

`bf_rate_jnu.c` 약 131행의 저장 단면적 규약을 따른다. 문턱을 가로지르는 bin의 값은 전체 bin 평균이므로 활성 구간으로 옮길 때 전체 폭/활성 폭을 곱한다. 이 보정 뒤 문턱 아래는 적분하지 않는다. 같은 파일 약 23행의 구간 적분과 대응하며, 중심 주파수 합이나 문턱 질량의 이중 절단을 사용하지 않는다. 지원 밖을 외삽하지 않는다.

\[
B_{Zs}=\gamma_{{NT},s}/\gamma_{{ph},g,s}^{*}.
\]
기본 pair의 유효 NT 출발점이 바닥준위 단독이고, 모든 준위 인구와 광전리 계수가 비음수이면
\[
F_{\rm NT}^{*}=n_g\gamma_{\rm NT}
 \le B_{Zs}\sum_{i\in {\rm II}}n_i\gamma_{{ph},i}^{*}
 =B_{Zs}F_{\rm ph}^{*}.
\]
따라서 \(B\)는 그 조건에서 전체 II 광전리 대비 NT 전리의 상한 계수다. \(n_g=0\)이면 NT 전리는 0이지만 실제 율비의 분모까지 0이면 실제 비는 미정의다. 큰 \(B\)는 큰 실제 소비를 뜻하지 않는다.

저장 NLTE II 총밀도를 \(N_{{II},Zs}\), 바닥에서 III 바닥까지의 전리 에너지를 \(\chi_Z>0\)라 하면
\[
0\le P_{{NT,ion},Zs}^{*}
 \le P^{\max}_{Zs}=V_sN_{{II},Zs}\gamma_{{NT},s}\chi_Z.
\]
이는 내부 전리 에너지 전력의 조건부 상한이다. 실제 \(n_g\)가 없으므로 실제 전력은 미식별로 남긴다. \(P^{\max}/D_s\)는 상한 규모이며 열화 분율이나 광도 회복률이 아니다. \(D_s=0\)이면 이 비는 미정의다.

단계 진단 전력과 NLTE 전력을 더하지 않는다. 상한을 실제 소비로 더하지 않는다. 원소별 상한 합도 본 단의 새 총에너지 장부로 만들지 않는다.

## §5. 에너지식과 CMFGEN 대응

총 물질 에너지식은 내부 전리·여기 에너지를 포함한
\[
\rho De/Dt-(P/\rho)D\rho/Dt=Q_{\rm rad}+q_{\rm dep}
\]
와 자유전자 열식을 구별한다. 비열적 전리·여기로 옮긴 에너지는 자유전자 열 배분에서 차감되는 항이지만 총식에 침적을 다시 추가 청구하는 독립 원천이 아니다. \(D+P_{\rm ion}+P_{\rm exc}\)를 총 공급으로 쓰지 않는다.

완전한 비열적 분해와 다른 손실 항을 확인하기 전에는 \(D-P_{\rm ion}-P_{\rm exc}\)도 실제 자유전자 가열로 확정하지 않는다. 고정 T의 외부 교환, 시간 항, 순경계가 없는 저장 장부로 완전 닫힘을 선언하지 않는다.

| CMFGEN 대응 | 직접 확인한 소스와 범위 |
|---|---|
| 전자 분포와 인구 소비 호출 | C의 new_main에 있는 `cmfgen_sub.f` 약 1690행 |
| 전자 가열·전리·여기 분율 | C의 non_therm에 있는 `electron_non_therm_spec.f90` 약 355·543·617·633행 |
| 총식 침적 가산 | 같은 non_therm의 `se_ba_non_therm_v3.f90` 약 87행, STEQ_T |
| 인구 가중 전리와 전리 에너지 | 같은 파일 약 118-161행 |
| 비열적 여기와 에너지 | 같은 파일 약 202-240행 |
| 자유전자식 NT 냉각 차감 | 같은 파일 약 241행, STEQ_T_EHB |
| 자유전자식 침적 가산 | `cmfgen_sub.f` 약 2972행 |

공개 계산은 비국소 감마선 MC 침적, Spencer-Fano 비열적 전리·여기, 시간의존 인구와 에너지식 결합을 사용한다. 총 에너지식과 자유전자 열식도 별도로 설명한다. [StaNdaRT 논문 3.2절](https://arxiv.org/html/2209.11671v2#S3.SS2)

U에서 확인 가능한 것은 공개 시각별 침적, T·밀도·전자밀도, 이온 분율, 광도와 스펙트럼이다. 공개 내부 J·준위 인구·채널별 NT 전력은 이번 자료에 없다. 로컬 CMFGEN 소스와 19.480 d 공개 실행의 정확한 판본 연결도 미확인이다. 따라서 이번 \(B\)와 \(P^{\max}\)를 실제 CMFGEN 율이나 전력의 비로 표시하지 않는다.

## §6. 주 표와 최소 재사용

주산출은 `nt_population_bound.csv` 하나다. 원소 3개와 셸 50개의 조합을 유지하며, 표시본은 같은 메모리 행에서 만든다. 별도 과학 요약표를 생성하지 않는다.

각 행은 다음 열군을 가진다.

| 열군 | 내용 |
|---|---|
| 신원 | 원소·셸·II/III·물질/복사/T 세대·침적 세대/epoch·표현 등급 |
| 상태 | 원소 총량·단계 II·NLTE II·n_atom·V·q_dep·D |
| 연결 | 바닥 전역 키·유효 SL 구성·단면적 존재·문턱·저장 지원·전제 상태 |
| 율 | gamma_NT·gamma_ph_ground·B의 중심값과 수치 구간 |
| 전력 | P_max·P_max/D의 중심값과 수치 구간 |
| 미식별 소비 | 실제 NT 전리 전력·실제 여기 전력·전자 가열·제약행 뒤 소비 확인 상태 |
| 판독 | 성분별 정의 상태·사유·율 상한 범주·후속 기록 필요 항목 |

과학적 부재는 `ABSENT_ELEMENT` 또는 `ABSENT_ION`, 정보 부족은 `UNDEFINED`로 구별한다. 원소 부재에서 율비를 0으로 만들지 않는다. 확인된 NLTE II 총량 0은 해당 조건부 전력 상한 0을 허용한다.

재사용 범위는 다음으로 제한한다.

- diag11의 `r1_common.py`: CSV·JSON·명시 입력 신원 판독. 기존 출력 위치를 고정하는 setup·finish는 호출하지 않는다.
- diag11의 `r2_nt_inventory.py`: 침적 배열 manifest와 유효 설정·발행 연결. 기존 Si/S/Ca 장부 전체 집계를 반복하지 않는다.
- `det_ionrate_bound_readout.py`: 단면적 바이너리 판독. Milne·DR·필요 총전리율 계산은 사용하지 않는다.
- `det_siiii_pop_readout.py`: 구간 사칙연산·외향 반올림·합산. 새 로그 적분에 필요한 부분만 보완한다.
- `det_si_trace_readout.py`: 주석 포함 trace 판독과 키 해석. staging·solve·행렬 전체 재생은 실행하지 않는다.

Si 셸10의 T 채택 호출36에서 바닥 BF 계수, NT 계수, SL 구성과 세대만 읽어 새 적분·발행 연결의 기준 대조로 사용한다. 원자료는 `si10_calls.csv`·`si10_ops.csv`·`si10_slmap.csv`다. 새 계산과 저장 소비 계수의 대칭 상대차 \(2|a-b|/(|a|+|b|)\le10^{-8}\)를 요구한다. 양쪽 0은 일치다.

이 대조의 성공은 Si의 해당 기록에 한정한다. 철족 생산 소비의 확인으로 일반화하지 않는다. 실패하면 원인을 고치거나 해당 연결을 미정의로 남기며 허용차를 확대하지 않는다.

보조 산출은 입력·코드 신원, 기준 대조, 전제와 미정의 사유, 음성 대조, 실행·비용 및 최종 분기를 담은 `evidence.json`과 실행 로그뿐이다.

## §7. 수치·미정의·전파

입력은 저장된 수치 표현에 대한 계산값으로 취급한다. 수치 구간은 물리 오차나 공개 원자자료의 불확실성이 아니다. 기존 구간 산술을 사용하고 로그 끝점은 Decimal의 50자리 로그와 외향 변환으로 감싼다. 유한성 검사는 파생값까지 적용한다.

음의 밀도·J·단면적, 잘못된 문턱·키·세대는 해당 의존 성분의 결함이다. 비유한 파생값, 비영 곱·적분의 0 소실, 필요한 비영 가산의 표현 소실, 0을 포함하는 분모 구간은 미정의다. clamp·floor·임의 작은 분모·상한 절단은 금지한다. 문턱에서 적분 지원을 자르는 것은 물리 정의이며 수치 수리가 아니다.

한 성분의 미정의는 그 성분과 의존 파생값에만 전파한다. 예를 들어 단면적 부재는 광전리와 B를 미정의로 만들지만 확인된 침적 입력을 지우지 않는다. 원소·셸 행을 삭제하거나 미정의 성분을 빼고 완전한 합으로 보고하지 않는다.

공통 epoch·물질/복사 연결·원자 행 대응·공통 단위·기하의 결함은 그 의존 상태 전체를 중단한다. 단일 원소의 유효 SL 구성 부족은 다른 원소를 중단하지 않는다. 모든 전파 범위를 evidence에 남긴다.

`offline_bound_valid`는 저장 표현·발행 연결·단독 바닥 SL·비음성·지원·수치 유효성을 확인했다는 뜻이다. `production_consumption_verified`와 별개이며 철족의 후자는 이번 저장자료에서 false다. 전자를 실제 생산 소비의 확인으로 바꾸지 않는다.

## §8. 기대치와 후속 분기

사전 기대는 발행 NT 배열이 diag11과 연결되고 Si 기준 대조가 일치하는 것이다. 철족 B의 크기·셸별 순서·전력 상한·최종 분기는 **예측 불가**로 등록한다. NT 처리 차이가 결손 원인이라는 기대를 두지 않는다.

율 상한의 판독 분해능은 \(\epsilon=0.01\)이다. 이는 후속 계측을 고르는 업무 기준이며 스펙트럼 정확도나 물리적 무시 가능성의 기준이 아니다. 대상 집합 A는 원소 총량과 NLTE II가 양수인 모든 등록 행이다. A의 멤버십 자체가 미확인이면 전체 비교도 미정의다.

다음 순서로 단 하나의 분기를 낸다.

| 코드 | 전제와 조건 | 처분 |
|---|---|---|
| 0 OFFLINE_REWORK | 입력·구현·연결·시험의 확인된 결함이 하나라도 있음 | 해당 오프라인 결함을 수정. GPU로 이관하지 않음 |
| 1 POPULATION_TRACE_NEEDED | 공통 검사·Si 대조·A 완전성 통과, A 전부 offline_bound_valid, 모든 B 구간 정의, 하나 이상 \(B_{\rm lo}\ge\epsilon\) | 저장 상한만으로 작은 NT를 보장하지 못함. 실제 인구 가중 기록 권고 |
| 2 NO_NT_TRACE_FROM_THIS_TEST | 분기1과 같은 유효 전제, A가 비어 있지 않고 전부 \(B_{\rm hi}<\epsilon\) | 이 저장 상태의 NT/광전리 질문만을 위한 GPU 기록은 권고하지 않음 |
| 3 INDETERMINATE | 결함은 확인되지 않았으나 필수 전제 미확인·A 공집합·B 미정의·경계 걸침 | 부족한 항목과 CPU 보완 가능 여부를 명시. 자동 GPU 이관 없음 |

분기 함수는 표의 모든 전제를 직접 검사한다. 판독 완료, 일부 유한값 또는 명목 NT 복원만으로 `offline_bound_valid`를 대신하지 않는다. 주 표를 만드는 같은 실행에서 메모리 결과로 판정하며 저장 표를 다시 읽는 분기 재계산 장치를 만들지 않는다.

분기1의 최소 후속 기록 대상은 \(B_{\rm hi}\ge\epsilon\)인 행과 미정의 행이다. 필요한 기록은 다음과 같다.

1. 물질13을 생산한 복사12·T·침적 epoch/hash, 후보·전자밀도 반복·CE 반복·채택 여부와 원소·셸 키.
2. 실제 NT 가산과 바닥 인구, 모든 II 준위의 광전리 소비 합, NT 출발·도착 준위와 전이 에너지, 제약행 전후 보존. 기본 pair와 단계 진단은 따로 기록.
3. 가열·여기 닫힘까지 후속 질문에 포함할 때만 같은 상태의 전자 열식·총식·직접 복사 원천·시간 항·고정 T 외부 교환을 추가. 총식과 전자식을 합산하지 않음.

분기2도 가열·여기 닫힘의 별도 생산 기록 필요성을 없애지 않는다. 분기3에서 원자자료·세대 해석이 부족하면 먼저 CPU 자료 연결을 보완한다. 실제 호출 인구·행렬만 없을 때 생산 기록을 권고한다. 어느 분기도 GPU 실행이나 생산 변경을 승인하지 않는다.

## §9. 게이트와 직접시험

| 게이트 | 검사 | 막는 실제 실패 |
|---|---|---|
| G-PLAN | 기존 prereg·referents·canonical·dispatch 검사와 러너 위생 | 잘못된 판본·참조·계약 재서술로 발주 |
| G-INPUT | 입력 신원·epoch·세대·키·단위·유효 설정 | 물질14 또는 관측 J13을 물질13 소비에 결합 |
| G-BOUND | 발행·분모·SL·문턱 적분·Si 대조·구간 산술 | 원자당 율을 이온당 율로 오인하거나 바닥 인구를 과대 대입 |
| G-DECIDE | 주 표 완전성·독립 미정의·분기 전건 전제 | 직전 단처럼 소비 유효 전제를 빠뜨린 분기 |
| G-RUN | 실제 하위 종료·자원·출력 신규성·입력 보존 | 실패한 하위 실행 뒤 완료 선언 |

게이트는 기존 검사와 같은 판독 경로의 검사 범주다. 게이트별 새 프로그램을 만들지 않는다.

직접시험은 새 코드에만 적용한다. 작은 정상 입력을 짝으로 두고, 아래 결함을 하나씩 실제 판독·계산·분기 경로에 넣어 거부 또는 등록된 미정의 전파를 시연한다.

- 입력: 셸 중복·누락, 원소/이온/준위 오결합, 잘못된 단위, 세대14·복사13 치환, 침적 manifest 불일치.
- 율: n_atom 대신 n_e 사용, NT 계수 0.05 누락, 단면적 행 치환, 문턱 부분 bin 보정 누락·이중 절단.
- 권위: 단계 II를 NLTE II로 사용, 유효 cutoff 누락으로 SL 0 다중 준위 유지, 상한을 실제 소비로 표시.
- 산술: 입력 비유한, 유한 피연산자 넘침, 비영 결과의 0 소실, 비영 가산 소실, 0 포함 분모.
- 전파: 한 원소·셸·성분 결함, 공통 세대 결함, 실제 여기 미저장값의 0 치환.
- 분기: 정상 대조에서 필수 전제를 하나씩 제거해 분기1·2 차단, B 경계 걸침·A 공집합·부분 누락, 하위 실행 비영 종료의 러너 전파.

각 시험은 피연산자가 실제로 바뀌었고 해당 경로에서 검출됐음을 기록한다. 결론 문자열만 바꾸는 시험, 저장 결과 변조 후 재판정, 쓰기 실패 모사, 기존 전체 시험 묶음 복제는 하지 않는다.

## §10. 변경집합·실행·비용

### 변경집합 - DET-NT-POP-OFFLINE-BOUND (표)

| 파일 | 허용 변경 |
|---|---|
| `/gpfs/kjhan/lumina/cmfgen_map/ntcons_20261012/offline/tools/det_nt_pop_bound.py` | 기존 판독 재사용, 조건부 적분·상한·주 표·직접시험·분기 |
| `/gpfs/kjhan/lumina/cmfgen_map/ntcons_20261012/offline/tools/run_det_nt_pop_bound.slurm` | CPU 실행·기존 게이트·자원 제한·종료 전달 |

### 변경집합 끝

기존 파일은 수정하지 않는다. 새 함수 이름은 `det_nt_pop_bound_` 접두를 사용한다.

모든 해시·기계 검사·직접시험·계산·감사 재현은 a10 CPU Slurm에서 수행한다. 로그인 노드는 조회만 허용한다. 작업명은 `LUMINA_` 접두이며 `--partition=a10 --ntasks=1 --cpus-per-task=1 --mem=8G --gres=gpu:0 --exclude=syn102,syn03 --no-requeue`를 사용한다.

작업당 벽시계 10분, 전체 할당 벽시계 합 20분, 전체 1/3 core·h, 신규 저장 64 MiB 중 먼저 닿는 상한을 적용한다. 실패·수정·게이트·감사 재현을 포함한다. GPU 0, 생산 변경 0, 모델 진입 0이다. 스레드는 모두 1로 고정한다. 자동 재제출이나 상한 확대는 없다.

원 포획·원자 바이너리·trace를 복제하지 않는다. 필요한 명시 파일만 읽고 결과를 신규 run 위치에 쓴다. 사용한 Python·NumPy 및 재사용 모듈 판본, 정확한 명령·작업 번호·종료·비용을 남긴다.

등록 인터페이스는 F를 작업 위치로 하여 Slurm 안에서 실행한다.

```bash
python3 -B tools/det_nt_pop_bound.py selftest --out tests/run1
python3 -B tools/det_nt_pop_bound.py run --out run1
```

재현은 같은 입력·명령의 새 출력 위치를 사용하며 잔여 상한 안에서 수행한다. trace13의 약 38.973 GPU·h는 같은 상태 재취득의 비용 선례일 뿐 이번 권한·예산으로 승계하지 않는다.

## §11. 완료·감사·주장 상한

완료는 주 표가 저장자료의 조건부 상한과 §8 분기를 결정하고, 실제 소비의 미식별 범위와 후속 최소 기록을 분리한 상태다. 발행 확인이나 계수 계산만 끝난 상태를 완료로 처리하지 않는다. 분기3도 미정의 원인과 필요한 자료가 특정돼야 제한된 결과로 닫을 수 있다.

종료감사는 주 표의 원자료 연결, 유효 cutoff와 바닥 SL, 문턱 단면적 규약, 저장 J의 표현 한계, 상한의 대수 전제, 독립 미정의 전파, 분기의 모든 전제와 GPU 권고 범위를 우선 검사한다.

생산 승격·물리적 정답·수리·새 T·자기일관 효과·청색 회복률을 주장하지 않는다. 비열적 처리 차이를 결손 원인으로 단정하지 않는다. 저장 표현의 상한과 실제 생산 소비를 구별하며, 작은 NT 상한을 모든 침적 효과의 부재로 확대하지 않는다.

기안자는 공유 맥락 전체와 cwd, CLAUDE 규약, 지정 사용자 결정, diag11 두 진단·두 감사·폐합 기록, 직전 단과 이온율·이중 권위·Si trace 기록, O 입력·덤프, 봉인 소스, CMFGEN 소스, 공개 자료 표제와 논문을 직접 조회했다. 계산·해시 재계산·기계 검사 통과는 주장하지 않는다.

집행 기록의 정본은 이 절이다. 운전자는 기계 검사 결과, 미실행 항목의 실행 결과, 구현·직접시험·실행 신원, 비용, 분기·이탈, astra 종료감사와 재현 상태를 추가한다.

## §12. 기계 선언

참조는 실재 파일만 지정한다. 아래 분기 예시는 기존 문면 검사의 입력이며 별도 결정표 재계산 장치를 요구하지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 - DET-NT-POP-OFFLINE-BOUND (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "/gpfs/kjhan/lumina/cmfgen_map/ntcons_20261012/offline/tools/(?:det_nt_pop_bound\\.py|run_det_nt_pop_bound\\.slurm)",
    "symbol": "det_nt_pop_bound_",
    "roots": [
      "/gpfs/kjhan/lumina/cmfgen_map/ntcons_20261012/offline/tools/det_nt_pop_bound.py",
      "/gpfs/kjhan/lumina/cmfgen_map/ntcons_20261012/offline/tools/run_det_nt_pop_bound.slurm"
    ],
    "expected_extra": [
      "/gpfs/kjhan/lumina/cmfgen_map/ntcons_20261012/offline/tools/det_nt_pop_bound.py",
      "/gpfs/kjhan/lumina/cmfgen_map/ntcons_20261012/offline/tools/run_det_nt_pop_bound.slurm"
    ]
  },
  "branches": {
    "regimes": [[0,0],[1,1],[2,2],[3,3],[9,9]],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,2,3) for x in v)",
      "c0": "sum(1 for x in v if x == 0)",
      "c1": "sum(1 for x in v if x == 1)",
      "c2": "sum(1 for x in v if x == 2)"
    },
    "rules": [
      {"name":"OFFLINE_REWORK","predicate":"valid and c0 == 1"},
      {"name":"POPULATION_TRACE_NEEDED","predicate":"valid and c1 == 1"},
      {"name":"NO_NT_TRACE_FROM_THIS_TEST","predicate":"valid and c2 == 1"},
      {"name":"INDETERMINATE","predicate":"not valid or (c0 == 0 and c1 == 0 and c2 == 0)"}
    ],
    "adversarial_fixtures": [
      {"name":"오프라인결함","mix":[[0,1]]},
      {"name":"인구기록필요","mix":[[1,1]]},
      {"name":"이질문추가기록불요","mix":[[2,1]]},
      {"name":"미정의","mix":[[3,1]]},
      {"name":"잘못된값","mix":[[9,1]]},
      {"name":"누락","mix":[]},
      {"name":"중복","mix":[[1,1],[2,1]]}
    ]
  },
  "references": [
    {"path":"docs/PREREG_DET_DEP_OFFLINE_RESPONSE_2026-10-12.md"},
    {"path":"docs/VERDICT_DET_DEP_OFFLINE_RESPONSE_2026-10-12.md"},
    {"path":"docs/VERDICT_DET_IONRATE_BOUND_2026-10-10.md"},
    {"path":"docs/VERDICT_DET_BLUE_ION_MAP_2026-10-10.md"},
    {"path":"docs/PREREG_DET_SI_CONSUMPTION_TRACE_2026-10-10.md"},
    {"path":"docs/VERDICT_DET_SI_CONSUMPTION_TRACE_2026-10-11.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/ntcons_20261012/evidence/USER_DECISION_2026-10-12.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag11_bluecause_20261012/DIAGNOSIS.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag11_bluecause_20261012/DRIVER_AUDIT_r2.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag11_bluecause_20261012/LOOP_CLOSED.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag11_bluecause_20261012/tools/r1_common.py"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag11_bluecause_20261012/tools/r2_nt_inventory.py"},
    {"path":"scripts/det_ionrate_bound_readout.py"},
    {"path":"scripts/det_siiii_pop_readout.py"},
    {"path":"scripts/det_si_trace_readout.py"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/work/physics_comparison/physics_DET_iter0010.manifest.json"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/resolved_lumina.exports"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_atomic.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/bf_rate_jnu.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/physics_comparison.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/radeq_publication.c"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/non_therm/se_ba_non_therm_v3.f90"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/non_therm/electron_non_therm_spec.f90"},
    {"path":"data/standart_data1/toy06/edep_toy06_cmfgen.txt"},
    {"path":"data/standart_data1/toy06/phys_toy06_cmfgen.txt"},
    {"path":"data/standart_data1/toy06/ionfrac_fe_toy06_cmfgen.txt"}
  ]
}
```