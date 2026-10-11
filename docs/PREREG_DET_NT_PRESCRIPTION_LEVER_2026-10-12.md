# 사전등록 - DET-NT-PRESCRIPTION-LEVER: 비열적 처방과 철족 II/III 균형의 조건부 규모 비교

2026-10-12 · astra 역할의 읽기 전용 기안  
상태: 지정 기록·소스·공개 자료 형식·문헌 조회 완료. 구현·계산·시험·제출·판정 미집행.

## §0. Q-GOAL·Q-LEAN 자기답

**Q-GOAL: PASS.** 최종목적은 공개 CMFGEN과 비교 가능한 SN Ia의 절대 광도와 스펙트럼을 얻는 것이다. 이번 단은 현행 비열적 전리 계수의 작은 상한을 CMFGEN의 Spencer-Fano 처리까지 작다는 뜻으로 확대하지 않고, 철족 II/III 균형에 영향을 줄 수 있는 처방 차이의 규모와 필요한 후속 증거를 가른다.

종료 산출은 같은 저장 상태에서 현행 계수, CMFGEN 대응량의 식별 상태, 조건부 전리 민감도와 필요한 에너지 몫을 연결한 표 하나다. 이를 생략하면 현행 처방의 인구 기록만으로 충분한지, 처방 자체의 검토가 먼저인지 구분하지 못한다.

**Q-LEAN: APPROPRIATE.** 직전 단의 판독·광전리 적분·침적 연결·구간 산술을 재사용한다. 신규 계산은 고정 상태의 대수적 민감도와 공개 II/III 비의 대응뿐이다. Spencer-Fano 해법 이식, 새 원자자료 구축, 생산 행렬 재생, 인구 재해와 온도 계산은 하지 않는다.

저장 산출 재검, 쓰기 결함 모사, 별도 결정표 재계산 장치를 만들지 않는다. 보조 증거는 주 표의 입력 신원·정의 상태·음성 대조·비용·분기에 소비되는 것만 남긴다.

## §1. 권한·기반·보존

기안·개정·모든 감사는 astra, 구현·직접시험·게이트·실행·판정은 운전자다. GPU 0·생산 변경 0이므로 경량 사전감사는 생략한다(25-ab). 코드 감사는 없다(25-z). 종료감사는 astra이며 계약 기안자와 같다는 이해관계를 기록한다.

| 약칭 | 위치 |
|---|---|
| P | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn` |
| F | `/gpfs/kjhan/lumina/cmfgen_map/ntpresc_20261012/offline` |
| O | `/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1` |
| S | `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src` |
| C | `/gpfs/kjhan/cmfgen_src/cur_cmf` |
| U | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06` |
| N | `/gpfs/kjhan/lumina/cmfgen_map/ntcons_20261012/offline` |
| D | `/gpfs/kjhan/lumina/cmfgen_map/diag11_bluecause_20261012` |

사용자 결정은 NT 처방 자체의 CMFGEN 대응 비교다. 결정 파일의 선택지 설명과 비용 예시는 운전자 문구이며 이번 과학 계약이나 자원 승인으로 승계하지 않는다.

신규 구현·시험·산출은 F 아래에만 둔다. 생산 소스·런·원자료·선행 산출·재사용 모듈은 수정하지 않는다. 명시적으로 소비한 파일의 크기와 SHA-256을 기록한다. 전체 트리 복사·해시, 하드링크, 파일시스템 고유 기능은 사용하지 않는다. 새 실행은 새 출력 위치를 사용하며 실패 산출도 보존한다.

## §2. 과학 질문과 재현 가능 범위

> O의 물질13·복사12·T12 저장 상태에서, 현행 비열적 처방을 에너지 분할에 따른 전리율로 바꿀 때 철족 II/III 비를 두 배 바꾸려면 어떤 침적 몫이 필요한가? 이 조건부 규모를 실제 CMFGEN 처리와 연결하는 데 무엇이 부족한가?

대상은 Fe·Co·Ni, 셸 0-49, II에서 III로 가는 직접 전리다. 모든 150행을 유지한다. III에서 IV로의 손실과 다른 이온의 경쟁은 실제 CMFGEN 대응에 필요한 정보로 등록하지만 새 사다리 계산을 하지 않는다.

**지정 자료만으로 CMFGEN의 실제 비열적 율·분할·균형 응답은 재현할 수 없다.** 본 단이 계산하는 것은 명시한 단순화 아래의 조건부 민감도이며, Spencer-Fano 근사해로 명명하지 않는다.

| 등급 | 이번에 가능한 것과 불가능한 것 |
|---|---|
| 현행 저장 표현 | 침적 발행, 원자당 NT 계수, 바닥 광전리 계수, 단계 및 NLTE II·III 총량 복원 |
| CMFGEN 소스 대응 | 전자분포 생성, 전리·여기 소비, 분할 진단, 총식·전자 열식의 연결 확인 |
| 조건부 규모 | 분할 몫을 미지수로 둔 전리 계수와 고정된 두 이온 단계 근사의 민감도 |
| 실제 Spencer-Fano | 같은 상태의 준위별 인구·전자 충돌 자료·주입 및 격자 설정이 연결되지 않아 미식별 |
| 실제 균형·관측 응답 | 전체 율, 시간 항, 인구·전자밀도·복사·온도 피드백이 없어 미식별 |

필요한 Spencer-Fano 입력은 `XKT`, `dXKT`, 주입 `SOURCE`와 `E_INIT`, `THD` 전리 경로·전자 충돌 단면적, 여기 자료, 전체 경쟁 이온의 준위 인구, 전자밀도, 생략 설정과 소비 시점이다. O의 광전리 단면적은 전자 충돌 전리 단면적의 대체물이 아니다.

공개 이온 분율이나 T·전자밀도가 비슷하다는 이유로 이 입력들을 복원하지 않는다. 예상 밖의 추가 자료를 발견해도 이번 단 안에서 새 Spencer-Fano 계산으로 확대하지 않는다.

## §3. 상태·율·분율·전력의 잣대

주 입력은 O의 `physics_DET_iter0010` manifest·shell·ion·spectral 자료와 해당 입력 덱이다.

- 물질13·복사12·T12, `COMMITTED`, `FIXED_T`, epoch 1683072 s를 사용한다.
- 침적은 generation 1의 발행 신원과 epoch로 연결한다. 인구 세대와 번호를 맞추지 않는다.
- 관측 포획의 복사13, iter0011의 물질14, 최종 인구14를 대입하지 않는다.
- NLTE II·III 합과 단계 II·III 합을 별도로 유지한다. 조건부 계산은 NLTE 합을 사용한다.
- 저장 J는 재격자된 표현이다. 모든 적분은 `STORED_REPRESENTATION_CONDITIONAL`이다.

셸 부피 \(V\), 침적 밀도 \(q\), 전체 원자밀도 \(n_{\rm atom}\), NLTE 밀도 \(N_{\rm II},N_{\rm III}\), 원소 총밀도 \(N_Z\)를 사용한다.

\[
D=Vq,\qquad
R_0={0.05q\over35\,{\rm eV}},\qquad
\gamma_0={R_0\over n_{\rm atom}}.
\]

\(q\)는 erg s\(^{-1}\) cm\(^{-3}\), \(D\)는 erg s\(^{-1}\), \(R_0\)는 cm\(^{-3}\) s\(^{-1}\), \(\gamma_0\)는 s\(^{-1}\)이다. 봉인 상수의 eV 변환을 사용한다.

\[
f_{\rm II}=N_{\rm II}/N_Z,\quad f_{\rm III}=N_{\rm III}/N_Z,\quad
r=N_{\rm III}/N_{\rm II}.
\]

분율과 비는 무차원이다. II·III 밖의 인구가 있으므로 두 분율을 합 1로 재정규화하지 않는다.

\[
\gamma_g=4\pi\sum_bJ_b\int_{\nu_{b,-}}^{\nu_{b,+}}
{\sigma_g^*(\nu)\over h\nu}\,d\nu,\qquad B=\gamma_0/\gamma_g.
\]

바닥 광전리 적분은 직전 단의 문턱 부분 bin 보정·지원·외향 구간 규약을 그대로 사용한다. 단면적 부재를 외삽이나 다른 원소로 메우지 않는다.

0.05는 코드 계수이며 실제 전리 에너지 분율이 아니다. 35 eV는 해당 전이의 문턱 에너지가 아니다.

## §4. CMFGEN 대응과 소비 방식

C의 소스 위치는 `new_main` 및 그 아래 `subs/non_therm`이다. 아래 행 번호는 기안 조회 판본 기준이며 실행 시 파일 신원을 고정한다.

| 대응량 | 소스 근거와 단위 해석 |
|---|---|
| 분포와 소비의 호출 | `cmfgen_sub.f` 약 1690행: 분포 계산 뒤 `SE_BA_NON_THERM_V3` 호출 |
| 주입·경쟁 인구·쿨롱 손실 | `electron_non_therm_spec.f90` 약 235·302·339행: 주입 정규화, 준위 인구 합, 전자밀도 의존 손실 |
| 전리·여기·분포 해 | 같은 파일 약 355·405·528행: 전리와 여기 행렬, 열화 분포 계산 |
| 에너지 분율 | 같은 파일 약 543·581·617·633행: 전자 가열·전리·여기 및 합. 무차원 진단 |
| 분포 정규화 | 같은 파일 약 664행: `YE/E_INIT`. 소비식에서 `E_INIT`를 다시 나누지 않음 |
| 전리 계수와 흐름 | `se_ba_non_therm_v3.f90` 약 115-156행: 정규화 `YE`·단면적 적분에 침적/eV와 배율을 곱하고 준위 인구·경로 가중으로 소비 |
| 여기 소비 | 같은 파일 약 202-240행: BETHE 계수, 하준위 인구, 상하준위 갱신 및 전이 에너지 |
| 에너지식 | 같은 파일 약 68-89·241행과 `cmfgen_sub.f` 약 2972행: 총식 침적, 전자 열식의 NT 차감과 침적 가산 |

CMFGEN 대응 계수는 전이별 \(\gamma_{ij}\) [s\(^{-1}\)], 흐름은 \(n_i\gamma_{ij}\) [cm\(^{-3}\) s\(^{-1}\)], 전력은 \(Vn_i\gamma_{ij}\Delta E_{ij}\) [erg s\(^{-1}\)]로 환산해야 한다. 내부 `STEQ` 값에는 \(4\pi\)와 \(10^{10}\) 배율이 있으므로 그대로 전력과 비교하지 않는다.

분포 생성의 `FRAC_*`와 인구식에서 합산한 소비 분율은 별개다. 준위 생략·경로·에너지 정의·배율이 일치하는지 확인하기 전에는 동일량으로 취급하지 않는다. 한 전자 경로와 다중 전자 경로도 II에서 III로의 흐름에 합쳐 넣지 않는다.

현행 배선은 `lumina_plasma.c` 약 23919행의 발행, 약 3943행의 단계 계수, 약 20423행의 기본 pair 바닥 전리 가산이다. 유효 바닥 SL의 단독 준위 구성은 직전 단과 같은 cutoff·원자 투영 연결로 확인한다. 단계 소비와 NLTE 소비는 같은 전력의 두 측정값으로 합산하지 않는다.

공개 계산은 비국소 감마선 침적, Spencer-Fano 비열적 처리, 시간의존 인구 및 에너지식을 사용한다. 총 물질 에너지식과 자유전자 열수지 검사는 구별된다. 이는 처리의 근거이며 로컬 소스가 공개 실행의 정확한 판본임을 보증하지 않는다. [Blondin et al. 2022, 3.2절](https://arxiv.org/html/2209.11671v2#S3.SS2)

## §5. 조건부 규모 계산과 근사 한계

### 고정 상태의 정의

직접 II 전리에 배정되는 침적 몫을 \(\eta_Z\), 전리 한 건당 내부 에너지 비용을 \(w_Z\)라 하면

\[
P_{{\rm ion},Z}=\eta_ZD,\qquad
\gamma_\eta={\eta_Zq\over N_{\rm II}w_Z}.
\]

이는 인구 평균 유효 계수의 정의다. \(\eta_Z\)는 전체 전리 분율이 아니며 실제 CMFGEN 값은 미식별이다. 셸마다 \(\eta_Z\ge0\), \(\sum_Z\eta_Z\le1\)은 필요조건일 뿐 Spencer-Fano 해의 충분조건이 아니다.

주 계산은 **단일 원소씩** \(w_Z=\chi_Z\), 즉 II 바닥에서 III 바닥까지의 에너지를 사용하는 조건부 모형이다. 여기 준위 전리·다중 전자 경로·다른 이온 경쟁을 재현하지 않는다.

현행 모형의 조건부 전력은

\[
P_0^*=VN_{\rm II}\gamma_0\chi_Z,\qquad \eta_0^*=P_0^*/D.
\]

\(P_0^*\)는 직전 단의 전력 상한과 같은 대수량이다. 실제 현행 소비나 CMFGEN 분율로 바꾸어 부르지 않는다.

### II/III 민감도

두 이온 단계의 정상 균형, 고정된 역방향 유효 계수, 모든 II가 바닥에 있다는 조건에서

\[
M(\eta)={\gamma_g+\gamma_\eta\over\gamma_g+\gamma_0},
\qquad \Delta\log_{10}r^*=\log_{10}M(\eta).
\]

이 식은 역방향 계수가 소거되는 조건부 비다. 저장 인구에서 재결합 계수를 역으로 만들어 물리 계수라고 주장하지 않는다. 새 인구해도 산출하지 않는다.

비를 두 배 올리는 데 필요한 몫은

\[
\boxed{\eta_2^*={N_{\rm II}\chi_Z\over q}
                 (\gamma_g+2\gamma_0)}.
\]

주 표에는 \(\eta_2^*\), \(M(0)\), \(M(0.01)\), \(M(1)\)을 기록한다. 0.01은 후속 우선순위용 예산 눈금이고 1은 단일 경로에 침적 전량을 배정하는 극단이다. 둘 다 CMFGEN의 예측 분율이 아니다. 원소별 \(M(1)\)을 동시에 실현 가능한 상태로 합치지 않는다.

### 결론을 바꿀 수 있는 조건

실제 바닥 몫을 \(a=n_g/N_{\rm II}\), 비열적 항 이외의 II 소실 유효 계수를 \(k\gamma_g\)로 쓰면 같은 제한적 균형에서

\[
\eta_2(a,k,w)={N_{\rm II}w\over q}(k\gamma_g+2a\gamma_0),\qquad
{\eta_2(a,k,w)\over\eta_2^*}
={w\over\chi_Z}{k\gamma_g+2a\gamma_0\over\gamma_g+2\gamma_0}.
\]

\(a,k,w\)는 저장 자료로 결정되지 않는다. 이 배율이 \(\eta_2^*\)를 0.01 또는 1의 반대편으로 옮기면 후속 우선순위도 바뀐다. 이 경계 조건을 행마다 남기며 임의의 좁은 오차막대를 붙이지 않는다.

여기·가열은 \(a,k\), 역방향 계수와 T를 바꿀 수 있고, III에서 IV로의 손실과 시간 항은 위 비례식 자체를 무효로 만들 수 있다. 따라서 \(M(1)\)도 실제 CMFGEN 균형 응답의 상한이 아니다.

## §6. 에너지 규칙과 공개 비교

총 물질 에너지식은 전리·여기 내부 에너지를 포함한다.

\[
\rho {De\over Dt}-{P\over\rho}{D\rho\over Dt}
=Q_{\rm rad}+q.
\]

총 공급을 \(D+P_{\rm ion}+P_{\rm exc}\)로 만들지 않는다. 자유전자 열식에서는 전리·여기로 이동한 에너지를 별도로 구별한다. 완전한 채널 집합과 동일 소비 상태가 없으므로 잔액 \(D-P_{\rm ion}-P_{\rm exc}\)를 실제 가열로 확정하지 않는다.

현행 `radeq_publication.c` 약 210행의 GAMMA 전량 기재와 NONTHERMAL 열 항 0은 실제 NT 소비 전력 0이나 이중 청구의 증거가 아니다. 본 단은 실제 가열·여기 분율을 미식별로 유지한다.

U의 `edep_toy06_cmfgen.txt`, `phys_toy06_cmfgen.txt`, 철족 `ionfrac_*_toy06_cmfgen.txt`는 침적·거시 상태·이온 분율을 제공한다. 내부 J, 준위 인구, 비열적 분포·채널 전력은 이 지정 파일들에 없다.

19.480 d 블록을 정확히 선택하고 O 셸 중간 속도에 대해 인접 공개 속도 두 점 사이의 선형 보간만 허용한다. 외삽·시간 최근접 대체·분율 재정규화는 하지 않는다. 공개 \(f_{\rm II},f_{\rm III}\)와 \(r_C\), O의 두 인구 권위별 \(r\)를 주 표에 둔다.

공개 상태는 같은 시각의 별도 해다. 이를 O와 같은 저장 상태로 부르거나 \(r_C/r\)를 NT 처방의 효과로 역산하지 않는다. 공개 자료의 T·전자밀도·침적은 비교 신원과 상태 차이를 표시하며 §5의 O 입력을 대체하지 않는다.

## §7. 주 표와 재사용

주 산출은 `nt_prescription_lever.csv` 하나다. 150행을 유지하며 표시본과 분기는 같은 메모리 행에서 만든다.

| 열군 | 내용 |
|---|---|
| 신원 | 원소·셸·세대·epoch·표현 등급·근사 등급 |
| 상태 | \(q,D,n_{\rm atom},N_Z,N_{\rm II},N_{\rm III}\), 단계 II·III, 공개 대응 상태 |
| 기준 | \(\chi_Z,\gamma_0,\gamma_g,B\), NLTE·단계·공개 분율과 II/III 비 |
| 조건부 규모 | \(\eta_0^*,\eta_2^*,M(0),M(0.01),M(1)\), 근사 변경의 경계 배율 |
| CMFGEN 연결 | 실제 전리·여기·가열 분율과 소비의 식별 상태, 부족한 입력 |
| 판독 | 성분별 상태·사유·의존 범위·행 범주·후속 최소 증거 |

N의 `det_nt_pop_bound.py`에서 입력 판독, 침적 연결, 상수·세대·cutoff 검사, 바닥 광전리 적분, 국소 행 생성과 Si 기준 대조를 재사용한다. 기존 `run`, `write`, `selftest`, `decide`와 고정 출력 위치는 호출하지 않는다.

그 모듈이 사용하는 `r1_common.py`, `r2_nt_inventory.py`, `det_ionrate_bound_readout.py`, `det_siiii_pop_readout.py`, `det_si_trace_readout.py`는 필요한 판독·구간 기능만 재사용한다. 공개 블록은 기존 `det_blue_ion_map_readout.py` 판독 기능을 사용한다. 전체 이전 분석이나 시험 묶음을 재실행하지 않는다.

보조 산출은 `evidence.json`과 실행·시험 로그다. 소스 대응, 실제 소비 미식별, 입력·코드 신원, 음성 대조, 분기 전제와 비용을 이 증거에 기록한다. 별도 과학 요약표는 만들지 않는다.

## §8. 미정의·전파·게이트

수치 구간은 저장 표현의 산술 오차를 감싸며 물리 근사 오차를 뜻하지 않는다. 비유한 파생값, 넘침, 비영 결과의 0 소실, 필요한 비영 항을 구간에도 보존하지 못한 가산 소실, 0 포함 분모는 미정의다. clamp·floor·작은 분모 대체·상한 절단은 금지한다.

- \(q=0,n_{\rm atom}>0\)이면 \(\gamma_0=0\)은 정의된다. \(q=0,n_{\rm atom}=0\)의 실패를 0으로 덮지 않는다. \(q=0\)의 \(\eta_2^*\)는 미정의다.
- 확인된 원소 부재는 `ABSENT_ELEMENT`, II 부재는 `ABSENT_ION`으로 유지한다. 비와 평균 계수를 0으로 만들지 않는다.
- 단일 J 결함은 그 셸의 광전리 의존량에, 원소별 단면적·문턱 결함은 해당 원소의 의존량에 전파한다.
- 공개 보간 불가나 분율 결함은 공개 대응 성분에만 전파한다. O의 정의된 조건부 규모는 보존한다.
- 공통 단위·epoch·세대·전역 키 대응 결함은 의존 상태 전체의 계산을 멈춘다. 행 골격과 독립 증거는 남긴다.
- 잘못된 입력·구현은 `DEFECT`, 자료 부재는 `UNIDENTIFIED`, 계산 불능은 `UNDEFINED`로 구별한다. 미정의 성분을 빼고 완전한 결과로 보고하지 않는다.

| 게이트 | 막는 실재 실패 |
|---|---|
| G-PLAN | 허용 변경집합·참조 파일·분장·기계 선언이 계약과 다른 상태의 착수 |
| G-INPUT | 다른 세대·권위·단위·원자 키를 같은 저장 상태로 결합 |
| G-MODEL | 조건부 식을 실제 Spencer-Fano 결과로 표시하거나 에너지를 중복 청구 |
| G-DECIDE | 국소 결함으로 표를 지우거나 불완전한 전제로 후속 분기를 확정 |
| G-RUN | 로그인 계산·GPU 사용·비용 초과·하위 실행 실패의 성공 처리 |

기존 문면 검사를 재사용한다. 새 범용 검사기나 저장 결과 재판독기는 만들지 않는다.

## §9. 기대치·분기·직접시험

기대는 현행 발행·바닥 계수가 직전 단과 같은 정의로 복원되고, 실제 CMFGEN 분율·소비는 미식별로 남는 것이다. \(\eta_2^*\)의 크기·공개 비와의 방향 관계·최종 분기는 예측 불가다. 처방 차이가 결손 원인이라는 기대는 두지 않는다.

활성 집합 A는 \(N_Z,N_{\rm II},N_{\rm III},q>0\)인 등록 행이다. 멤버십 미확인은 완전성 실패다. 공통 유효성 C는 신원·세대·발행·적분·Si 대조·새 음성 대조·에너지 규칙 통과다. Si 대조 허용차는 직전 단의 대칭 상대차 \(10^{-8}\)을 유지한다.

다음 순서로 단 하나의 전체 분기를 낸다. 개별 행의 정의 상태와 범주는 전체 분기와 별도로 보존한다.

| 코드 | 모든 전제와 조건 | 후속 처분 |
|---|---|---|
| 0 OFFLINE_REWORK | 확인된 계약·입력·구현·시험 결함이 하나라도 있음 | 해당 오프라인 결함 수정. GPU로 넘기지 않음 |
| 4 INDETERMINATE | 결함은 없으나 C·A 완전성 미확인, A 공집합, 필수 규모 미정의 또는 아래 경계 판정 불가 | 부족한 성분과 CPU 자료 보완·생산 기록의 구별을 명시 |
| 1 PRESCRIPTION_REVIEW_CANDIDATE | C, A 완전·비공집합, A 전부 조건부 계산 유효, 하나 이상 \(\eta_{2,\rm hi}^*\le0.01\) | 작은 배정 몫으로 큰 조건부 민감도. 생산 처방 변경의 검토 후보로 남기되 먼저 CMFGEN 입력·분할 근거 확보 |
| 3 OTHER_TARGET_CONDITIONAL | 같은 유효 전제, A 전부 \(\eta_{2,\rm lo}^*>1\), A 전부 \(M_{\rm lo}(0)>1/2\) | 이 근사의 직접 II 전리 규모는 낮은 우선순위. 다른 표적 검토 후보 |
| 2 CONSUMPTION_TRACE_CANDIDATE | 같은 유효 전제, 분기1·3 제외가 구간으로 확정됨 | 바닥 몫·전체 II 소실·역방향 율·III 손실의 실제 소비 기록을 후속 후보로 제시 |

구간이 0.01·1·1/2 경계를 걸치면 그 경계가 분기 선택에 영향을 주는 경우 분기4다. 분기 함수가 C와 각 행의 권위·정의·완전성·근사 등급을 직접 검사한다. 결과 코드만 입력받아 분기를 정당화하지 않는다.

분기1·2·3은 조건부 우선순위다. 실제 Spencer-Fano 분율을 입증하거나 생산 변경·GPU 실행을 승인하지 않는다. 분기2에서도 GPU가 공급하지 못하는 전자 충돌 자료·주입 설정은 별도 CPU 자료 확보가 필요하다. 분기3으로 여기·가열·시간의존 효과까지 기각하지 않는다.

직접시험은 **새 코드의 음성 대조**만 수행한다. 작은 정상 입력은 결함 주입의 기준으로만 사용한다. 다음 결함을 실제 판독·계산·행 생성·분기 경로에서 각각 시연한다.

- 세대·단위·중복/누락 키·원소 오결합·단계/NLTE 교체·공개 시각 오선택·외삽.
- \(n_{\rm atom}\) 대신 \(n_e\), 0.05 누락, eV 변환 오류, 기존 NT에 새 계수를 대체하지 않고 가산, \(\eta_2^*\)의 \(2\gamma_0\) 누락.
- 개별 원소의 전량 배정들을 동시 상태로 합산, 총식 이중 청구, 미식별 여기·가열의 0 대체, 조건부 값을 실제 CMFGEN 값으로 표시.
- 입력·파생 비유한, 넘침, 비영 곱·나눗셈·가산의 표현 소실, 0 포함 분모, \(q=0,n_{\rm atom}=0\).
- 한 셸의 음의 J, 한 원소의 음의 단면적·밀도, 공개 성분만의 결함, 공통 세대 결함. 각각 의존량 중단과 독립 성분·150행 보존을 확인.
- 분기 표의 전제를 하나씩 제거, A 공집합·부분 누락, 각 경계의 걸침, 하위 실행 비영 종료. 원래 허용되던 분기가 차단되는지 확인.

피연산자 변경, 검출 위치, 의존 전파와 도달 분기를 기록한다. 판정 문자열만 바꾸거나 저장 표를 변조해 재판정하는 시험은 금지한다.

## §10. 변경집합·실행·비용

### 변경집합 - DET-NT-PRESCRIPTION-LEVER (표)

| 파일 | 허용 변경 |
|---|---|
| `/gpfs/kjhan/lumina/cmfgen_map/ntpresc_20261012/offline/tools/det_nt_prescription_lever.py` | 재사용 연결, 조건부 규모·공개 대응·미정의 전파·주 표·음성 대조·분기 |
| `/gpfs/kjhan/lumina/cmfgen_map/ntpresc_20261012/offline/tools/run_det_nt_prescription_lever.slurm` | CPU 실행·기존 게이트·비용 제한·종료 전달 |

### 변경집합 끝

새 함수 접두는 `det_nt_prescription_lever_`다. 모든 해시·검사·시험·계산·실행 재현은 a10 CPU Slurm에서 수행한다. 로그인 노드는 조회만 허용한다.

작업명 `LUMINA_` 접두, `--partition=a10 --ntasks=1 --cpus-per-task=1 --mem=8G --gres=gpu:0 --exclude=syn102,syn03 --no-requeue`를 사용한다. 작업 안에서 `module purge` 후 사용할 모듈을 명시하고 Python·NumPy·의존 모듈 판본을 기록한다. 모든 수치 라이브러리 스레드는 1이다.

상한은 작업당 10분, 전체 실제 할당 벽시계 합 20분, CPU 1/3 core·h, 신규 저장 64 MiB다. 실패·수정·게이트·감사 재현을 포함하며 먼저 닿는 상한을 적용한다. GPU 0, 생산 변경 0, 생산 모델 진입 0이다. 자동 재제출·상한 확대는 없다.

F를 작업 위치로 하는 등록 인터페이스는 다음과 같다.

```bash
python3 -B tools/det_nt_prescription_lever.py selftest --out tests/run1
python3 -B tools/det_nt_prescription_lever.py run --out run1
```

재현은 같은 명령·입력·코드의 새 출력 위치에서 잔여 상한 안으로 제한한다. 실제 명령, 작업 번호, 종료 코드, 경과·CPU·메모리·신규 저장량을 기록한다.

## §11. 완료·판정·감사

완료는 150행의 주 표가 현행 처방, CMFGEN 대응의 식별 한계, 조건부 규모, 근사 변경 조건과 후속 분기를 연결한 상태다. 실제 CMFGEN 값의 미식별을 숨기거나 계수 복원만 마친 상태를 목표 달성으로 판정하지 않는다.

분기4도 부족한 자료·영향 범위·최소 후속 증거가 특정되면 제한된 결과로 종료할 수 있다. 분기0은 수정 전 완료가 아니다. 기안자는 계산·기계 검사 통과를 주장하지 않는다.

운전자는 구현·직접시험·실행·비용·이탈·판정의 증거 신원을 기록한다. astra 종료감사는 특히 조건부 민감도를 실제 Spencer-Fano 효과로 확대했는지, 에너지와 인구 권위를 섞었는지, 국소 미정의와 분기 전제를 보존했는지 검사한다. 문면 대조와 실행 재현의 수행 여부를 구별한다.

생산 승격·물리적 정답·수리·새 T·자기일관 효과·광도 회복률을 주장하지 않는다. 처방 차이를 결손 원인으로 단정하지 않는다.

## §12. 기계 선언과 참조

아래 분기 예시는 기존 문면 검사의 입력이다. 과학 분기 함수의 전제 검사를 대신하지 않는다. 참조는 실재 파일만 지정한다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 - DET-NT-PRESCRIPTION-LEVER (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "/gpfs/kjhan/lumina/cmfgen_map/ntpresc_20261012/offline/tools/(?:det_nt_prescription_lever\\.py|run_det_nt_prescription_lever\\.slurm)",
    "symbol": "det_nt_prescription_lever_",
    "roots": [
      "/gpfs/kjhan/lumina/cmfgen_map/ntpresc_20261012/offline/tools/det_nt_prescription_lever.py",
      "/gpfs/kjhan/lumina/cmfgen_map/ntpresc_20261012/offline/tools/run_det_nt_prescription_lever.slurm"
    ],
    "expected_extra": [
      "/gpfs/kjhan/lumina/cmfgen_map/ntpresc_20261012/offline/tools/det_nt_prescription_lever.py",
      "/gpfs/kjhan/lumina/cmfgen_map/ntpresc_20261012/offline/tools/run_det_nt_prescription_lever.slurm"
    ]
  },
  "branches": {
    "regimes": [[0,0],[1,1],[2,2],[3,3],[4,4],[9,9]],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,2,3,4) for x in v)",
      "c0": "sum(1 for x in v if x == 0)",
      "c1": "sum(1 for x in v if x == 1)",
      "c2": "sum(1 for x in v if x == 2)",
      "c3": "sum(1 for x in v if x == 3)"
    },
    "rules": [
      {"name":"OFFLINE_REWORK","predicate":"valid and c0 == 1"},
      {"name":"PRESCRIPTION_REVIEW_CANDIDATE","predicate":"valid and c1 == 1"},
      {"name":"CONSUMPTION_TRACE_CANDIDATE","predicate":"valid and c2 == 1"},
      {"name":"OTHER_TARGET_CONDITIONAL","predicate":"valid and c3 == 1"},
      {"name":"INDETERMINATE","predicate":"not valid or (c0 == 0 and c1 == 0 and c2 == 0 and c3 == 0)"}
    ],
    "adversarial_fixtures": [
      {"name":"오프라인결함","mix":[[0,1]]},
      {"name":"처방검토후보","mix":[[1,1]]},
      {"name":"소비기록후보","mix":[[2,1]]},
      {"name":"다른표적후보","mix":[[3,1]]},
      {"name":"미정의","mix":[[4,1]]},
      {"name":"잘못된값","mix":[[9,1]]},
      {"name":"누락","mix":[]},
      {"name":"중복","mix":[[1,1],[2,1]]}
    ]
  },
  "references": [
    {"path":"docs/PREREG_DET_NT_POP_OFFLINE_BOUND_2026-10-12.md"},
    {"path":"docs/VERDICT_DET_NT_POP_OFFLINE_BOUND_2026-10-12.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/ntpresc_20261012/evidence/USER_DECISION_2026-10-12.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag11_bluecause_20261012/DIAGNOSIS.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag11_bluecause_20261012/LOOP_CLOSED.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/ntcons_20261012/offline/tools/det_nt_pop_bound.py"},
    {"path":"scripts/det_ionrate_bound_readout.py"},
    {"path":"scripts/det_blue_ion_map_readout.py"},
    {"path":"scripts/det_siiii_pop_readout.py"},
    {"path":"scripts/det_si_trace_readout.py"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/non_therm/electron_non_therm_spec.f90"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/non_therm/se_ba_non_therm_v3.f90"},
    {"path":"data/standart_data1/toy06/edep_toy06_cmfgen.txt"},
    {"path":"data/standart_data1/toy06/phys_toy06_cmfgen.txt"},
    {"path":"data/standart_data1/toy06/ionfrac_fe_toy06_cmfgen.txt"},
    {"path":"data/standart_data1/toy06/ionfrac_co_toy06_cmfgen.txt"},
    {"path":"data/standart_data1/toy06/ionfrac_ni_toy06_cmfgen.txt"}
  ]
}
```