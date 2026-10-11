# 사전등록 - DET-LINE-NET-PROVENANCE: 고정 T 선 순율 장부의 정의와 세대 추적

2026-10-11 · astra 읽기 전용 기안  
상태: 기안. 소스와 저장 자료 조회 완료. 구현·계산·직접시험·제출·판정 미집행.

## §0. Q-GOAL·Q-LEAN 자기답

**Q-GOAL: PASS.** 최종목적은 공개 CMFGEN과 비교 가능한 SN Ia 이온화 상태와 스펙트럼을 얻는 것이다. 이번 단은 고정 T 장부의 큰 선 순율을 관측 광도 부족과 연결하기 전에, 그 장부가 실제로 소비한 선 집합·복사장·물질 상태를 특정한다. 이를 생략하면 세대가 섞인 선 순율이나 서로 다른 전달 표현의 차이를 실제 에너지 손실로 오인할 수 있다.

관측 가능한 종료 산출은 셸별 정의·세대·대응량·귀속 한계를 담은 표 하나다. 이 표로 전체 장부에서 확인된 차와 제한된 선 표본에서 수치로 분리한 차를 구별하고, 다음 판단에 필요한 미저장 자료가 무엇인지 결정한다. 전체 차의 지배 원인이 반드시 식별된다는 목표는 등록하지 않는다.

**Q-LEAN: APPROPRIATE.** GPU 0·생산 변경 0인 단 하나로 묶는다. 기존 비교 덤프 판독과 선 입력 판독을 재사용한다. 새 계산은 장부의 출처 연결, 저장 선 평균장 관계의 확인, 이미 선정된 Si II 여섯 선의 제한된 산술 분해와 그 음성 대조다.

삭제 대상은 저장 산출 재검 장치·쓰기 결함 모사·결정표 재계산 장치·전체 선 재구성·새 원소별 순위표·전달 재실행이다. 비교 덤프 재적분은 기존 판독 함수를 소비하는 데 한정한다. 관측 포획 선 전체를 다시 읽지 않는다.

보조 증거의 소비처는 주 표의 출처·산술 항·미정의 사유·시험 적격성·비용뿐이다. 종료감사의 1차 대상은 **과학 표의 수치·귀속·주장 한정**이다.

## §1. 권한·기반·입력

기안·개정·모든 감사는 astra, 구현·직접시험·게이트·실행·판정은 운전자다. GPU 0·생산 변경 0이므로 경량 사전감사는 생략한다(25-ab). 코드 감사는 없다(25-z). 종료감사는 astra가 수행한다(25-ac).

| 약칭 | 위치 |
|---|---|
| P | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn` |
| B | `/gpfs/kjhan/lumina/cmfgen_map/linenet_20261011` |
| O | `/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1` |
| T | `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1` |
| E | `/gpfs/kjhan/lumina/cmfgen_map/energyx_20261011/results/run7` |
| D | `/gpfs/kjhan/lumina/cmfgen_map/diag9_absdef_20261011` |
| S | `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src` |
| C | `/gpfs/kjhan/cmfgen_src/cur_cmf` |

사용자의 선택은 B의 `evidence/USER_DECISION_2026-10-11.txt`다. 선택지 설명의 기전 문장은 확정 결론이나 수용 기준으로 승계하지 않는다.

필수 입력은 O의 iter0010·iter0011 비교 덤프 `shell.csv`·`spectral.csv`·`manifest.json`, 실행 설정, 관련 로그, `work/w5_lines.csv`다. E의 표와 증거는 선행 수치의 출처이며 새 추적의 대체 입력이 아니다.

제한된 선 계산에는 T의 `si10_calls.csv`·`si10_ne.csv`·`si10_cevec.csv`·`si10_commit.csv`·`si10_vec.csv`·`si10_slmap.csv`·`si10_ops.csv`, iter0009·iter0010의 필요한 덤프, 원자 입력 `line_list.csv`·`levels.csv`와 실행 설정을 쓴다. O와 T의 관련 신원·상태 연결이 확인된 범위만 결합한다.

W5와 원자 선 목록은 순차 판독하며 필요한 호출·셸·선만 보존한다. 대용량 로그는 관련 표식만 추출한다. 원본 전체 트리 탐색·해시·복사는 하지 않는다.

## §2. 과학 질문과 사전 확인

> `line_abs_heat`·`line_emit_cool`로 표현된 선 순율은 어느 선 집합의 어느 물질 상태와 선 평균장을 결합한 것인가? 같은 덤프의 구간 전달량과 등치할 조건은 충족되는가? 충족되지 않는 조건 중 무엇을 셸별로 확인하고, 무엇의 크기까지 저장 자료로 분리할 수 있는가?

기안 조회에서 다음을 원자료와 소스로 확인했다. 아래는 새 독립 계산 결과가 아니다.

- iter0010 manifest는 물질13·복사12·T12·소광23·방출23, iter0011은 물질14·복사13·T13·소광25·방출25다.
- O 로그의 W5 호출12는 복사12·물질12, 호출13은 복사13·물질13이다.
- 같은 로그의 복사12·13 등록은 각각 `q_lines=1391131`, `e_lines=2180286`이며 집합 해시가 두 세대에서 같다. 이것은 등록 집합 수이며 셸별 실제 합산 선 수가 아니다.
- W5는 Z=14·16·20, ion=1·2의 매핑 선과 셸10·15·19·24·30만 기록한다. 전 에너지 선 집합이 아니다.
- E 표에서 셸2의 L은 A10에서 약 \(+4.01229\times10^{44}\), A11에서 약 \(-1.58473\times10^{45}\)이며, 대응 \(Q_{\rm bb}\)는 각각 약 \(-4.75384\times10^{44}\), \(-4.75401\times10^{44}\ {\rm erg\,s^{-1}}\)다.
- 셸49의 저장 차는 A10에서 약 \(+1.22607\times10^{40}\), A11에서 약 \(-6.20014\times10^{39}\ {\rm erg\,s^{-1}}\)다. 일부 큰 차의 사례를 전 셸의 규모로 일반화하지 않는다.

새 계산은 이 수치의 원인 해석을 승계하지 않는다. 전체 셸의 장부와 대응량은 원 덤프에서 기존 판독 함수를 통해 얻는다.

## §3. 장부의 정의와 물질 출처

양의 L은 복사에서 물질로 향하는 부호로 정의한다.

\[
V_s=\frac{4\pi}{3}(r_{{\rm out},s}^{3}-r_{{\rm in},s}^{3}),\qquad
L_s=V_s(\mathrm{line\_abs\_heat}-\mathrm{line\_emit\_cool})
=4\pi V_s\sum_{l\in\mathcal E_s}d_l(k_l^{m}\bar J_l^{p}-e_l^{m}).
\]

여기서 \(m\)은 장부 물질, \(p\)는 선 평균장을 생산한 물질 이력이다. 복사 세대 번호와 물질 세대 번호는 별도 열로 둔다.

**선 집합.** line_jbar.c 약 398행의 `line_jbar_eset_build`는 매핑 여부와 무관하게 BB 영역 안의 선을 에너지 집합으로 만든다. line_jbar.h 약 25행의 영역은 \(1.49896229\times10^{14}\le\nu\le2.99792458\times10^{16}\ {\rm Hz}\)다.

실제 합산은 lumina_plasma.c 약 16509행에서 에너지 뷰를 순회하고 publication 주파수 경계의 엄격한 내부 조건을 추가한다. 비활성 원소는 명시적 영 기여로 처리한다. 따라서 등록 에너지 집합·실제 합산 집합·매핑 율계 집합·관측 소비 선 집합을 구별한다.

**물질 항.**
\[
e_l^{m}=\frac{n_{u,l}^{m}A_{ul}h\nu_l}{4\pi},\qquad
k_{{\rm raw},l}^{m}=\frac{\tau_{{\rm raw},l}^{m}\nu_l}{ct}.
\]
\(e\)는 적분 방출률의 입체각당 값, \(k\)는 주파수 적분 소광이며 각각 구간 방출계수·구간 소광과 단위가 다르다.

lumina_plasma.c 약 10250행의 `a209_upper_population_for_tau`는 소광과 같은 준위 소유권을 사용한다. 소유권과 공개 세대가 맞는 선은 NLTE 인구, 나머지 허용 경로는 해당 물질의 T·이온밀도·분배함수에 따른 LTE 준위 인구다. 매핑 여부만으로 모든 선을 NLTE 인구라고 부르지 않는다.

**음의 소광 처방.** line_net_rate.c의 `line_net_sobolev_material`은 `CMFGEN_SRCE_CHK`에서 \(\tau_{\rm raw}<-0.5\)이면
\[
k_{\rm eff}=10^{-10}/N_{\rm simultaneous},\qquad
\tau_{\rm eff}=k_{\rm eff}ct/\nu
\]
를 쓴다. 이 호출의 \(N_{\rm simultaneous}=1\)이다. 그 밖에서는 원 소광을 유지한다. 이는 봉인 생산식의 설명이며 이번 분석에 새 clamp/floor를 허용하는 조항이 아니다.

**에너지 가중.** superlevel 에너지는 원자 입력의 통계 가중치로 평균한다. lumina_plasma.c 약 16285행과 line_net_rate.c 약 386행에 따라
\[
r_l=\frac{(\bar E_u-\bar E_l)\,{\rm eV\_to\_erg}}{h\nu_l},\qquad
d_l=\begin{cases}1,&n_{\rm atom}\ge10^{30}\ \text{또는}\ |r_l-1|>0.5,\\r_l,&\text{그 밖}.\end{cases}
\]
유효성이 확인되지 않은 \(d_l\)을 1로 대체하지 않는다.

생산식은 선별 `fma(-k,Jbar,e)`를 계산하고 셸별로 합산·불확실성 판정을 수행한다. radeq_publication.c 약 178행은 **셸 합의 부호를 나눈다.** 저장 두 열을 선별 총흡수·총방출로 해석하지 않는다. physics_comparison.c 약 565행은 내부 `A_line`·`E_line`을 출력하지 않는다.

## §4. 선 평균장의 정의와 세대 연결

생산 호출12·13은 `CMFGEN_NONOVERLAP_SOBOLEV`다. 초기화의 `INIT_SHARED_GAUSSIAN`과 섞지 않는다.

\[
J_{c,l}^{p}=
\frac{\sum_{i:|x_i|\le4}e^{-x_i^2}\Delta\nu_iJ_i^{\rm fine,p}}
{\sum_{i:|x_i|\le4}e^{-x_i^2}\Delta\nu_i},\quad
x_i=\frac{\nu_i-\nu_l}{\nu_l v_D/c},\quad v_D=10^6\ {\rm cm\,s^{-1}}.
\]
\[
\bar J_l^{p}=\beta_N(\tau_{\rm eff}^{p})J_{c,l}^{p}
+e_l^{p}\frac{ct}{\nu_l}C_N(\tau_{\rm eff}^{p}).
\]

line_jbar.c 약 80행은 가우스 이산 평균, lumina_cmfgen.c 약 9778행은 그 평균의 소비, line_net_rate.c의 `line_net_cmfgen_exponx`는 \(\beta_N,C_N\)을 정의한다.

- \(|\tau|<10^{-3}\): \(C_N=1/2-\tau(1-\tau/4)/6,\ \beta_N=1-\tau C_N\).
- 그 밖의 \(\tau<40\): \(\beta_N=(1-e^{-\tau})/\tau,\ C_N=(1-\beta_N)/\tau\).
- \(\tau\ge40\): \(\beta_N=1/\tau,\ C_N=(1-\beta_N)/\tau\).

\(\tau=0\)에서도 이 식의 유한한 극한을 사용한다. \(e/k\) 나눗셈으로 바꾸지 않는다. W5의 `L_l`은 위 식의 **국소 평균장 항**이며 셸 에너지 장부 L이 아니다.

| 상태 | 장부·전달 계수의 물질 | 소비 복사 | 선 평균장 생산 물질 | 저장 근거 |
|---|---:|---:|---:|---|
| A10 | 13 | 12 | 12 | iter0010 manifest, W5 호출12, 등록 로그 |
| A11 | 14 | 13 | 13 | iter0011 manifest, W5 호출13, 등록 로그 |

lumina_cmfgen.c 약 11770행에서 선 평균장 생산·등록이 물질 갱신보다 앞선다. 약 6890행의 등록은 에너지 집합과 율계 뷰를 같은 복사 세대에 연결한다. lumina_plasma.c 약 16342행은 장부가 에너지 뷰를 소비하고, 약 17646행은 고정 T에서 새 물질 후보의 장부를 작성함을 보인다.

따라서 장부의 \(e^m,k^m\)와 이미 등록된 \(\bar J^p\) 안의 \(e^p,\tau^p\)는 같은 물질로 자동 정렬되지 않는다. 이것은 **이력 차의 존재**이며, 그 자체로 잘못된 세대 소비나 큰 차의 지배 원인을 뜻하지 않는다.

선행 JC 정의 단의 한계도 유지한다. 직접 선 침적이 없는 분기라도 비매핑 선의 상류 산란 몫 차감 실행은 별도 증거가 필요하다. \(J_c\)를 모든 선 영향이 제거된 연속장이라고 확정하지 않는다. 미저장 미세장 원배열을 거친 덤프장으로 대체하지 않는다.

## §5. 전달 쪽 잣대와 동일성 조건

같은 덤프의 대응량은
\[
Q_{{\rm bb},s}=4\pi V_s\sum_b
(\chi_{{\rm bb},sb}J_{sb}-\eta_{{\rm bb},sb})\Delta\nu_b
\]
다. 장·계수·체적은 셸 공이동계이며 결과 단위는 \({\rm erg\,s^{-1}}\)다.

lumina_plasma.c 약 10030행과 10715행, emissivity_publication.c 약 52행에 따라
\[
\chi_{{\rm bb},b}\Delta\nu_b
=\sum_{l\in b}\frac{\nu_l(1-e^{-\tau_{{\rm raw},l}^{m}})}{ct},\quad
\eta_{{\rm bb},b}\Delta\nu_b=\sum_{l\in b}e_l^{m}\beta_T(\tau_{{\rm raw},l}^{m}),
\]
\[
\beta_T(\tau)=
\begin{cases}1,&\tau=0,\\-\operatorname{expm1}(-\tau)/\tau,&\tau\ne0.\end{cases}
\]

이 전달 표현은 원 \(\tau\)를 쓰며 장부의 `SRCE_CHK`와 같은 치환을 하지 않는다. \(\beta_N\)의 근사 분기와 \(\beta_T\)의 계산식도 구별한다.

**같아야 하는 충분조건**은 동일 선 집합·활성 조건·주파수 배정·준위 인구·원자 상수·물질 이력, \(d_l=1\), 동일 소광 처방·동일 \(\beta\), 그리고
\[
\bar J_l=\beta_lJ_{b(l)}+e_l(ct/\nu_l)C_l,\qquad
\tau_l C_l=1-\beta_l
\]
가 함께 성립하는 것이다. 이때 선별로 \(k_l\bar J_l-e_l=\beta_l(k_lJ_b-e_l)\)다.

현재 자료에서는 물질 이력·평균장 정의·가중·음의 소광 처방·집합 조건의 차이가 허용된다. 수치 불일치만으로 배선 결함을 판정하지 않는다. 수치가 가까워도 정의 동일성은 입증되지 않는다.

BF·FF 항등 대응은 기존 판독기의 연결 확인을 재사용한다. 그 결함이 선 성분과 독립이면 선 수치를 지우지 않으며, 공통 단위·세대·격자 연결 결함이면 해당 상태를 정지한다.

C의 `cmfgen_sub.f` 약 3500행과 `sobjbar_sim.f` 약 130행은 연속 전달장과 광선별 Sobolev 가중을 구별한다. 이 로컬 구현을 공개 toy06의 실제 실행 설정이나 Lumina와 동일한 연산자로 간주하지 않는다.

## §6. 최소 추적 계산과 주 표

정본은 `line_net_provenance.csv`, 표시본은 같은 표의 `line_net_provenance.md`다. **전체 셸 100행, A10 셸10의 제한된 선 6행, 상태별 전역 요약 2행**을 한 표에 둔다. 범위 열로 전체 셸과 부분집합을 구별한다.

| 열 묶음 | 필수 내용 |
|---|---|
| 신원·범위 | 상태·셸·체적·속도 경계·전체 또는 제한 선·선 ID·입력 출처 |
| 장부 구성 | L·두 저장 선 열·집합 정의·인구 소유권·\(e,k,d,\bar J\)의 정의와 세대 |
| 전달 대응 | \(A_{\rm bb},E_{\rm bb},Q_{\rm bb}\)·장 정의·계수 세대·\(L-Q_{\rm bb}\)·차/\(\Delta\) |
| 추적 | W5 지원 여부·생산 호출·등록 연결·조건별 충족·불충족·미확인 |
| 귀속·판정 | 수치로 분리한 항·미귀속 잔여·적용 범위·미정의 사유·허용 주장 |

전체 장부의 원 선별 \(e,k,d\) 배열과 총흡수·총방출이 없으면 해당 수치 칸은 `null`이다. 소스 정의와 세대는 별도로 유지한다. W5에 없는 셸도 전체 L·Q와 소스 기반 출처를 갖는다.

W5 호출12·13의 저장 대상 5셸에서는 상태·신원·지원 조건과 \(\bar J=\beta J_c+\ell\), \(\beta=\beta_N(\tau_{\rm eff})\)를 확인한다. 표에는 사용 행 수·제외 사유·최대 정규화 잔차만 넣는다. 원 미세장으로부터 \(J_c\)를 재계산했다는 주장은 하지 않는다.

수치 분해 대상은 기존 대표선 `760321`, `632403`, `757019`, `1676767`, `1353542`, `1679119`로 고정한다. 결과를 보고 교체하거나 늘리지 않는다.

T의 채택 호출은 기존 판독 논리로 재확인한다. 물질12 인구는 호출1 `PRE_FL`, 장부가 소비한 물질13 인구는 `si10_commit.csv`의 `kind=level` 행에서 취한다. 후자는 채택 호출의 `(bundle, ne_iter, ce_pass)`에 해당하는 `si10_cevec.csv`의 `after`와 준위별로 정확히 일치해야 한다. `si10_commit.csv`의 `i`, `si10_cevec.csv`의 `i_fl`, 호출1과 채택 호출의 `si10_slmap.csv`를 연결하여 준위 인덱스의 완전성·유일성·원자 준위 대응을 확인한다. 채택 호출·commit의 T·전자밀도·물질 세대·덤프 복사 세대·iteration 연결도 유지하며, 번호가 13이라는 이유만으로 채택하지 않는다.

호출1 `PRE_FL`과 채택 commit의 인구 정규화·공개 인구 연결을 각각 확인한다. O의 iter0009·iter0010 `ion.csv`에서 셸10의 `Z14_nlte_q1_cm3`·`Z14_nlte_q2_cm3`를 사용하여, 준위 대응으로 구분한 Si II·Si III 인구 합이 각각 물질12·물질13 공개 인구와 §7 허용량 안에서 대응할 때만 계산한다. `Z14_stage_q1_cm3`·`Z14_stage_q2_cm3`로 NLTE 준위 인구 합을 대신하지 않는다. 이온 합의 일치만으로 준위별 commit 연결을 대신하지 않는다.

채택 `FINAL_FL`은 commit과 구별되는 호출 내부 인구다. `FINAL_FL=commit` 또는 `FINAL_FL=물질13 공개 인구`를 계산의 필수조건으로 두지 않으며, `FINAL_FL`을 장부 물질 인구로 대입하지 않는다. commit이나 채택 CE `after`가 누락·불일치하면 `FINAL_FL`로 대체하지 않고 §7에 따라 제한 선 계산과 의존값을 미정의로 남긴다.

개정 근거 [저자 실행: `cat /gpfs/kjhan/lumina/cmfgen_map/linenet_20261011/evidence/probe_commit_link-419767.out` 및 탐침 코드·판독기·관련 원장 행 조회; 독립 수치 재계산 미실행]: 운전자 탐침에서 채택 호출36의 commit Si II·Si III 합과 O 물질13 공개값의 상대차는 각각 약 1.17e-16·3.95e-16이며, `FINAL_FL` 합의 상대차는 약 3.51e-6·9.85e-7이다. 272개 준위 모두 commit과 `FINAL_FL` 값이 다르다. 호출36의 저장 phase는 `PRE_FL`·`RAW_SL`·`XFL`·`FINAL_FL`이며 commit은 별도 파일의 기록이다. 이 정정은 장부가 소비한 물질13의 기록 출처를 바로잡는다. F0–F4 정의·등록 여섯 선·§7 허용량·§8 분기 규칙·주장 상한은 변경하지 않는다.

각 선에서 \(J_c^p,\bar J^p\)는 호출12 기록, \(e^m,k^m,d^m\)은 물질13에 대응하는 원자·인구 자료를 쓴다. 다음 경로를 고정한다.
\[
F_0=4\pi Vd^m(k_{\rm eff}^{m}\bar J^p-e^m),\quad
F_1=4\pi V(k_{\rm eff}^{m}\bar J^p-e^m),
\]
\[
\bar J^*=\beta_N(\tau_{\rm eff}^{m})J_c^p+
e^m(ct/\nu)C_N(\tau_{\rm eff}^{m}),\quad
F_2=4\pi V(k_{\rm eff}^{m}\bar J^*-e^m),
\]
\[
F_3=4\pi V\beta_T(\tau_{\rm raw}^{m})(k_{\rm raw}^{m}J_c^p-e^m),\quad
F_4=4\pi V\beta_T(\tau_{\rm raw}^{m})(k_{\rm raw}^{m}J_{b(l)}^{12}-e^m).
\]

차는 순서대로 가중 \(F_0-F_1\), 국소항·\(\beta\)의 물질 이력 \(F_1-F_2\), 소광 처방·\(\beta\) 표현 \(F_2-F_3\), 장 표현 \(F_3-F_4\)에 귀속한다. 이 합은 \(F_0-F_4\)와 대응해야 한다.

이는 고정 \(J_c^p\)에서의 **등록 순서에 따른 산술 분해**다. \(F_2\)는 물질13의 자기일관 복사장이 아니며 \(F_1-F_2\)는 세대 변경의 전체 물리 효과가 아니다. \(F_2-F_3\)를 추가 자료 없이 소광 처방과 \(\beta\) 효과로 다시 나누지 않는다.

\(F_4\)는 해당 선의 전달 표현 기여이며 구간 전체 \(Q_{\rm bb}\)가 아니다. A10 셸10에는
\[
D_{\rm rest}=(L-Q_{\rm bb})-\sum_{l\in P}(F_{0,l}-F_{4,l})
\]
를 병기한다. 부분집합의 장부·전달 구성원 연결이 확인될 때만 나머지 선의 집계 차로 부르며, 그 밖에서는 미귀속 잔여로 남긴다. 상쇄가 있으므로 부분집합의 비율을 설명률로 부르지 않는다.

보조 `evidence.json`은 표에서 소비하는 소스 위치·입력 신원·선 연결·중간항·허용량·미정의 전파·직접시험·비용만 담는다. 새 독립 과학 표는 만들지 않는다.

## §7. 수치 기준과 미정의 전파

저장 수치의 산술 대조는 상쇄 전 절댓값 항의 합 \(S\)에 대해 \(\epsilon=10^{-10}S\)를 쓴다. \(S=0\)이면 정확한 영만 허용한다. 이는 물리 오차나 미저장 장의 불확실성 상한이 아니다.

W5의 장 오차 상한은 별도로 전파한다. 선의 물질 항을 고정하면 장부 항의 장 오차 상한은 \(4\pi Vd|k|\sigma_{\bar J}\)다. 공유 입력을 독립 오차로 가정하지 않으며 보수적 절댓값 합을 사용한다. 전체 선 상한이 없으면 전체 L의 물리적 부호 확실성을 새로 인증하지 않는다.

**진술 단위**는 상태·셸·성분 또는 상태·셸·선·파생식이다. 누락·중복·잘못된 연결·유효하지 않은 상태·단위 불일치·비유한·표현 소실을 이유와 함께 독립 기록한다. 유효한 음의 소광은 그대로 유지한다.

판독 중 비영 텍스트가 영으로 바뀌는 경우, 비영 곱·몫이 영으로 소실되는 경우, 유한 입력의 합·차·체적 가중·전역합·허용량·비가 비유한이 되는 경우도 미정의다. 정확한 입력 영·정확한 상쇄와 구별한다. 검사는 결과 표시 전의 모든 파생식에 적용한다.

한 선의 누락은 그 선의 의존 항과 여섯 선 완전합·잔여에 전파한다. 다른 선과 독립인 전체 덤프 L·Q는 유지한다. 한 구간의 선 방출 결함은 해당 셸의 선 방출·Q·차를 막지만 독립적인 장부 L이나 흡수 적분을 지우지 않는다.

한 장부 항의 결함이 모든 장부 성분을 지우는 공통 마스크는 금지한다. 관련 없는 BF·FF·침적 항의 국소 결함을 선 L의 미정의로 확대하지 않는다.

공통 원자 신원·세대 연결·셸 기하·주파수 격자·단위·열 의미 결함은 이를 소비하는 상태 전체를 정지한다. O와 T의 연결 결함은 제한 선 계산 전체를 멈추되 O만으로 정의되는 L·Q는 유지한다. 영향 범위를 한정할 수 없는 공통 판독 결함은 모든 의존 상태를 멈춘다.

전역 완전합은 필요한 모든 셸이 유효할 때만 산출한다. 부분합은 완전합으로 표시하지 않는다. 비가 미정의여도 유효한 원합은 보존하고, 필수 비의 미정의는 행 상태·최종 분기에 전파한다.

구조적으로 미저장인 전체 선별 배열·미세장·실제 관측 광선 교환은 사전등록된 `null`이다. 필수 추적 입력의 예상 밖 누락과 구별한다. 미정의를 영값·이전 세대·보간·작은 양수·절댓값 대체로 채우지 않는다.

## §8. 기대치와 해석 분기

예측은 다음과 같다.

1. 장부 물질과 선 평균장 생산 물질의 한 세대 차, 에너지 집합과 W5 지원 범위 차는 저장 기록과 소스로 재확인될 것으로 기대한다.
2. 같은 입력의 L·Q는 직전 수치와 같은 규모가 예상된다. 전 셸 차의 재발견 자체는 이번 단의 새 성과가 아니다.
3. 여섯 선의 가중·이력·처방·장 표현 항의 크기·부호·상대 순서는 예측하지 못한다. 양의 \(\tau\)라는 선행 기록도 이번 연결 확인을 대신하지 않는다.
4. 전체 장부 차를 위 네 항으로 완전히 분해할 자료는 확보됐다고 보지 않는다. 특히 A11의 물질14 전체 선 인구를 T의 물질13 기록으로 대신할 수 없다.
5. 관측 부족 \(\Delta=8.308031852615979\times10^{42}\ {\rm erg\,s^{-1}}\)에 대한 실제 기여율·응답 부호·개선량은 예측하지 못한다.

| 분기 | 사전 조건 | 허용 해석 |
|---|---|---|
| 0 `RESOLVED_SCOPED_DIFFERENCE` | 필수 추적과 수치가 유효하고, 등치 조건의 불충족 또는 등록 부분집합의 차가 분리됨 | 어느 정의·이력이 갈리는지 해당 범위에서 특정. 수치 귀속은 계산된 범위만 |
| 1 `NO_RESOLVED_SCOPED_DIFFERENCE` | 필수 추적과 수치가 유효하고 등록 범위에서 차가 분리되지 않음 | 해당 자료·허용량에서 미분리. 연산자 동일성·물리적 정확성의 증명은 아님 |
| 2 `INDETERMINATE` | 필수 연결·계산·파생값에 미정의가 남거나 예상 밖 자료 부재로 목표 미달 | 유효한 독립 결과를 보존하고 미달 범위·필요 자료를 명시 |

필수 범위는 전체 100셸의 L·Q·정의·세대 추적, 저장 W5 범위 확인, 등록 여섯 선의 연결과 산술 분해다. 사전등록된 구조적 미저장 항만으로 분기2를 강제하지 않지만, 이를 수치로 귀속 완료했다고 표시해서도 안 된다.

분기0에서도 전체 차의 지배 원인은 미식별로 남을 수 있다. \(\Delta\) 대비는 규모 안내다. 국소 공이동계 교환과 관측 대역 광도 사이의 순경계·시간 저장·일·프레임·주파수 이동 연결이 없어, 큰 차도 작은 차도 관측 결손 원인의 확정·배제가 되지 않는다.

## §9. 변경집합·직접시험·게이트

### 변경집합 - DET-LINE-NET-PROVENANCE (표)

| 경로 | 변경 |
|---|---|
| `scripts/det_line_net_provenance.py` | 새 추적 판독·등록 산술 분해·주 표·음성 대조 |
| `/gpfs/kjhan/lumina/cmfgen_map/linenet_20261011/tools/run_det_line_net_provenance_offline.slurm` | CPU 실행·자원 기록 |

### 변경집합 끝

`det_line_energy_corr.py`의 판독·적분·수치 유효성 함수, `det_si_line_input.py`와 기존 trace 판독기의 선 연결·생산식 함수를 무수정 재사용한다. 이전 분석 전체 명령·저장 산출 재검·기존 전체 시험군은 호출하지 않는다. 재사용 함수의 계약 부족은 새 추적 경계에서 처리하며 조용히 검사나 자료 범위를 생략하지 않는다.

직접시험은 새 코드의 음성 대조만 한다. 작은 정상 입력을 기반으로 **실제 파일 판독부터 연결·계산·행 상태까지의 경로**에 아래 결함을 각각 주입한다. 보조 함수를 단독 호출하거나 예상 상태 문자열만 바꾸는 시험으로 대신하지 않는다.

| 주입 결함 묶음 | 반드시 시연할 결과 |
|---|---|
| 세대·호출·선·준위의 오연결, 필수 행 누락·중복 | 잘못 연결된 진술과 그 의존값 거부, 독립 진술 유지 |
| commit과 `FINAL_FL`이 서로 다른 정상 입력에서 장부 물질 인구를 `FINAL_FL`로 대체 | 정상 입력은 commit=채택 CE `after` 및 공개 인구 연결을 만족하여 산출되고, 대체 주입은 commit으로 독립 산정한 F0–F4와의 불일치로 검출 |
| 채택 CE `after`와 commit의 준위별 불일치, 해당 행 누락·중복·인덱스 오연결 | 이온별 합이 보존되는 준위 재분배 결함도 거부; 제한 선 계산·완전합·잔여는 미정의, O만의 L·Q는 유지하고 최종 분기2 전파 |
| commit=채택 CE `after`를 유지한 채 물질13 공개 NLTE 이온 인구와의 연결만 훼손 | 공개 인구 연결 결함을 검출; 제한 선 계산·완전합·잔여는 미정의, O만의 L·Q는 유지하고 최종 분기2 전파 |
| 공통 단위·프레임·격자·신원의 누락과 불일치 | 각각 모든 의존 상태 정지, 무관 상태 유지 |
| 초기화 행을 Sobolev 행으로 사용, 지원 밖 행의 영 대입 | 연산자·지원 혼동 거부 |
| W5를 전체 에너지 집합으로 표기, 여섯 선 합을 셸 전체로 사용 | 범위 확장 거부 |
| W5 `L_l`을 에너지로 사용, \(\beta\) 이중 적용, 부호·체적·\(4\pi\) 오류 | 손계산 가능한 입력에서 실제 산술 경로의 불일치 검출 |
| \(F_2\)에 옛 물질, \(F_3\)에 유효 \(\tau\), \(F_4\)에 다른 세대 J 연결 | 서로 다른 입력값을 가진 시험에서 각각 오연결 검출 |
| \(d\) 무조건 1, 유효한 음의 \(\tau\) 제거, 영 \(\tau\) 나눗셈 | 가중·음의 소광·유한 극한의 잘못된 처리 검출 |
| 선·장부 한 성분의 비유한 | 국소 의존값만 미정의, 독립 성분 보존 |
| 텍스트·곱·몫의 영 소실, 합·차·체적 가중·허용량·전역합의 넘침 | 각 연산 종류에서 미정의와 전파 시연 |
| 필수 차/\(\Delta\)만 미정의, 한 셸 또는 한 선 누락 | 행·최종 분기2 전파, 유효 원합 보존, 불완전 완전합 거부 |

각 주입은 해당 계산을 실행할 수 있는 나머지 입력을 정상으로 둔다. 앞선 다른 결함에 걸린 것을 목표 결함 검출로 세지 않는다. 수치가 다른 물질·복사 입력을 써서 오연결이 우연히 통과하지 않게 한다.

| 게이트 | 막는 실재 실패 |
|---|---|
| G-INPUT | 다른 세대·집합·준위·단위의 자료를 동일 상태로 결합하는 실패 |
| G-CALC | 산술 경로 오류와 국소·공통 미정의를 영값·부분합·유효 판정으로 숨기는 실패 |
| G-RUN | 허용 밖 자원·생산 변경 또는 필수 증거 없는 실행을 등록 결과로 채택하는 실패 |

기존 기계 프리플라이트는 기존 도구로 집행한다. 새 게이트 엔진·쓰기 결함 모사·과학 결정표의 별도 구현은 만들지 않는다.

## §10. 비용·실행·보존

GPU·생산 변경·생산 실행·새 T 또는 인구 풀이·전달 재실행은 모두 0이다.

a10 CPU Slurm에서 작업명 `LUMINA_` 접두, `--ntasks=1`, `--cpus-per-task=1`, `--mem=4G`, `--gres=gpu:0`, `--time=00:03:00`, `--exclude=syn102,syn03`, `--no-requeue`를 사용한다. 실제 할당에 GPU가 없어야 한다.

전체 상한은 **할당 벽시계 600 s·CPU 600 core·s·신규 저장 30 MiB**다. 직접시험·기계 검사·수정 실행·감사 재계산을 포함한다. OpenMP·BLAS 스레드는 각각 1이다.

작업 안에서 `module purge`와 `module load gnu13/13.2.0`을 실행하고 실제 Python·NumPy 판본을 기록한다. 로그인 노드에서는 파일 조회만 한다.

```bash
python3 -B scripts/det_line_net_provenance.py selftest
python3 -B scripts/det_line_net_provenance.py analyze --out /gpfs/kjhan/lumina/cmfgen_map/linenet_20261011/results/run1
```

두 명령은 위 Slurm 할당 안에서만 실행한다. 재현은 같은 입력·코드와 새 출력 경로를 사용한다. 상한에 도달하면 미완료 범위를 기록하며 자동 증액하지 않는다.

원자료·기존 결과·실패 결과를 보존하고 존재하지 않는 실행 경로에만 출력한다. 일반 파일 읽기·쓰기만 사용하며 하드링크는 사용하지 않는다.

소비한 파일의 경로·크기·SHA-256, 코드 판본, 선택 키, 명령·작업 번호·종료 상태·비용을 보존한다. 큰 입력의 해시는 순차 판독과 함께 산출하여 중복 읽기를 줄인다.

## §11. 판정·종료감사·주장 상한

운전자는 전체 셸의 정의·세대 연결, 여섯 선의 제한된 수치 귀속, 미귀속 잔여, 분기와 미달 범위를 판정한다. 구조적 미저장을 해소한 것처럼 쓰거나, 표본에서 확인된 기전을 전체 장부의 지배 원인으로 확대하지 않는다.

폐합에는 주 표·필수 증거·직접시험·운전자 판정·astra 종료감사가 필요하다. 분기2는 목표 미달을 명시한 기록 폐합이며 과학 목표 달성으로 표시하지 않는다.

run1은 개정 전 계약에 따른 등록 계산의 분기 2 기록으로 보존한다. 원래 산출물·직접시험 증거·작업 로그와 미달 범위를 유지하며, 개정 후 결과로 덮어쓰거나 탐침으로 재분류하지 않는다. 개정 후 계산은 개정 계약과 대응 코드로, run1과 다른 아직 존재하지 않는 새 실행 경로에서 수행한다. 새 실행의 계약·코드 판본, 출력 경로와 증거를 run1과 구별하여 기록하고, 그 결과는 기존 분기 규칙과 폐합 요건에 따라 별도로 판정한다. 새 실행의 성공 여부는 run1의 분기 2 기록을 소급 변경하지 않는다. 이 개정은 기존 자원 상한을 늘리거나 새 실행을 자동 승인하지 않는다.

종료감사는 수치의 원자료 연결, 장부와 전달량의 부호·집합·물질 이력, 산술 분해의 실제 범위, 미정의의 독립 유지·전파, \(\Delta\) 연결의 한정을 우선한다. astra는 계약 기안자라는 이해관계를 밝히고 저장 자료 대조와 독립 실행 재현을 구별한다.

생산 승격·물리적 정답·수리·새 T 계산·자기일관 효과·스펙트럼 개선·철족 일반화를 선취하지 않는다. 장부 정의의 차를 관측 결손 원인으로 단정하지 않는다. 어느 분기도 다음 실행을 자동 승인하지 않는다.

기안자는 공유 맥락 전체를 읽고 현재 디렉터리와 프로젝트 신원을 확인했다. 관련 소스·manifest·선별 원장 행·로그·선행 표를 조회했으며 파일 변경·수치 계산·해시 재산출·직접시험·제출은 하지 않았다.

### 집행 기록 (운전자 추가 — 정본 장부, 개정16 ①; 위 기안 기록은 그대로 둔다)

증거 접두 `B=/gpfs/kjhan/lumina/cmfgen_map/linenet_20261011`. 작업 번호·경과는 `B/JOBS.txt`·`B/evidence/sacct_offline_jobs.txt`, 게이트 집행은 `B/evidence/GATE_RECORD.json`.

| 항목 | 담당(실제) | 상태·증거 |
|---|---|---|
| user 결정 | user | 「선 순율 L 의 정의·세대 추적」(`B/evidence/USER_DECISION_2026-10-11.txt`), DET-LINE-ENERGY-CORR 폐합 후 |
| 기안 | astra(자기보고 「gpt-6-astra」) | 3b3bdc8 |
| 사전감사 | — | 생략(25-ab: GPU 0·생산 변경 0) |
| 기계 게이트(기안) | 운전자 | 기안 브리프 canonical·위생 rc 0(a10 419744), 기안 prereg·referents·canonical rc 0(419746) |
| 구현 | 운전자(Opus 5.5) · 코드 감사 없음(25-z) | ff3f517 — `scripts/det_line_net_provenance.py`; 러너 `B/tools/run_det_line_net_provenance_offline.slurm` |
| 직접시험(§9) | 운전자 | 커밋 전 개발 작업 419757·419758·419759·419762(실패, 결과 아님)·419763·419765; 커밋본 40/40(run1 작업) |
| 등록 계산 run1 | 운전자 | a10 419766 `B/results/run1/`: **분기 2 INDETERMINATE** — 채택 FINAL_FL 공개 물질13 연결 실패(`TRACE_FINAL_FL_VS_COMMIT`·`FINAL_FL_VS_MATERIAL13`), 등록 6선 미산출; 전체 셸·W5·전역 행 산출 |
| 탐침 | 운전자 | a10 419767(`B/tools/probe_commit_link.py`): commit(= 채택 CE after) 이온 합 상대차 ~1e-16, FINAL_FL 3.5e-6·9.9e-7, 272 준위 모두 commit ≠ FINAL_FL |
| 개정 1 | astra | `B/astra_rev1.md` 쌍 1·2 + `B/astra_rev1b.md` 쌍 3(rev1 쌍 3 은 한국어 아닌 줄로 끝나 미적용) → e5374b4; 행 diff OLD 1행만 소실; 개정 게이트 prereg·referents·canonical rc 0(419777), 개정 브리프 게이트 rc 0(419768·419773) |
| 개정 반영 구현 | 운전자 | e827252(물질13 = commit, 준위별 CE after·인덱스·세대 연결, FINAL_FL 기록만); 직접시험 42/42(419776) |
| 등록 계산 run2 | 운전자 | a10 419778 `B/results/run2/`(12 s): 게이트 지적 0, 미정의 0; **분기 0 RESOLVED_SCOPED_DIFFERENCE** — 물질 이력 조건 불충족(13/12·14/13), G1 3선 F0/F4 1.2e4–1.1e5·F1−F2 몫 99.97 %, 6선 Σ(F0−F4) −21.9Δ, 셸10 L−Q +31.2Δ, D_rest +53.0Δ; 판독 탐침 419780 |
| 판정 | 운전자(구현·판정 겸임) | `docs/VERDICT_DET_LINE_NET_PROVENANCE_2026-10-11.md` — 1회차 3ea0c8d(run2 정본, 판정 게이트 a10 419782 rc 0), 1회차 반영판(§8 신설·run3 정본) |
| 종료감사 1회차 | astra(자기보고 「gpt-6-astra」, 계약 저자 이해관계) | 판정 3ea0c8d 대상: **폐합 불가 · «폐합 차단» 1 · «기재로 족함» 2** — W5 검사·생산식 연결이 유효성·게이트·분기에 미전파, §7 장 오차 상한 미산출(+ selftest 요약 변수, 해시 범위); 독립 실행 재현 BLOCKED(`B/astra_closeout.md`); 발주 게이트 rc 0(419785) |
| 1회차 반영 | 운전자 | 전건 반영(1·2회차 규약): 판독기 642c09c(직접시험 49/49, 419794); **run3 419795** `B/results/run3/`: 분기 0, 6선 표 run2 와 바이트 동일, W5 검사 실패 0, 장 오차 상한 ≤ |F0| 의 2.2e-7; 판독 탐침 419796 |
| 종료감사 2회차 | astra(자기보고 「GPT-6」, 세부 ID 미확인; codex 헤더 model gpt-6-astra) | 판정 4075616 대상: **폐합 불가 · «폐합 차단» 1**(1회차 지적 1 잔존: 등록 선 W5 행 실패의 의존값·6선 합·D_rest 미전파), 1회차 2·3 해소; 독립 실행 재현 BLOCKED(`B/astra_closeout_r2.md`); 발주 게이트 rc 0(419802; 419801 은 계약 미커밋으로 rc 1) |
| 2회차 반영 | 운전자 | 전건 반영(1·2회차 규약): 판독기 7aaf3ac(직접시험 50/50, 419818); **run4 419819** `B/results/run4/`: 분기 0, 표 두 파일 run3 과 바이트 동일 |
| 종료감사 3회차 | astra | 대기 — 3회차부터 «폐합 차단» 만 반영(개정18 ③); 1·2회차 «폐합 차단» 2건 모두 판독기 검사 전파, 물리량·판정치 불변(개정18 ④ 계수 2/3) |
| 비용 | 운전자 | GPU 0; 오프라인 a10 27 작업 할당 246 s · 246 core·s / 600 s · 600 core·s(run4 까지); 신규 저장 결과 1109 KiB + 로그 / 30 MiB |

**이탈(기재):** ① run1 분기 2 후 개정 1(물질13 출처 정정) — run1 기록 보존, 개정 후 새 경로 run2. ② 개정 1 쌍 3 미적용·보완 발주. ③ 커밋 전 직접시험 작업 4건 실패(시험·판독기 결함 수정). ④ 로그인 노드에서 `python3 --version` 1회 실행(연산 없음, 규약 위반).

## §12. 기계 선언

참조는 기존 실재 파일만 가리킨다. 분기 예시는 기존 프리플라이트 문면 검사에만 사용한다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 - DET-LINE-NET-PROVENANCE (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_line_net_provenance\\.py|/gpfs/kjhan/lumina/cmfgen_map/linenet_20261011/tools/run_det_line_net_provenance_offline\\.slurm)",
    "symbol": "det_line_net_provenance",
    "roots": [
      "scripts/det_line_net_provenance.py",
      "/gpfs/kjhan/lumina/cmfgen_map/linenet_20261011/tools/run_det_line_net_provenance_offline.slurm"
    ],
    "expected_extra": [
      "scripts/det_line_net_provenance.py",
      "/gpfs/kjhan/lumina/cmfgen_map/linenet_20261011/tools/run_det_line_net_provenance_offline.slurm"
    ]
  },
  "branches": {
    "regimes": [[0,0],[1,1],[2,2],[9,9]],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,2) for x in v)",
      "c0": "sum(1 for x in v if x == 0)",
      "c1": "sum(1 for x in v if x == 1)"
    },
    "rules": [
      {"name":"RESOLVED_SCOPED_DIFFERENCE","predicate":"valid and c0 == 1"},
      {"name":"NO_RESOLVED_SCOPED_DIFFERENCE","predicate":"valid and c1 == 1"},
      {"name":"INDETERMINATE","predicate":"not valid or (c0 == 0 and c1 == 0)"}
    ],
    "adversarial_fixtures": [
      {"name":"difference","mix":[[0,1]]},
      {"name":"no_difference","mix":[[1,1]]},
      {"name":"indeterminate","mix":[[2,1]]},
      {"name":"invalid","mix":[[9,1]]},
      {"name":"missing","mix":[]},
      {"name":"multiple","mix":[[0,1],[1,1]]}
    ]
  },
  "references": [
    {"path":"docs/PREREG_DET_LINE_ENERGY_CORR_2026-10-11.md"},
    {"path":"docs/VERDICT_DET_LINE_ENERGY_CORR_2026-10-11.md"},
    {"path":"docs/VERDICT_DET_JC_DEFINITION_2026-10-11.md"},
    {"path":"docs/VERDICT_DET_SI_LINE_INPUT_2026-10-11.md"},
    {"path":"docs/VERDICT_DET_SI_BB_COUNTERFACTUAL_2026-10-11.md"},
    {"path":"scripts/det_line_energy_corr.py"},
    {"path":"scripts/det_si_line_input.py"},
    {"path":"scripts/det_si_trace_readout.py"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/linenet_20261011/evidence/USER_DECISION_2026-10-11.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/energyx_20261011/results/run7/shell_energy_correspondence.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/energyx_20261011/results/run7/evidence.json"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag9_absdef_20261011/DIAGNOSIS.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_net_rate.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_jbar.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_jbar.h"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/radeq_publication.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/emissivity_publication.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/physics_comparison.c"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/subs/sobjbar_sim.f"}
  ]
}
```