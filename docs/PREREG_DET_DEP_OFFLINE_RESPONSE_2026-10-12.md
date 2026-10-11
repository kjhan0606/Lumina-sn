# 사전등록 - DET-DEP-OFFLINE-RESPONSE: 고정 포획에서 직접 침적 원천의 광도 응답

2026-10-12 · astra 읽기 전용 기안  
상태: 원자료와 소스 조회 완료. 구현·빌드·직접시험·계산·제출·판정 미집행.

## §0. Q-GOAL·Q-LEAN 자기답

**Q-GOAL: PASS.** 최종목적은 Monte-Carlo 수송으로 공개 CMFGEN과 비교 가능한 SN Ia 스펙트럼을 얻는 것이다. 이번 단은 그 중간 시험으로, O의 물질13·복사13 포획을 기준으로 직접 침적 원천의 추가가 구간 J와 관측 광도를 얼마나 바꾸는지 GPU 없이 측정한다. 이를 생략하면 요청 침적량을 탈출 광도로 간주하거나, 광도 응답도 모르는 상태에서 생산 배선 변경과 GPU 비용을 먼저 지출하게 된다.

관측 가능한 종료 산출은 **가지별 원천 소비·구간 J 변화·바깥 경계 광도·관측 광도·대역 광도·공개 대비를 담은 표 하나**다. 이 표로 보류된 GPU 계약의 재개를 권할지, 목적과 원천 처방을 개정할지, 현재 처방의 시험을 폐기할지 판단한다.

**Q-LEAN: APPROPRIATE.** 물리 가지는 기준 A와 연결 B 두 개뿐이다. O 원 포획의 관측 재생은 기준 자격 확인이며 제삼의 물리 가지가 아니다. diag10의 선 집계·정적 형식해와 봉인 생산 관측 재생을 재사용한다. 새 부분은 고정 계수 해법, 직접 원천 구성, 파생 포획 연결, 외표면 적분, 주 표 판독과 그 음성 대조다.

삭제 대상은 생산 소스 변경, GPU 상태 재현, 물질 장부 재평가, 원소별 순위표, 새 선별 기여 추적, 저장 산출 재검 장치, 쓰기 결함 모사, 결정표 재계산 장치다. 선 기록은 고정 계수 구성에 한 번 순차 소비하고, 재생기는 자기 입력 검사를 그대로 수행한다.

보조 산출의 소비처는 주 표의 수치·기준 재현 편차·미정의 사유·입력 신원·직접시험 적격성·비용으로 한정한다. 게이트는 공통 검사 결과를 사용하는 판정 범주이며 별도 검사기를 늘리는 단위가 아니다.

## §1. 권한·작업 기반·보존

기안·개정·모든 감사는 astra, 구현·직접시험·게이트·실행·판정은 운전자다. GPU 0·생산 변경 0이므로 경량 사전감사는 생략한다(25-ab). 코드 감사는 없다(25-z). 종료감사는 astra가 수행하며, 계약 기안자가 자기 계약에 따른 판정을 감사한다는 이해관계를 밝힌다.

| 약칭 | 위치 |
|---|---|
| P | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn` |
| B | `/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012` |
| F | `/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/offline` |
| O | `/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1` |
| C | `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006` |
| D | `/gpfs/kjhan/lumina/cmfgen_map/diag10_energybudget_20261012` |
| U | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06` |

권한 근거는 B의 두 `USER_DECISION` 파일이다. 둘째 선택의 오프라인 선행을 적용한다. 선택지 설명과 운전자 확인 문구는 감사된 사실로 승계하지 않는다. 보류된 GPU 계약의 실행 권한·수용 기준·소비 장부 정의도 자동 승계하지 않는다.

필수 입력은 O의 포획 14개 payload와 manifest, 저장 관측 스펙트럼, 실행 exports, 빌드 신원, `input/model/deposition_cmfgen.csv`, C의 봉인 소스·CPU 빌드 처방, U의 공개 스펙트럼·광도 표다. D의 결과는 선행 척도와 이식 근거로만 사용한다.

생산 소스·O 런·원 포획·선행 결과는 수정하지 않는다. 신규 파일과 실패 기록은 F 아래에 둔다. 입력 목록에 한정해 크기·SHA-256과 판본 연결을 기록한다. 전체 트리 해시·복사, 하드링크, 파일시스템 고유 기능을 사용하지 않는다.

## §2. 과학 질문·확인 사실·범위

> O의 물질 상태·소광·기존 방출·선 원천·경계를 고정하고, 요청된 직접 침적의 Planck 모양 원천을 추가하면, 같은 정적 구간 전달식의 해 J와 생산 관측 재생의 광도는 A에 비해 얼마나, 어느 셸·구간·대역에서 달라지는가?

기안 조회에서 다음을 확인했다. 수치는 저장 원자료 인용이며 이번 기안의 독립 계산 결과가 아니다.

- O 포획은 `NS=50`, `NB=1234`, 물질13·복사13·iter11·조립13·tau26이다. 소비 선쌍은 `19620090`, payload는 `670663148`바이트다.
- O는 직접 원천과 침적 게이트를 요청했다. 봉인 DET 호출 뒤 반환 때문에 레거시 setter 호출을 지나지 않는다는 D의 배선 판단은 관련 소스와 대응한다.
- D의 이식 코드는 저장 J에 형식해를 **한 번** 적용한다. 저장 최대 대칭 상대차는 `0.0003912629371059669`다. 재수렴이나 광도 오차 상한을 측정한 결과가 아니다.
- `eta_c.f64`는 BF/FF 분자와 직접 침적의 합이며, 전자산란과 선 항을 포함하지 않는다.
- 봉인 재생기는 저장 J를 사용해 전자산란과 비매핑 선 산란을 다시 구성한다. J의 전달 해를 구하거나 물질 상태를 갱신하지 않는다.
- DET-OBS-CAPTURE의 과거 다른 포획에서는 scalar 재생과 저장 관측 CSV의 바이트 동일성이 기록돼 있다. 이번 O의 동일성을 대신하지 않는다.
- DET-OBS-WRITER·DET-OBS-LINEDECOMP 판정은 관측 원천 보존과 고정 상태 출력 귀속의 선례다. 그 결과를 O의 침적 효과나 물리적 정답으로 옮기지 않는다.

이번 단은 **조건부 원천 감도**를 가른다. 생산 setter 배선 검증, 같은 상태의 물질 가열·냉각 장부, 실제 비열적 에너지 소비, 물리적 이중 계산 해소, 인구·T의 자기일관 응답은 가르지 못한다.

## §3. 두 가지와 직접 원천

A와 B의 부모는 같은 O 포획이다. T·전자밀도·기하·시각·주파수 격자·소광·BF/FF 방출·선 목록·tau·S·매핑·epsilon·안쪽 경계를 고정한다. 구간 J와 그 J를 소비하는 산란 항만 가지별로 바뀐다.

셸 체적과 입력 침적을 다음과 같이 정의한다.
\[
V_s=\frac{4\pi}{3}(r_{{out},s}^3-r_{{in},s}^3),\qquad
d_s=\mathrm{heating\_rate}_s,\qquad D_s=V_sd_s.
\]
`heating_rate`의 단위는 `erg/s/cm3`다. 셸 번호를 결합 키로 사용하며 행 순서에 의존하지 않는다.

요청 분율 \(f\)는 O의 유효 설정과 봉인 기본값으로 확정한다. 이번 단의 등록값은 \(f=1\)이다. 실제 설정이 다르면 조용히 1로 바꾸지 않고 입력 조건 불일치로 처리한다.
\[
b_s=\sum_b B_\nu(\nu_b,T_s)\Delta\nu_b,\qquad
\eta_{{dep},sb}=\frac{f d_s B_\nu(\nu_b,T_s)}{4\pi b_s}.
\]

정규화는 포획의 실제 `nu.f64`·`dnu.f64`만 사용한다. 무한 주파수 Planck 적분이나 공개 대역으로 재정규화하지 않는다. \(d_s\ge0\), \(T_s>0\), \(\Delta\nu_b>0\), 유한한 \(b_s>0\)를 요구한다.

- **A:** 직접 배정 0, 직접 원천 0, 기존 고정 원천과 산란으로 J를 푼다.
- **B:** 직접 배정 \(fD_s=D_s\), 기존 고정 원천에 \(\eta_{\rm dep}\)를 더하고 J를 푼다.
- 두 가지 모두 새 물질 갱신과 새 비열적 인구 풀이의 호출 수는 0이다.

A의 미직접배정 \(D_s\), B의 미직접배정 0은 **이번 반사실의 배정 표기**다. A가 D를 실제 물질 가열로 소비했다거나 B의 기존 물질 가열이 제거됐다는 뜻이 아니다. 물질 침적 소비량·비열적 이력 소비량·물리적 분할 잔차는 두 가지 모두 미정의다.

## §4. 이식 연산자·산란·해의 수렴

D의 `r1_transfer_budget.py`에서 선 기록 판독·구간 집계·`segment`·`formal_once`를 재사용한다. 과거 결과 표를 새 가지의 입력 계수 대신 쓰지 않는다.

포획 선의 구간 소광 기여를 \(w_l\)라 할 때
\[
w_l=\frac{(1-e^{-\tau_l})\nu_l}{ct\,\Delta\nu_b}.
\]
작은 tau 처리와 선의 지원 조건은 봉인 코드에 맞춘다. 매핑 선은 저장 S 전체가 고정 원천이다. 비매핑 선은 \(\epsilon S+(1-\epsilon)J\)로 분리한다. 매핑 선에 기록된 epsilon 0은 산란 분율이 아니라 sentinel이다.
\[
\begin{aligned}
\chi&=\chi_a+\chi_e+\sum_lw_l,\\
\eta_0&=\eta_{\rm BF/FF}+\sum_{\rm mapped}w_lS_l+
                      \sum_{\rm unmapped}w_l\epsilon_lS_l,\\
\chi_{\rm sc}&=\chi_e+\sum_{\rm unmapped}w_l(1-\epsilon_l),\\
J_X&=\Lambda[(\eta_0+\eta_{{dep},X}+\chi_{\rm sc}J_X)/\chi].
\end{aligned}
\]
산란 계수는 양의 항을 직접 합산하며, 큰 두 합의 차를 음수 절단으로 보정하지 않는다.

봉인 형식해의 8개 core ray, 셸 바깥 반경의 접선 ray, 중점 경로, 양방향 표본과 `ray_w=p`를 유지한다. 이는 관측기의 256개 ray·이동매질 공명 전달과 다르다. 이식 결과를 생산 관측 연산자의 해라고 부르지 않는다.

새 clamp/floor는 금지한다. 재사용 코드의 음수 광학깊이 절단·중점 위치 보정·영 가중 대체가 필요한 입력은 보정 전에 거부한다. 구간 번호를 정하는 정수 내림은 물리량 보정과 구별한다. 진공 셀은 \(\chi=0\)이고 모든 원천도 0인 경우만 투명 전달로 정의한다.

**해법.** 고정 계수 식은 구간별 50차원 선형식이다. 같은 `formal_once`를 영 경계의 단위 원천에 적용해 선형 부분을 만들고, 원래 경계의 비동차 항을 별도로 계산한다.
\[
M_bJ_{X,b}=h_{X,b},\qquad M_b=I-K_b.
\]
공통 행렬 분해를 A·B에 재사용한다. 음의 해를 잘라내지 않으며, 실패 시 다른 물리 처방으로 대체하지 않는다. 잔차 보정은 최대 8회다.

연속 두 확인에서 행렬 잔차와 **원래 `formal_once`에 다시 넣은 잔차**의 셀별 대칭 상대값이 모두 \(10^{-8}\) 이하여야 한다. 보정 전후 J 변화도 같은 기준을 만족해야 한다. 양쪽이 정확히 0일 때만 대칭 상대값을 0으로 정의한다.

작은 잔차를 곧바로 작은 해 오차로 읽지 않는다. 비음수 \(K\)에 대해 \(Mw=\mathbf1\)을 풀고, 유한한 \(w>0\)와 반올림을 포함한 \(Mw\)의 양의 하한 \(\ell\)을 확인한다. 구성 행렬의 잔차 상한 \(r^\uparrow\)에 대해
\[
|e_s|\le w_s\max_i(r_i^\uparrow/\ell_i)
\]
를 사용한다. 이 상한과 형식해 재대입 편차를 함께 기록한다. 조건 확인 실패는 그 구간의 해 미정의다. 이는 구성한 이산식의 수치 확인이며 연산자 이식·격자·물리 오차 인증이 아니다.

## §5. 기준 대조·파생 포획·실제 소비

**대조를 먼저 적용한다.**

1. 원 O 포획을 봉인 CPU 재생기로 읽어 `scalar.csv`가 저장 `lumina_spectrum_det_obs.csv`와 바이트 동일해야 한다. 재생기의 scalar/vector 일치·성분 폐합·비유한 검사도 통과해야 한다.
2. 저장 J에 이식 형식해를 한 번 적용한 최대 대칭 상대차는 \(10^{-3}\) 이하여야 한다.
3. 수렴 A와 저장 J의 최대 대칭 상대차는 \(10^{-2}\) 이하, \(V_s\Delta\nu_b\) 가중 절댓값 차의 합을 같은 가중의 \(|J_A|+|J_O|\) 합으로 나눈 값은 \(10^{-3}\) 이하여야 한다.
4. A 재생과 O 저장 관측의 \(\int|L_{\lambda,A}-L_{\lambda,O}|d\lambda/\int L_{\lambda,O}d\lambda\)는 전체 저장 지원·공통 지원·각 등록 대역에서 각각 1% 이하여야 한다.

이는 생산과의 대조 자격 기준이다. 1회 잔차에서 수렴 A의 편차를 추정하지 않는다. 공통 대조 실패면 B 해석을 중단한다. 특정 대역만 실패하면 그 대역의 결론을 미정의로 유지한다. 기준을 결과에 맞춰 완화하지 않는다.

**정당한 봉인 경로.** lumina_cmfgen.c 약 6007행의 `det_obs_capture_seal_fixture`는 `LUMINA_DET_OBS_SELFTEST` 빌드에서 파생 자료를 `fixture=1`로 봉인한다. 약 5962행의 검사는 최종 디렉터리 이름 `det_obs_capture`, `COMMITTED`, 14개 파일의 크기·SHA-256을 요구한다. 약 6033행의 재생은 이 검사를 먼저 호출한다.

F의 새 실행 디렉터리에 A·B의 payload를 일반 파일 복사로 만들고 다음만 바꾼다.

- A: `J.f64`.
- B: `J.f64`, `eta_dep.f64`=\(\eta_{\rm dep}\), `eta_c.f64`=\(\eta_{c,O}+\eta_{\rm dep}\).
- 나머지 payload는 부모와 바이트 동일해야 한다. 원 O의 직접 레인이 전부 0인지 입력에서 확인한다.

봉인 함수 호출 전 파생 대상과 manifest의 신규성을 확인한다. 원 manifest를 복사해 생산 신원을 가장하지 않는다. 부모 manifest 해시·바뀐 배열·가지·해법 신원은 별도 실행 기록에 둔다. meta의 세대는 부모 상태를 가리키며 새 생산 commit을 뜻하지 않는다.

봉인 함수의 `fixture=1`은 해석용 파생 자료임을 명시하는 표식이다. 무결성 검사를 끄거나 재생기의 hash 검사·선 판독·성분 폐합을 수정하지 않는다.

**소비량.** 셸마다 다음을 별도로 적분한다.
\[
E_s^x=4\pi V_s\sum_b\eta_{sb}^x\Delta\nu_b.
\]
명목 \(\eta_{\rm dep}\), 실제 전달 원천의 가산분 \(\chi(S_{\rm fixed,B}-S_{\rm fixed,A})\), 재생기가 읽는 `eta_c`의 가산분, 직접 레인 `eta_dep`를 구별한다. 양수 요청이 실제 합에서 사라졌으면 소비 성공이 아니다.

B의 각 소비량은 \(D_s\)와 대칭 상대차 \(10^{-10}\) 이내여야 한다. A는 정확한 영 직접 소비를 요구한다. BF/FF 원천 보존은 `eta_c-eta_dep`의 가산·감산 반올림 상한 안에서 확인한다. 유효한 불일치는 오프라인 구성 결함이지 생산 배선 실패가 아니다.

## §6. 바깥 경계·관측 광도·공개 비교

**바깥 경계 광도**는 가지별 수렴 원천과 같은 구간 소광을 써서 실제 \(R=r_{{out},49}\)까지 전달한 정적 구간 진단량이다.
\[
L_{\nu,{\rm out}}=8\pi^2\int_0^R I_\nu^+(R,p)p\,dp,\qquad
L_{\rm out}=\sum_bL_{\nu_b,{\rm out}}\Delta\nu_b.
\]
외부 입사는 0이다. 안쪽에 닿는 ray는 원래 Planck 경계를 쓰고, 나머지는 반대편 외표면부터 전달한다. 셸의 실제 교차 길이와 `segment`를 재사용한다. 마지막 셸 중점 강도·Planck 주입 규모·J의 정규화 가중치를 광도로 대신 쓰지 않는다.

p 구간은 0·안쪽 반경·모든 바깥 반경으로 나누고 각 구간에 4개 중점, 확인용 8개 중점을 쓴다. 실제 환대 가중치를 사용한다. 두 적분의 대칭 상대차 \(10^{-3}\) 초과는 해당 경계량 미정의다. 이 차는 경험적 적분 민감도이며 엄밀한 오차 상한이 아니다.

이 외표면 적분은 중점 J 연산자의 보존 항등이 아니다. 새로운 순안쪽 경계·시간 항·에너지 폐합을 만들지 않는다. 이동매질 observer와의 차를 전부 좌표 효과로 귀속하지 않는다.

**관측 광도**는 수정하지 않은 봉인 재생기의 scalar와 일곱 성분 출력을 사용한다.
\[
L_{\nu,{\rm obs}}=8\pi^2\int I_{\nu,{\rm obs}}(p)p\,dp.
\]
생산 코드의 주파수 변환·연속 전달·공명 도약·ray 가중치를 그대로 사용한다. 직접 탈출 레인과 전체 B-A는 별개다. 후자는 J 변화에 따른 전자산란·비매핑 선 산란 응답도 포함한다.

`flux` 열은 이 출력에서 `erg/s/Ang`의 \(L_\lambda\)다. 새 거리 인자를 곱하지 않는다. 변환이 필요한 경우 \(L_{\lambda,{\rm Ang}}=L_\nu c\lambda_{\rm cm}^{-2}10^{-8}\)을 적용한다.

공개 비교는 `spectra_toy06_cmfgen.txt`의 **19.480일 열**을 사용한다. 공통 지원은 O·A·B·공개 스펙트럼 지원의 교집합이다. 각 곡선을 자기 절점과 적분 끝점을 포함한 구간별 선형 함수로 적분하며 외삽하지 않는다.

등록 대역은 `2500-4000`, `4000-6000`, `6000-8000`, `8000-10000`, `10000-25500 Ang`이다. 대역 전체를 지원하지 않으면 그 이름의 비교는 미정의다. 잘린 대역을 원 대역으로 표시하지 않는다.

공통 지원 광도와 각 파일 전체 지원 광도를 구별한다. 공개 `lbol_edep_toy06_cmfgen.txt`의 Lbol은 참고 규모로만 싣고, 유한 지원 적분과 같다고 두지 않는다. 공개 원천 분해가 없으므로 성분별 CMFGEN 비는 미정의다.

## §7. 주 표·불확실성·미정의 전파

주 산출은 `response.csv` 하나이며 아래 열을 사용하는 긴 형식이다.

| 행 범위·물리량 | A | B | B-A | O 또는 공개 기준 | A/CMFGEN·B/CMFGEN | 편차·유효성 |
|---|---|---|---|---|---|---|
| 셸·전체: 요청·직접 배정·각 실제 소비·미직접배정 | 값 | 값 | 차 | 입력 D | 해당 없음 | 소비 불일치·표현 소실 |
| 셸·구간: J 및 대역별 J 요약 | 값 | 값 | 차 | 저장 J | 해당 없음 | 해 오차·기준 편차 |
| 전체: 정적 바깥 경계 광도 | 값 | 값 | 차 | 해당 없음 | 해당 없음 | 표면·적분 민감도 |
| 전체 지원·공통 지원·다섯 대역: 관측 총광도와 일곱 성분 | 값 | 값 | 차 | O·공개 총광도 | 정의 가능한 총광도만 | 재생 상한·기준 편차 |
| 공통 지원·대역: 공개 대비 거리와 후속 분기 | 값 | 값 | 변화 | 공개값 | 정의 가능한 비 | 분기 사유 |
| 물질 소비·비열적 이력·물리적 중복 해소 | 미정의 | 미정의 | 미정의 | 자료 없음 | 해당 없음 | 이 단의 범위 밖 |

J 원배열과 셸·구간 대응은 보조 배열에 두고, 주 표에는 셸별·등록 대역별 가중 J, 최대 상대 변화와 해당 구간 위치를 싣는다. 이것으로 변화 위치를 식별하며 광도 변화의 셸별 인과 분해라고 부르지 않는다.

재생 `bound`를 성분·대역 적분·차·비에 전파한다. 적분과 차의 반올림, 공개 텍스트 반올림도 포함한다. 양의 전달에서 J의 상대 오차 상한은 J 의존 관측 성분의 상대 오차로 전달할 수 있다. 그 조건이 성립하지 않으면 임의 오차율을 붙이지 않고 해당 응답을 미정의로 둔다.

기준 A-O 편차는 수치 반올림 상한에 섞지 않는다. 별도의 **생산 기준 민감도**로 표시하고 §8의 판단 여유에 사용한다. 선행 재생 상한은 공유 계산 계수 위의 반올림 상한이며, 보간·격자·이식·물리 오차 전체를 인증하지 않는다.

미정의 규칙은 다음과 같다.

- 모든 값은 `DEFINED`, `UNDEFINED`, `NOT_APPLICABLE`과 사유를 갖는다. 미정의를 0으로 저장하지 않는다.
- 입력뿐 아니라 합·차·곱·단위변환·적분·비의 비유한, 비영 값의 0 표현 소실, 직접 가산 소실도 미정의다.
- 미정의 성분을 빼고 합계를 만들지 않는다. 합·차·비는 실제 의존값의 미정의를 상속한다.
- 비의 분모 구간이 0을 포함하면 비는 미정의다. 부호 있는 분자의 구간 나눗셈은 네 끝점 조합을 사용한다.
- 한 공개 대역의 결함은 그 비교와 의존 결론만 막는다. 독립된 다른 대역·가지·성분은 보존한다.
- 한 셸의 원천 결함은 그 셸에서 멈추지 않는다. 같은 주파수의 모든 셸 해와 이를 보간해 소비하는 관측 구간까지 전파한다.
- 부모 신원·세대·공통 기하·단위·격자·선 기록 순서의 결함은 의존 상태 전체를 정지한다.
- 정상 종료 코드나 유한한 출력 한 개로 미정의 상태를 복구하지 않는다.

보조 산출은 부모·코드 신원과 실행 기록, J·원천·잔차 배열, 파생 포획, 봉인 재생 출력, 직접시험·자원 기록뿐이다. 주 표를 다시 읽어 판정을 재계산하는 장치는 만들지 않는다.

## §8. 기대치·GPU 계약에 대한 사전 분기

B의 직접 배정과 실제 소비가 D에 대응하고, A의 직접 소비가 0일 것으로 기대한다. 고정 소광·비음수 원천·안정된 양의 선형 전달이라는 등록 조건 아래에서는 \(J_B-J_A\)와 관측 총광도 응답이 음이 아닐 것으로 기대한다. 수치 상한을 넘어선 음의 응답은 구현·조건 위반을 먼저 조사한다.

매핑 선·BF/FF·안쪽 경계의 직접 탈출 레인은 고정되고, 직접 침적·전자산란·비매핑 산란 레인이 응답할 것으로 기대한다. 고정 레인의 허용차 밖 변화는 가지 구성 또는 재생 연결 결함이다.

응답 크기·파장 분포·공개 대비 접근량은 예측하지 못한다. `O+D`를 예상값·상한·하한으로 쓰지 않는다. 광도 증가가 기존 적외선 초과를 키울 수도 있다.

응답 \(d_y=L_{B,y}-L_{A,y}\), 수치 불확실성 \(u_y\), 기준 민감도 \(b_y=|L_{A,y}-L_{O,y}|\), 공개 광도 \(C_y>0\)를 쓴다. 비교 집합 y는 공통 지원과 다섯 대역이다.

**분리된 응답**은 \(|d_y|-u_y>0.01C_y\)와 \(|d_y|-u_y>5b_y\)를 모두 만족하는 경우다. 1%는 후속 비용 판단의 등록 분해능이며 물리적 정확도 주장이 아니다. 두 경계에 걸리면 분리됐다고 하지 않는다.

공개 접근은 다섯 대역의
\[
E_X=\sum_y|L_{X,y}-C_y|
\]
로 평가한다. 대역이 겹치지 않으므로 공통 지원 행을 이 합에 다시 넣지 않는다. \(E_A-E_B\)의 불확실성 하한이 공통 지원 공개 광도의 1%를 넘어야 등록된 접근으로 인정한다.

| 코드·분기 | 사전 조건 | GPU 계약에 대한 권고 |
|---|---|---|
| 0 `OFFLINE_REWORK` | 실제 소비 불일치, 고정 레인 변화, 허용차 밖 음의 응답 등 오프라인 결함 확인 | 재개 권고 없음. 해당 결함만 기존 상한 안에서 처분 |
| 1 `RESUME_WITH_REVISION` | 공통 기준·소비·해가 유효하고, 비교 집합 전부 정의되며, 분리된 응답과 등록된 공개 접근을 함께 확인 | 생산 배선·같은 상태 소비 장부를 검증하도록 보류 계약 개정 후 재개 권고 |
| 2 `DO_NOT_RESUME` | 전 대역 응답 상한이 각각 1% 이내이거나, 분리된 응답은 있으나 공개 거리 감소가 없음을 불확실성까지 포함해 확인 | 현재 전량 Planck 주입을 결손 대응으로 시험하는 GPU 계약의 폐기 또는 표적 변경 권고 |
| 3 `INDETERMINATE` | 기준 편차 우세, 조건 경계, 수렴 실패, 필수 비교 미정의, 상한 종료 등 | 보류 유지. 오프라인에서 못 가른 항목을 명시하며 GPU로 자동 이관하지 않음 |

거리가 줄지만 등록 접근량에는 못 미치는 경우도 3이다. 정의된 일부 대역의 큰 응답만으로 1이나 2를 결정하지 않는다. 결함 확인인 0이 우선하며 나머지 분기는 서로 배타적이다.

분기 1도 생산 승격이나 직접 원천의 물리적 정당성을 뜻하지 않는다. 후속 GPU 계약은 기존 방출과 침적의 이중 계산 문제를 명시적으로 유지해야 한다.

## §9. 게이트와 새 코드의 음성 대조

| 게이트 | 통과 기준 | 막는 실재 실패 |
|---|---|---|
| G-PLAN | 기존 사전등록·참조·정본·발주 검사와 등록 변경 범위 일치 | 보류 GPU 계약이나 미등록 소스 변경을 실행 |
| G-INPUT | 부모·단위·세대·지원·payload·파생 변경 범위 일치 | 다른 상태나 잘못 결합한 셸을 같은 포획으로 비교 |
| G-OPERATOR | §4 수렴·조건 확인과 §5 기준 대조 | 작은 1회 잔차나 잘못된 이식 해를 생산 대응으로 인정 |
| G-CONSUME | 전달·관측의 실제 가산과 고정 성분 보존 | 명목 직접 레인만 기록하고 실제 원천에 넣지 않음 |
| G-READ | 표면·적분·성분 폐합·미정의 전파 충족 | 중점 J를 광도로 사용하거나 결함 대역을 합계에서 누락 |
| G-RUN | 등록 자원·명령·종료·입력 보존 확인 | 실패한 하위 명령 뒤 rc=0 표식으로 완료 선언 |

게이트마다 새 프로그램을 만들지 않는다. 공통 판독·산술·신원 검사를 재사용한다. 직접시험은 새 코드의 음성 대조로 한정하며, 작은 정상 대조와 같은 실제 판독·실행 경로를 호출한다.

1. **입력·연결:** 중복·누락 셸, 잘못된 단위·세대·격자, 허용 외 배열 변경, manifest와 payload 불일치를 각각 주입한다. 최초 판독에서 해당 의존 범위가 멈추는지 확인한다.
2. **원천·소비:** 음의 침적, 영 정규화, `eta_dep`만 변경, `eta_c`만 변경, 전달 가산 누락, BF/FF 변경을 각각 실제 구성·봉인·작은 재생 경로로 통과시켜 거부를 시연한다.
3. **산란·해:** 매핑 sentinel 오해, 비매핑 산란 누락, 부적격 행렬, 잔차 기준 초과, 보정 상한 종료를 실제 해법에서 시연한다. 판정 문자열을 직접 바꾸는 시험은 인정하지 않는다.
4. **경계·판독:** 중점 강도의 표면 대입, 잘못된 환대 가중·파장 단위·지원 밖 보간을 알려진 작은 입력으로 시연한다. 전체 생산 관측기를 재구현해 시험하지 않는다.
5. **산술·전파:** 비유한 입력, 유한 피연산자의 오버플로, 비영 적분의 0 표현 소실, 직접 가산 소실, 0을 포함한 비의 분모, 한 성분·셸·대역 결함과 공통 결함을 실제 판독에서 시연한다.
6. **집행:** 하위 실행의 비영 종료와 수치 미정의를 러너가 소비해 후속 판정을 중단하는지 확인한다. 파일 쓰기 실패를 모사하지 않는다.

각 시험은 주입이 실제 피연산자에 도달했고 등록 실패를 검출했음을 남긴다. 기존 전체 시험 묶음 복제·저장 산출 재검·별도 결정표 시험기는 금지한다.

## §10. 변경집합·실행·비용

### 변경집합 - DET-DEP-OFFLINE-RESPONSE (표)

| 파일 | 허용 변경 |
|---|---|
| `/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/offline/tools/det_dep_offline.py` | 기존 판독·형식해 재사용, 고정 계수 해법, 원천·표면·주 표·음성 대조 |
| `/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/offline/tools/det_dep_offline_capture.c` | 신규 대상 확인 후 기존 봉인·재생 함수 호출 |
| `/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/offline/tools/run_det_dep_offline.slurm` | CPU 빌드·검사·실행·자원 제한과 종료 전달 |

### 변경집합 끝

기존 C 소스는 수정 없이 링크한다. `build_stage1.sh`의 CPU 소스 목록·컴파일 처방을 재사용하되 출력은 F 아래로 제한한다. 기존 빌드 스크립트를 원래 출력 경로로 실행하지 않는다. 새 함수 이름은 `det_dep_offline_` 접두를 사용한다.

**GPU 0, 생산 변경 0.** 모든 빌드·해시 계산·기계 검사·직접시험·수치 계산·재생은 a10 CPU Slurm에서 한다. 로그인 노드는 파일 조회만 허용한다.

작업은 `LUMINA_` 접두, `--partition=a10 --ntasks=1 --cpus-per-task=4 --mem=16G --gres=gpu:0 --exclude=syn102,syn03 --no-requeue`를 사용한다. 작업당 벽시계 최대 60분, 전체 할당 벽시계 합 **90분**, 전체 **6 core·h**, 신규 저장 **8 GiB** 중 먼저 닿는 상한을 적용한다. 실패·수정·감사 재현도 포함한다.

BLAS 스레드는 1, 관측 vector 재생 스레드는 최대 4다. O·A·B 재생은 순차 실행한다. 상태·원천 분율 탐색이나 자동 재제출은 없다. 상한 도달은 미완료 범위를 기록하며 수용 기준을 낮추지 않는다.

`module purge` 뒤 CPU 빌드의 `gnu13/13.2.0`과 사용할 Python·NumPy 판본을 명시한다. 기존 스크립트의 느슨한 종료 전달은 승계하지 않는다. 모든 하위 명령의 실제 종료와 수치 상태를 확인한다.

실행 위치는 F의 신규 `run1`이다. 파생 포획 두 개와 원 재생 출력은 보존하되 원 O payload를 세 번째로 복제하지 않는다. 크기는 명시 파일 목록으로 합산하고, 기존 실행 디렉터리는 덮어쓰지 않는다.

등록 실행 인터페이스는 다음과 같다. 모두 해당 Slurm 작업 안에서 호출한다.
```bash
python3 -B tools/det_dep_offline.py selftest --out tests/run1
python3 -B tools/det_dep_offline.py run --out run1
```
운전자는 정확한 컴파일·링크·실행 명령, 입력 해시, 실행 파일 해시, 작업 번호·종료·할당 비용을 기록한다. 재현은 같은 입력과 명령의 새 출력 경로를 사용하며 잔여 상한 안에서만 수행한다.

## §11. 판정·종료감사·주장 상한

완료 판정은 주 표가 과학 질문과 §8 분기를 실제로 결정했는지 평가한다. 기준 대조만 통과하거나 원천만 구성한 상태를 목표 달성으로 처리하지 않는다. 미완료면 필요한 응답과 실패 범위를 명시한다.

종료감사의 우선 대상은 주 표의 수치·원자료 연결, 기준 편차와 B-A의 분리, 실제 소비, 산란 처리, 파생 포획 신원, 공개 지원, 미정의 전파, GPU 권고와 주장 상한이다. 기안자의 조회와 운전자의 계산·재현을 구별한다.

생산 승격·물리적 정답·수리·새 T 계산·자기일관 효과·철족 일반화를 선취하지 않는다. 연결 가지의 광도 증가를 결손 해소나 정당한 물리로 단정하지 않는다. 어느 분기도 GPU 제출이나 생산 변경을 자동 승인하지 않는다.

기안자는 공유 맥락 전체, 현재 프로젝트 신원, 지정 사용자 결정·보류 계약·diag10 기록, O 포획 metadata·manifest·입력, 봉인 소스와 CPU 빌드 처방, 관련 관측 판정과 공개 자료 표제를 조회했다. 파일 변경·계산·빌드·시험·제출은 하지 않았다. 아래 기계 선언의 검사 통과도 아직 주장하지 않는다.

집행 기록은 이 절에 둔다. 운전자는 기계 검사, 코드·실행 신원, 음성 대조, 기준 대조, 계산·재생, 비용, 주 표 판정, astra 종료감사와 이탈 처분을 기록한다.

### 집행 기록 (운전자 추가 — 정본 장부, 개정16 ①; 위 기안 기록은 그대로 둔다)

증거 접두 `B=/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012`, `F=B/offline`. 작업 번호·경과는 `B/JOBS.txt`·`F/evidence/sacct_offline_jobs.txt`, 게이트 집행은 `F/evidence/GATE_RECORD.json`.

| 항목 | 담당(실제) | 상태·증거 |
|---|---|---|
| user 결정 | user | 「침적 소비·직접 원천 대조 시험」(`B/evidence/USER_DECISION_2026-10-12.txt`) → 「오프라인 선행(GPU 0) → 필요 시 GPU」(`B/evidence/USER_DECISION_2026-10-12_offline_first.txt`) |
| 기안 | astra(gpt-6-astra) | 7ad42af; 기안 게이트 prereg·referents·canonical rc 0(a10 419982) |
| 사전감사 | — | 생략(25-ab: GPU 0·생산 변경 0) |
| 구현 | 운전자(Opus 5.5) · 코드 감사 없음(25-z) | `F/tools/det_dep_offline.py`(0172af3f…), `det_dep_offline_capture.c`(dab17c03…), `run_det_dep_offline.slurm`(952d86dc…); 봉인 소스 무수정 링크, diag10 `segment`·`formal_once` 재사용 |
| 구현 게이트 | 운전자 | a10 420013: prereg·referents·canonical·위생 rc 0(`B/evidence/gates_off1.summary`) |
| 빌드·직접시험 | 운전자 | a10 420014: 빌드 rc 0(봉인 함수용 `-DLUMINA_DET_OBS_SELFTEST`·재생용 생산 정의, `F/build/`), 직접시험 34/36 — 연산자 결함 2종이 기준 대조 2 로 거부됐으나 시험 기대가 「check3」 만으로 좁아 실패 표기 → 기대 정정; a10 420017 `F/tests/run2` **36/36**(기존 빌드 신원 확인 재사용) |
| 계산 run1 | 운전자 | a10 420018 `F/run1/`(2855 s, rc 0, 처리 완료, 입력 불변): 기준 대조 1 바이트 동일·2 3.9126e-4·3 최대 8.80e-3/가중 3.84e-5·4 ≤ 4.77e-5; A·B 해 1234/1234 수렴·확인; 결함 0; 소비 — 명목·직접 레인 50/50 일치, 전달·eta_c 가산은 셸 0 만 정의(셸 1–49 직접 가산 소실 미정의); 비교 집합 전부 정의; 자동 분기 code 1 |
| 판독 탐침 | 운전자 | a10 420038(`F/probe/probe_loss_cells.py`, `F/probe/loss_cells_420038.json`): 소실 248 구간·8725 셀, 74.47–275.64 Å, η_dep/η_c ≤ 1.04e-16, 소실 합 2.6e21 erg/s |
| 판정 | 운전자(구현·판정 겸임) | `docs/VERDICT_DET_DEP_OFFLINE_RESPONSE_2026-10-12.md` — **분기 3 INDETERMINATE**(§8 분기 1 의 「소비 유효」 미성립; 자동 code 1 은 판정 장치의 전제 검사 누락) |
| 판정 게이트·종료감사 발주 게이트 | 운전자 | a10 420039: roster·prereg·referents·canonical·위생·발주 프리플라이트 rc 0(`B/evidence/gates_off2.summary`) |
| 종료감사 1회차 | astra(자기보고 「gpt-6-astra(발주 지정명; 런타임 ID 독립 확인 못 함)」, 계약 기안자 이해관계) | 판정 10ae071 대상: **폐합 가 — 분기 3 한정 · «폐합 차단» 0 · «기재로 족함» 4**(자동 code 1 과 판정자 정정의 구별, A 경계 광도 저장 상한 1.664e59 누락, 「1e-10」 서술·J 최대값 한정·`np.float64` 범위, 비용 집계 시점); 실행 재현 BLOCKED(`B/astra_closeout_off.md`) |
| 1회차 반영 | 운전자 | 전건 반영(판정문 문면 OLD/NEW 4쌍, §8 신설) — 분기·주 표·게이트 판정치 불변 |
| **폐합** | 운전자 | 2026-10-12, **분기 3 INDETERMINATE**, 정본 run1; 보류 GPU 계약은 보류 유지(재개·개정·폐기는 user 결정); 폐합 게이트 a10 420043: roster·prereg·referents·canonical·위생 rc 0(`B/evidence/gates_off3.summary`); 같은 작업의 발주 프리플라이트 rc 1 은 커밋 전 이 장부 편집의 작업트리 차이(contract-worktree-drift)이며 폐합 뒤 발주가 없어 해당 없음 |
| 비용(최종) | 운전자 | GPU 0; a10 7 작업 할당 벽시계 합 3036 s / 5400 s, 11781 core·s / 21600, 작업당 최대 2855 s / 3600; 신규 저장 1.35 GB / 8 GiB(작업별 값은 `F/evidence/sacct_offline_jobs.txt`) |

**이탈(기재):** ① 자동 분기 함수가 §8 분기 1 의 전제 「소비 유효」를 검사하지 않음 — 판정자가 §8 을 주 표에 적용(재실행은 남은 상한 밖). ② 첫 직접시험 34/36(시험 기대 과협소) → 정정·새 경로 재현 36/36. ③ 첫 직접시험 뒤 러너 수정(빌드 재사용·출력 경로 인자). ④ 주 표 소비 행 `baseline_dev` 표기 결함(`np.float64(…)`). ⑤ 판독 탐침 1건(판정 재계산 아님).

## §12. 기계 선언

참조는 실재 파일만 지정한다. 분기 예시는 기존 사전등록 문면 검사의 입력이며 별도 결정표 재계산 장치를 요구하지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 - DET-DEP-OFFLINE-RESPONSE (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/offline/tools/(?:det_dep_offline\\.py|det_dep_offline_capture\\.c|run_det_dep_offline\\.slurm)",
    "symbol": "det_dep_offline_",
    "roots": [
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/offline/tools/det_dep_offline.py",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/offline/tools/det_dep_offline_capture.c",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/offline/tools/run_det_dep_offline.slurm"
    ],
    "expected_extra": [
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/offline/tools/det_dep_offline.py",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/offline/tools/det_dep_offline_capture.c",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/offline/tools/run_det_dep_offline.slurm"
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
      {"name":"RESUME_WITH_REVISION","predicate":"valid and c1 == 1"},
      {"name":"DO_NOT_RESUME","predicate":"valid and c2 == 1"},
      {"name":"INDETERMINATE","predicate":"not valid or (c0 == 0 and c1 == 0 and c2 == 0)"}
    ],
    "adversarial_fixtures": [
      {"name":"구성결함","mix":[[0,1]]},
      {"name":"개정후재개권고","mix":[[1,1]]},
      {"name":"재개권고없음","mix":[[2,1]]},
      {"name":"미정의","mix":[[3,1]]},
      {"name":"잘못된값","mix":[[9,1]]},
      {"name":"누락","mix":[]},
      {"name":"중복","mix":[[0,1],[1,1]]}
    ]
  },
  "references": [
    {"path":"docs/PREREG_DET_DEP_SOURCE_CONSUMPTION_2026-10-12.md"},
    {"path":"docs/PREREG_DET_LINE_NET_PROVENANCE_2026-10-11.md"},
    {"path":"docs/VERDICT_DET_OBS_CAPTURE_2026-10-07.md"},
    {"path":"docs/VERDICT_DET_OBS_WRITER_2026-10-02.md"},
    {"path":"docs/VERDICT_DET_OBS_LINEDECOMP_2026-10-07.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/evidence/USER_DECISION_2026-10-12.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/evidence/USER_DECISION_2026-10-12_offline_first.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag10_energybudget_20261012/DIAGNOSIS.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag10_energybudget_20261012/LOOP_CLOSED.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag10_energybudget_20261012/tools/r1_transfer_budget.py","flags_existing":["formal_once","segment"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag10_energybudget_20261012/out/r1_budget_419905/result.json"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag10_energybudget_20261012/out/r2_material_419922/summary.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/work/det_obs_capture/manifest.json"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/work/det_obs_capture/meta.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/work/lumina_spectrum_det_obs.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/model/deposition_cmfgen.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/resolved_lumina.exports"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/run_build_identity.json"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c","flags_existing":["det_obs_capture_seal_fixture","det_obs_capture_verify_committed","det_obs_capture_replay","formal_solve_bin"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/tools/det_obs_capture_replay.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/impl/build_stage1.sh"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/cuda_build1/LINEAGE_NOTE.txt"},
    {"path":"data/standart_data1/toy06/spectra_toy06_cmfgen.txt"},
    {"path":"data/standart_data1/toy06/lbol_edep_toy06_cmfgen.txt"}
  ]
}
```