# 사전등록 — DET-SIIII-1206-MECH: 초기 복사장 씨앗의 유지와 1206.50 Å 자기 되먹임 기전 시험

2026-10-07 · 초안 · read-only  
운전자 파일화 대상: `docs/PREREG_DET_SIIII_1206_MECH_2026-10-07.md` — 신설  
상태: **발주 전 실측과 T2 소스를 읽은 뒤 작성한 기전 시험 계약. 독립 Q-GOAL·Q-LEAN 사전감사, 구현, 기계 게이트, 직접시험 및 등록 분석은 미집행.**

## §0. Q-GOAL·Q-LEAN 자기답과 과학 질문

### Q-GOAL — 최종목적에 직접 기여하는가?

DET-SIIII-UPSTREAM은 준위 4의 세대 7 밀도 차이를 복사 입력 항 지배로 판정했지만, E′의 1206.50 Å J̄가 큰 이유를 고정 상태의 율 분해로 구별하지 못했다. J̄의 대부분이 해당 선의 상위 인구에서 나오는 국소 방출이어서 인구와 복사 입력이 서로를 설명하는 순환이 남았다.

이번 목표는 **저장된 반복 이력을 이용해 그 순환이 초기 씨앗을 보존하는 느린 되먹임인지 시험하고, 관측된 씨앗 유지와 그 기전의 설명력을 구별하여 판정하는 것**이다. 다음 질문에 답한다.

1. 1206.50 Å의 call 1 J̄는 희석되지 않은 \(B_\nu(T_{\rm inner})\)의 생산 프로파일 평균과 얼마나 가까운가?
2. E′는 call 1–7 동안 그 값을 유지하는가? 단순 off 반복 사상에서 유도한 변화량이 실제 변화량을 설명하는가?
3. 세대 7의 \(n_4/n_0\)는 초기 J̄가 정하는 복사 여기비에 가까운가? \(T_{\rm inner}\)와 국소 \(T_e\)의 Boltzmann 비를 구별할 수 있는가?
4. R′에서는 같은 초기 씨앗에서 출발한 뒤 씨앗 유지가 해제되는가?
5. 같은 **씨앗 유사성·반복 중 유지**가 기록된 S II/III·Ca III 등의 두꺼운 선에서도 나타나는가? 어느 범위까지 성립하고 어디에서 반증되는가?

이 결과는 공개 CMFGEN 잣대가 있는 UV2500 가지와 황 이온화 전선에서 다음 수리 검증의 대상을 고르는 데 쓰인다. 생략하면 큰 J̄를 원인의 이름으로 되풀이하거나, 두꺼운 선 일부의 정체를 원자외 복사장 전체의 정체로 확대하게 된다.

이번 단의 **스펙트럼 잔차 감소는 0**이다. UV2500 E′N의 +0.61 dex는 선행 비교값이며 이번 단에서 다시 측정하거나 개선하지 않는다.

**기안자 자기답: PASS.**

### Q-LEAN — 불필요한 계측·게이트를 추가하는가?

주 시험은 Si III 1206.50 Å와 W5의 다섯 셸로 한정한다. 황 전선 연결에는 같은 W5에서 씨앗·정체 통계만 확장한다.

- 1206.50 Å: 반복 이력, 생산 계수로 유도한 단순 반복식, 최종 캡처의 준위비를 연결한다.
- 다른 두꺼운 선: 초기 씨앗 유사성, 반복 중 유지, 국소 방출 몫과 β 분포를 센서스로 남긴다.
- 다른 선의 준위 인구 복원·율 예산·충돌 계산·연립해는 추가하지 않는다.
- 중간 두께 선은 같은 판독 중 대조 통계만 계산한다.
- 세대 8 인구와 반복별 이온 인구는 읽지 않는다. 이번 준위비와 같은 세대의 대체 자료가 아니며 등록 질문에 필요하지 않다.
- 생산 계측, 새 캡처, 생산 solver 수정, 전달 replay, GPU 판정런은 추가하지 않는다.
- 새 판독기와 러너만 만든다. 시험·입력 연결·산술 검증은 두 내부 게이트에 묶고 별도 계측 체계를 만들지 않는다.

소비자는 운전자와 사용자다. 1206.50 Å 결과는 수리 검증으로 진행할지 결정하는 데, 센서스는 그 대상과 주장 범위를 제한하는 데 사용한다.

**기안자 자기답: APPROPRIATE.**

자기답은 독립 사전감사를 대신하지 않는다. Fable fresh가 Q-GOAL, 이어서 Q-LEAN에 독립으로 답한다. 호출 실패 시 사용자 지정 대체 모델 gpt-6.1-sol을 사용하고 실패 증거와 실제 모델 ID를 남긴다.

### 승인 근거와 사전 노출

승인 근거는 다음 사용자 결정문이다.

`/gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007/evidence/USER_DECISION_2026-10-07.txt`

DET-SIIII-UPSTREAM §10.2 코드 1 경로의 1206.50 Å J̄·NRB 괄호 기전 시험에 대한 「승인합니다」를 승계한다. 바닥 인구 항도 기여하므로 단일 원인 경로로 강제하지 않는다.

기안자는 다음 실측 기록을 직접 읽었다.

`/gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007/evidence/w5_1206_history_2026-10-07.txt`

따라서 본 계약은 **결과를 전혀 보지 않은 맹검 사전등록이 아니다.** 알려진 1206.50 Å 이력과 기존 센서스 요약을 공개한 상태에서, 재현 가능한 판독·예측식·반증 조건을 등록한다. 기존 실측의 재현을 독립 자료에 대한 예측 성공으로 표시하지 않는다.

### 주장 상한

판정 대상에는 다음 사실 명제를 포함한다.

> 등록된 E′의 λ<2000 Å 두꺼운 선 가운데, 어떤 선·셸의 J̄가 call 1–7 동안 초기 씨앗에 등록된 허용 범위 안에서 머물렀는가?

범위와 반례를 함께 판정한다. 다음은 판정하지 않는다.

- E′ 또는 R′ 중 어느 형태가 물리적으로 옳은지.
- NRB의 생산 승격, 생산 해 수렴성 또는 반복 연장의 효과.
- UV2500 잔차 감소나 황 과이온화의 인과 기여율.
- 선 J̄의 정체가 연속체 광이온화장 전체의 정체라는 주장.
- 초기 씨앗과 흑체의 수치적 유사성만으로 그 광자의 발생·전달 경로를 유일하게 특정하는 주장.

## §1. 경로·기준선·승계 사실

### 1.1 경로

```text
P  = /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
C  = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006
T2 = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source
E  = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Eprime/run1
R  = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1
U  = /gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007
D  = /gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007

선행 정본 = U/results/t4
새 러너 = D/tools/run_1206mech.slurm
새 산출 = D/results/<실행 식별자>
새 집행 증거 = D/evidence/<집행 식별자>
```

### 1.2 승계 기록과 도구 신원

- 선행 폐합: `12b58ab`.
- 직전 계약: `docs/PREREG_DET_SIIII_UPSTREAM_2026-10-07.md`, 개정 1.
- 직전 판정: `docs/VERDICT_DET_SIIII_UPSTREAM_2026-10-07.md`.
- 선행 종료감사: U의 astra_closeout.md.
- 선행 신원 장부: U/results/t4/identity.sha256.
- 생산식의 권위: T2. 현재 작업 트리의 생산 소스로 대체하지 않는다.

기존 판독기를 import할 경우 다음 선행 신원과 실제 import 대상을 대조하고, 사용한 함수와 경로를 manifest에 남긴다.

```text
scripts/det_siiii_rates_readout.py
2fb17ef88356cbb5841463c2f2de6c5674d140e332be0474bb8c8b5330229dfd

scripts/det_siiii_pop_readout.py
63cbea718eec73a84b21e4cf907e89934b5cbf2011026834cd50cc8e686976be
```

이는 선행 장부에 기록된 신원이며 기안자가 이번에 새로 해시 검증한 결과는 아니다. 기존 도구는 수정하지 않는다. 기존 analyze의 고정 셸·call 7·세대 8 접근을 새 경로로 그대로 들여오지 않는다.

### 1.3 재판정하지 않는 승계 사실

1. 준위 4의 세대 7 밀도 차이는 셸 19·24·30에서 복사 입력 항 지배이며 NRB 형태 항은 반대 부호다.
2. 바닥 상대 인구 항은 특히 셸 19에서 무시할 수 없다. 여기비 시험으로 그 항의 원인까지 설명했다고 쓰지 않는다.
3. Si III 준위 4의 직접 과정은 bb 복사·충돌이며 나머지 직접 과정의 `STRUCTURAL_ZERO`를 승계한다. 다른 이온으로 일반화하지 않는다.
4. 세대 7 캡처의 생산 해 재현 불가는 §5.3에 승계한다.
5. 선행 판정과 감사의 저자·처분은 보존한다.

황 전선 기록은 정정까지 함께 승계한다.

- 09-25 1단계의 69–82%는 **명시 들뜬 준위 경로가 총 S II 광이온화율에서 차지하는 몫**이다. 명시 들뜬 준위와 λ<2000 Å의 교집합은 조사한 경우 총율의 54–74%다.
- 2단계 기록은 반복마다 두께 집합을 다시 나눈 평균과 일부 인과 서술을 철회했다. 일부 전이의 지속성을 전체 장의 원천 확증으로 높이지 않는다.

근거는 `docs/READOUT_CAUSE1_SII_IONIZATION_CHANNEL_2026-09-25.md` §5–§6 및 `docs/READOUT_CAUSE2_FUV_LINE_LAMBDA_STALL_2026-09-25.md` §6–§7이다.

### 1.4 입력 권위

| 입력 | 이번 사용 |
|---|---|
| E·R의 work/w5_lines.csv | call 1–7의 선·셸별 J̄, 오차, operator, 세대 및 call 2–7의 J_c·β·L_l·τ_eff |
| E·R의 work/det_obs_capture/lines.bin | 1206.50 Å의 세대 7 τ·S_used·소비 상태 |
| 같은 캡처의 manifest.json·meta.txt·Te.f64 | 캡처 권위·세대·T_inner·inner_bb_scale·최종 국소 T_e |
| E·R의 input/model/line_list.csv·levels.csv | 행 ID, 끝점, 주파수·파장, A·B·f, 에너지·통계중량 |
| E·R의 input/deck.sha256·resolved_lumina.exports·run_build_identity.json | 입력·설정·생산 신원 |
| T2의 관련 소스·헤더 | 초기 프로파일, off/armed 계수, Sobolev 생산식·상수 |

W5의 call 1은 Sobolev 전용 칸이 `NOT_APPLICABLE`인 정상 초기 레코드다. 이 칸들을 0으로 채우거나 call 2의 β·τ로 소급 대입하지 않는다.

대형 입력의 기존 봉인 신원은 출처를 명시해 승계한다. 이번에 읽은 선택 레코드와 새 도구·작은 의존 파일의 신원은 별도로 기록한다. 전체 캡처 복제나 전체 CSV 변환을 하지 않는다.

### 1.5 직접 읽은 T2 경로

| 근거 | 확인 내용 |
|---|---|
| [lumina_cmfgen.c, cm_planck](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c:675) | 생산 Planck 함수와 inner_bb_scale 정의 |
| [lumina_cmfgen.c, 미세 격자 구성](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c:8256) | 로그 주파수 간격·프로파일 지지구간·격자 기원 |
| [lumina_cmfgen.c, 초기 warm start](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c:9048) | warm start는 국소 \(B_\nu(T_e)\)이며 이후 전달해를 계산함 |
| [lumina_cmfgen.c, 미세 전달해 호출](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c:9263) | T_inner와 inner_bb_scale을 전달해 경계 입력으로 넘김 |
| [lumina_cmfgen.c, 선 J̄ 생산](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c:9772) | INIT는 미세 전달해의 Gaussian 평균, 후속은 Sobolev 생산식 |
| [lumina_cmfgen.c, 초기·후속 operator 선택](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c:11770) | INIT와 후속 operator의 구별·복사 커밋 순서 |
| [lumina_plasma.c, bb_nrb_arm](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c:18684) | R′도 INIT에서는 armed가 아님 |
| [lumina_plasma.c, 생산 bb 율](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c:19174) | off의 B·J̄ 및 A, armed의 line_net_bracket_rates 적용 |
| [line_net_rate.c, 물질·복사 생산](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_net_rate.c:220) | τ와 상위 인구에서 L_l·β·J̄ 생성 |
| [line_net_rate.c, 계수 정규화 비](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_net_rate.c:350) | Einstein 흡수 계수와 Sobolev 계수의 비를 명시적으로 계산 |
| [line_jbar.c, Gaussian 평균](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_jbar.c:77) | 이산 가중치·정규화·지지구간 |

따라서 **INIT라는 이름 자체가 모든 선의 call 1 값을 \(B_\nu(T_{\rm inner})\)로 직접 대입했다는 증거는 아니다.** 이번 씨앗 시험은 저장된 초기 출력과 흑체 비교값의 일치를 측정한다.

## §2. 표적·셸·고정 센서스

### 2.1 주 표적

| 항목 | 등록값 |
|---|---|
| 원소·이온 | Z=14, ion=2, Si III |
| 전이 | 준위 0→4, 3s² ¹S₀→3s3p ¹P₁ |
| 선 행 | 헤더를 제외한 0-based 데이터 행 600693 |
| 파장 | 약 1206.499509 Å |
| 주요 셸 | 19 |
| 공간 비교 셸 | 10·15·24·30 |
| 복사 이력 | E′·R′, call 1–7 |
| 인구 비교 | E′·R′, 세대 7 캡처 |

선택은 행·원소·이온·양 끝 준위·주파수를 함께 대조한다. CSV의 별도 line_id 열로 행 번호를 대체하지 않는다. 1206.56 Å의 4→15 전이와 구별한다.

다섯 셸 모두 필수 비교표에 남긴다. 주요 판정은 셸 19이며 다른 셸의 성공으로 대체하지 않는다. “다섯 셸에서 성립”은 다섯 셸 각각의 결과가 성립할 때만 쓴다.

### 2.2 확장 센서스

센서스의 모집단은 W5에 기록된 Si II/III·S II/III·Ca II/III의 다섯 셸이다. 미기록 이온·셸·전이는 모집단에 포함하지 않는다.

두께 집합은 **E′ call 2의 저장 τ_eff**로 고정한다.

\[
\mathcal H=\{(\ell,s):\tau^{E'}_{\ell s,2}>100\},
\qquad
\mathcal M=\{(\ell,s):1<\tau^{E'}_{\ell s,2}\le100\}.
\]

- \(\mathcal H\): 두꺼운 선 센서스.
- \(\mathcal M\): 중간 두께 대조 통계.
- 양쪽 모두 같은 선·셸 키로 E′·R′의 전체 call 이력을 연결한다.
- 이후 얇아진 선도 집합에서 제거하지 않는다. 두께 변화는 결과다.
- R′를 자기 최종 τ로 다시 선별하지 않는다.
- call 2 결손·지지구간 밖·비정상 상태는 별도 수량과 사유를 남긴다.

각 집합을 λ<2000 Å와 λ≥2000 Å로 나누고, 이온·셸별 결과도 남긴다. 빈 집합의 비율은 0이 아닌 미정의로 표시한다.

β별 비교는 E′ call 2 기준의 고정 구간을 사용한다.

\[
(0,10^{-6}],\quad(10^{-6},10^{-4}],\quad
(10^{-4},10^{-2}],\quad(10^{-2},1].
\]

실측 요약의 최종 τ 기준 센서스와 이번 call 2 기준 센서스는 분모가 다르다. 기존 1581건이나 동결 비율을 재현 목표로 강제하지 않는다.

### 2.3 확장 범위의 한계

확장 센서스에는 §3의 씨앗 비교, §4.1의 유지 지표, 국소 방출 몫 및 β별 분포만 적용한다.

다른 전이에도 동일한 두 준위 고립 사상이 성립한다고 가정하지 않는다. 다른 이온의 여기온도·준위 예산·광이온화율·스펙트럼 가중치는 계산하지 않는다.

## §3. 씨앗 동일성 시험

### 3.1 비교값

생산 상수와 실제 메타데이터를 사용한다.

\[
B_\nu(T)=\frac{2h\nu^3}{c^2}\frac{1}{e^{h\nu/kT}-1}.
\]

각 선에 대해 다음을 구별한다.

1. 선 중심의 \(B_{\nu_\ell}(T_{\rm inner})\).
2. 생산 미세 격자·Gaussian 규칙으로 계산한 흑체 평균 \(\mathcal B_\ell\).
3. 실제 W5 call 1의 \(\bar J_{\ell s,1}\).

\[
\mathcal B_\ell
=b_{\rm inner}
\frac{\sum_{i\in Q_\ell}w_iB_{\nu_i}(T_{\rm inner})}
{\sum_{i\in Q_\ell}w_i},
\]

\[
w_i=e^{-x_i^2}\Delta\nu_i,\qquad
x_i=\frac{\nu_i-\nu_\ell}{\nu_\ell v_D/c},\qquad |x_i|\le4.
\]

격자 간격·기원·\(v_D\)·ppd·파장 경계는 봉인 설정과 T2를 따른다. \(\Delta\nu_i=\nu_i\Delta\ln\nu\) 및 생산의 경계 포함 규칙을 유지한다. 필요한 프로파일 주변 격자만 계산하고 전체 전달 격자나 전달해를 재구축하지 않는다.

\(\mathcal B_\ell\)는 **흑체 가설의 비교값**이다. 저장되지 않은 초기 미세 전달해를 복원했다고 표시하지 않는다. 선 중심값과 프로파일 평균의 차이를 실제 초기 전달해와 흑체의 차이와 혼동하지 않는다.

### 3.2 등록 지표

\[
d_{\rm seed}^{X}=
\left|\frac{\bar J^{X}_1}{\mathcal B_\ell}-1\right|,
\qquad
d_{\rm ER,1}=
\left|\frac{\bar J^{E'}_1}{\bar J^{R'}_1}-1\right|.
\]

공간 균일성은 같은 선의 다섯 셸에 대해

\[
d_{\rm shell,1}^{X}
=\frac{\max_s\bar J^{X}_{s,1}}
{\min_s\bar J^{X}_{s,1}}-1
\]

로 보고한다. 모든 분모는 양수와 분리되어야 한다.

**씨앗 유사성의 과학 기준은 \(10^{-3}\)**이다. E′·R′ 각각의 \(d_{\rm seed}\) 및 \(d_{\rm ER,1}\)이 이 기준 안에 있으면 해당 셸의 흑체 씨앗 가설이 생존한다. 공간 균일성은 별도 결과이며 주요 셸 판정을 대신하지 않는다.

이 기준은 비트 동일성이나 반올림 허용폭이 아니다. 수치 구간을 계산한 뒤 적용하는, 사전에 정한 0.1% 근접성 기준이다.

### 3.3 반증과 해석

- \(d_{\rm seed}\)가 기준 밖으로 분리되면 해당 선·셸의 **희석 없는 흑체 씨앗 가설은 반증**된다.
- 초기값이 흑체와 달라도 이후 초기값 유지 가설은 독립적으로 시험한다.
- 다섯 셸의 값이 비슷하더라도 흑체 비교가 실패하면 흑체 씨앗으로 부르지 않는다.
- 수치 구간이 기준을 가로지르면 미분리다.
- 일치해도 초기 전달해의 유일한 생성 원인이나 물리적 희석의 부재를 증명한 것으로 해석하지 않는다.

## §4. 반복 이력과 off/armed 기전 예측

### 4.1 씨앗 유지의 관측 정의

각 런·선·셸에서

\[
D_{\rm seed}
=\max_{k=2,\ldots,7}
\left|\frac{\bar J_k}{\bar J_1}-1\right|,
\qquad
\delta_k=\frac{\bar J_{k+1}}{\bar J_k}-1
\]

을 구한다. 끝점 비 \(\bar J_7/\bar J_1\)도 함께 보고한다.

**씨앗 유지:** \(D_{\rm seed}\le10^{-3}\).

끝점이 우연히 돌아온 경우를 동결로 세지 않는다. 최대 이탈의 call과 구간을 남긴다. 유의미한 중간 이탈 한 건이면 전체 이력의 유지 가설을 반증할 수 있다.

call 2–7에는

\[
f_{{\rm local},k}=\frac{L_k}{\bar J_k}
\]

를 보고한다. 1206.50 Å의 국소 방출 지배 예측은 모든 해당 call에서 \(f_{\rm local}\ge0.99\)다.

W5 원장식

\[
\bar J_k=\operatorname{fl}\{\operatorname{fl}(\beta_kJ_{c,k})+L_k\}
\]

의 일치는 입력 연결·산술 확인이다. 이 식의 재계산 성공만으로 기전 예측 성공을 선언하지 않는다.

### 4.2 생산 계수를 보존한 단순 off 사상

다음 유도는 **한 전이가 두 준위의 비를 정하고, 물질 갱신이 그 전이의 통계평형 비를 따르는 조건부 모형**이다. 실제 다준위·전자밀도·온도 갱신의 전체 Jacobian이 아니다.

\[
a=\frac{g_l}{g_u},\qquad
K_\tau=C_{\rm Sob}f_{lu}\lambda t_{\rm exp},\qquad
k_\tau=C_{\rm Sob}f_{lu}\lambda\frac{\nu}{c}.
\]

\[
\tau=K_\tau(n_l-a n_u),\qquad
\eta=\frac{h\nu}{4\pi}A_{ul}n_u.
\]

\(r=n_u/n_l\)라 하면, `srce_chk=0`인 양의 τ에서

\[
L=(1-\beta)
\frac{h\nu A_{ul}}{4\pi k_\tau}
\frac{r}{1-ar}.
\]

off 계수로부터 고립 전이의 평형비는

\[
r_{\rm off}(J)=\frac{B_{lu}J}{A_{ul}+B_{ul}J}.
\]

다음 두 양을 정의한다.

\[
\kappa=
\frac{h\nu B_{lu}/4\pi}{k_\tau},
\qquad
\Delta B=B_{ul}-aB_{lu}.
\]

따라서 다음 복사 생산의 조건부 예측은

\[
\boxed{
F_k(J)=\beta_{k+1}J_{c,k+1}
+(1-\beta_{k+1})
\frac{\kappa A_{ul}J}{A_{ul}+\Delta B J}
}
\]

이다. \(\beta_{k+1},J_{c,k+1}\)는 저장된 다음 call 값을 조건으로 사용한다. 그러므로 이는 전체 이력을 독립적으로 예보하는 계산이 아니라 **관측된 배경에서 물질 되먹임 관계를 시험하는 조건부 한 단계 예측**이다.

배경 β·J_c를 고정했을 때의 미분은

\[
\boxed{
q_k(J)=
(1-\beta_{k+1})
\frac{\kappa A_{ul}^{\,2}}
{(A_{ul}+\Delta B J)^2}
}
\]

이다.

Einstein 중량 관계가 정확히 맞아 \(\Delta B=0\)이면

\[
F_k(J)=\beta_{k+1}J_{c,k+1}
+\kappa(1-\beta_{k+1})J,\qquad
q_k=\kappa(1-\beta_{k+1}).
\]

추가로 \(\kappa=1\)일 때에만 익숙한 \(q=1-\beta\)와

\[
\delta_k^{(0)}
=\beta_{k+1}\left(\frac{J_{c,k+1}}{J_k}-1\right)
\]

가 나온다.

**생산의 \(\kappa\), \(\Delta B\), 직렬화된 \(\lambda\nu/c\)를 1·0·1로 강제하지 않는다.** T2의 `line_net_einstein_opacity_ratio`가 보존하는 정규화 비를 같은 정의로 계산한다. β가 매우 작으면 \(1-\kappa\)가 β보다 큰 변화항일 수 있다.

\(q\ge1\)이면 수축이라고 부르지 않는다. β·J_c·다른 준위가 인구에 따라 바뀌는 실제 반복의 전체 미분이나 수렴 반복 수로 승격하지 않는다.

### 4.3 정량 예측과 반증 기준

E′의 1206.50 Å, 각 셸, \(k=1,\ldots,6\)에 대해 다음을 계산한다.

\[
\widehat J_{k+1}=F_k(J_k),\qquad
\widehat\delta_k=\frac{\widehat J_{k+1}-J_k}{J_k},
\]

\[
\rho_k=
\frac{|J_{k+1}-\widehat J_{k+1}|}
{|\widehat J_{k+1}-J_k|}.
\]

등록한 **단순 off 사상 생존 기준**은 다음의 결합이다.

1. 필요한 operator·물질·원자 계수 조건이 적격하다.
2. 모든 비교에서 \(0<q_k<1\), \(1-q_k\le10^{-3}\).
3. 예측 변화량이 수치적으로 0과 분리된다.
4. 모든 비교에서 \(\rho_k\le1/4\).

\(1/4\)은 단순 기전 모형의 변화량 설명 허용 범위이며 수치 오차가 아니다. 분모를 작은 상수로 바꾸지 않는다. 예측 변화가 수치적으로 미분리이면 그 비교의 정량 판정도 미분리다.

β만 남긴 이상화 예측과 생산 계수를 보존한 예측을 함께 출력한다. 전자가 실패하고 후자만 생존하면 **“β만의 변화율 설명은 반증되었고, 생산 정규화 비를 포함한 느린 되먹임 설명은 생존했다”**고 구별한다.

실제 \(\delta_k\)로 \(\kappa\), 감쇠율 또는 허용폭을 맞추지 않는다. 관측된 \(L_{k+1}\)를 \(F_k\)의 예측 항에 대입하지 않는다.

단순 사상이 실패해도 씨앗 유지라는 관측 사실이 자동으로 반증되지는 않는다. 다준위 교환·물질 갱신 등 생략한 효과를 이 단계에서 새 계산으로 추적하지 않고 §6의 부분 설명 분기로 남긴다.

### 4.4 R′의 구조적 대조와 시간 순서

armed 계수는

\[
P_{lu}=B_{lu}\beta J_c,\qquad
P_{ul}=\beta A_{ul}+B_{ul}\beta J_c.
\]

고립 전이의 평형비에서는 β가 소거되어

\[
r_{\rm armed}(J_c)
=\frac{B_{lu}J_c}{A_{ul}+B_{ul}J_c}
\]

가 된다. 여기에는 해당 선의 **이전 L_l을 전체 J̄로 다시 넣는 직접 펌프 항**이 없다. 다준위·연속체를 통한 간접 되먹임까지 없다는 뜻은 아니다.

R′도 INIT에서는 armed가 아니다. 따라서 call 1→2를 NRB 효과의 필수 발현 구간으로 요구하지 않는다. call 2 이후에 조립된 armed 물질 갱신은 뒤의 복사 call에 반영된다.

등록한 관측 대조는 다음과 같다.

- §3의 공통 씨앗 조건.
- 해당 선의 후속 armed 적용 조건.
- 최종 이탈
  \[
  \left|\bar J^{R'}_7/\bar J^{R'}_1-1\right|>10^{-2}.
  \]

비단조 변화는 허용한다. 이 기준을 만족해도 “수렴했다”거나 “모든 선을 풀었다”고 쓰지 않는다. R′의 실제 준위비가 고립 전이의 \(r_{\rm armed}\)와 같을 것을 요구하지 않는다.

## §5. 준위비·여기온도·수치 기준

### 5.1 세대 7의 \(n_4/n_0\)

1206.50 Å의 같은 캡처 레코드에서 생산 역산을 수행한다.

\[
A_\nu=\frac{2h\nu_{\rm cap}^3}{c^2},\qquad
d=\frac{\tau}{K_\tau},
\]

\[
n_0=d\left(1+\frac{S}{A_\nu}\right),\qquad
n_4=\frac{g_4}{g_0}d\frac{S}{A_\nu}.
\]

\[
r_7=\frac{n_4}{n_0}.
\]

역산의 생산 상수·연산 순서·구간·소비 적격성은 기존 역산기를 승계한다. 여기서 \(\nu_{\rm cap}=c/\lambda_{\rm cm}\)인 캡처식과 W5·율의 저장 \(\nu\)를 구별한다.

이번은 연결선 총예산이 아닌 **동일 전이의 준위비 시험**이다. 준위 4의 21개 연결선을 다시 복원하여 교집합을 만드는 작업은 포함하지 않는다. 단일선 역산임을 명시하며 선행 교집합 결과를 새 계산으로 표시하지 않는다.

### 5.2 여기 예측

각 셸에서 다음을 나란히 보고한다.

\[
r_{\rm seed}=r_{\rm off}(\bar J^{E'}_1),
\]

\[
r_{\rm Boltz}(T)=
\frac{g_4}{g_0}
\exp\!\left[-\frac{E_4-E_0}{kT}\right],
\]

\[
T_{\rm exc}
=\frac{E_4-E_0}
{k\ln[(g_4n_0)/(g_0n_4)]}.
\]

- \(r_7^{E'}\), \(r_7^{R'}\).
- \(r_{\rm seed}\).
- \(r_{\rm Boltz}(T_{\rm inner})\)와 각 런의 \(r_{\rm Boltz}(T_{e,7})\).
- E′·R′의 \(T_{\rm exc}\).
- \(r_{\rm off}(\mathcal B_\ell)\)와 Boltzmann 비 사이의 원자 계수·에너지 표현 차이.

여기비의 주 예측은

\[
\left|\frac{r_7^{E'}}{r_{\rm seed}}-1\right|\le10^{-3}
\]

이다. 이것이 실패하면 **“준위비가 초기 씨앗에 묶인다”는 정량 예측은 반증**된다.

“\(T_{\rm inner}\) 여기”라는 추가 표현은 \(r_7^{E'}\)가 \(r_{\rm Boltz}(T_{\rm inner})\)에도 같은 기준으로 가깝고, 로그 비 거리에서 국소 \(T_e\) 예측보다 가깝다는 것이 구간으로 분리될 때만 사용한다.

\(T_e\)와 \(T_{\rm inner}\)가 가까워 두 예측을 구별하지 못하면 `TEMPERATURE_NOT_DISCRIMINATING`을 남긴다. 씨앗 추종 자체와 구별한다. 역전·로그 분모 0·비유한 값은 그대로 기록하고 유한한 여기온도로 고치지 않는다.

### 5.3 생산 해 재현 불가의 승계

직전 계약 §5.3의 `PRODUCTION_SOLUTION_NOT_TESTABLE`을 승계한다.

세대 7 캡처는 후속 물질 갱신의 평형해가 아니며, 후보 전자밀도 폐합·경계 인구·감쇠 이력을 모두 제공하지 않는다. 이번의 조건부 한 단계 사상은 그 생산 solve를 재현하지 않는다.

여기온도는 캡처된 준위비의 표현이다. 열역학적 평형 온도나 생산 해 수렴 판정으로 사용하지 않는다.

### 5.4 수치 구간과 과학 기준

- binary64·텍스트 writer·생산 연산 순서에 따른 반올림을 전파한다.
- W5의 J̄·J_c 오차 상한을 반영한다.
- 상쇄하는 변화량에는 절대오차를 유지한다.
- exp·log·프로파일 평균의 수치 오차는 연산 근거와 독립 정답 시험으로 확인한다.
- 공유 입력의 상관을 무시하여 구간이 독립적으로 좁아졌다고 주장하지 않는다.
- 결손을 넓은 오차막대나 다른 세대의 값으로 대체하지 않는다.

\(10^{-3}\), \(10^{-2}\), 0.99, \(1/4\)는 각각 등록된 과학 기준이다. 이를 수치 연산의 오차 허용폭으로 사용하지 않는다.

판정은 다음과 같다.

- 조건 전체를 구간이 만족: `SURVIVES`.
- 조건 위반이 구간으로 분리: `REFUTED`.
- 유효 구간이 경계를 가로지름: `UNRESOLVED`.
- 필요한 입력·권위·정의역이 없음: `NOT_IDENTIFIABLE`.

결합 예측은 확정 반례 하나로 반증할 수 있다. 다른 항목의 결손 때문에 이미 확인한 반례를 지우지 않는다.

## §6. 결과 분기와 사실 명제의 범위

### 6.1 1206.50 Å의 구성 예측

각 셸에서 별도로 기록한다.

| 기호 | 예측 |
|---|---|
| S | §3의 흑체에 가까운 공통 초기 씨앗 |
| H | §4.1의 E′ 씨앗 유지 |
| M | §4.1의 국소 방출 지배 및 §4.3의 생산 계수를 보존한 단순 off 사상 |
| X | §5.2의 E′ 준위비 씨앗 추종 |
| R | §4.4의 armed 구조·공통 씨앗·R′ 최종 이탈 |

온도 판별력, β만의 이상화 예측, 공간 균일성은 독립 보조 결과로 남긴다. 보조 결과를 숨기거나 주 예측의 성공으로 바꾸지 않는다.

### 6.2 셸별 결과 코드

아래 순서로 적용한다. 입력 오연결·판독기 산술 결함은 이 코드로 정상 처리하지 않고 §10의 `REWORK`로 분류한다.

| 우선순위·코드 | 이름 | 조건 |
|---|---|---|
| 2 | `SEED_RETENTION_REFUTED` | H가 반증됨 |
| 1 | `SEED_NOT_BLACKBODY` | H는 생존하고 S가 반증됨 |
| 3 | `RETAINED_BUT_MECHANISM_INCOMPLETE` | H·S는 생존하지만 M·X·R 중 확정 반증이 있음 |
| 0 | `MECHANISM_SURVIVES` | S·H·M·X·R 모두 생존 |
| 5 | `NOT_IDENTIFIABLE` | 위 확정 분기에 해당하지 않고 필요한 입력·정의가 결손 |
| 4 | `UNRESOLVED` | 나머지 유효 자료의 수치적 미분리 |

각 코드에 구성 예측의 상태와 반증된 조건을 붙인다. 코드 3은 원인 미특정의 정도를 나타내며 씨앗 유지의 관측을 취소하지 않는다.

주요 결론은 셸 19 코드다. 나머지 네 셸의 코드와 반례는 같은 표에 공개한다.

### 6.3 λ<2000 Å 두꺼운 선에 대한 사실 판정

고정 집합 \(\mathcal H_{\rm UV}\)에 대해 다음을 따로 집계한다.

1. 흑체 씨앗 유사성이 생존한 수.
2. E′ 씨앗 유지가 생존한 수.
3. 두 조건이 함께 생존한 수.
4. 반증·미분리·결손 수.
5. 동일 키 R′의 씨앗 유지와 최종 이탈 결과.
6. 이온·셸·β 구간별 분모와 위 수량.

불확실한 분류는 가능한 수량의 하한·상한으로 남긴다. 완전한 이력을 가진 행만 골라 전체 모집단의 비율로 표시하지 않는다.

다음 두 명제를 구별한다.

- **전칭 명제:** 등록된 \(\mathcal H_{\rm UV}\)의 모든 선·셸이 흑체에 가까운 초기 씨앗을 유지한다.
- **부분집합 명제:** 등록된 \(\mathcal H_{\rm UV}\) 중 명시한 선·셸 집합에서 그 사실이 성립한다.

전칭 명제는 반례 하나로 반증된다. 전칭이 실패해도 부분집합의 존재와 크기는 정당한 결과다. 모든 행이 적격하고 생존한 경우에만 등록 모집단 전체에 대한 전칭을 사용한다.

허용되는 최종 문장은 다음 형식이다.

> E′의 기록된 λ<2000 Å 두꺼운 선·셸 M건 중 N건에서, call 1–7의 J̄가 초기값의 0.1% 이내에 머물렀다. 그중 K건은 희석 없는 내경계 흑체의 프로파일 평균에도 0.1% 이내로 가까웠다. 반증·미분리·결손은 각각 …건이다.

선 개수 비율은 에너지·불투명도·광이온화율 기여율이 아니다. 이를 “λ<2000 Å 장 전체가 고정됐다”로 줄여 쓰지 않는다.

## §7. 최소 변경집합·산출

### 변경집합 — DET-SIIII-1206-MECH (표)

| 경로 | 함수·항목 | 허용 변경 |
|---|---|---|
| `scripts/det_siiii_1206mech_readout.py` | `det_1206mech_*` — 신설 | 고정 표적·센서스, W5 이력 연결, 흑체 프로파일 비교, 조건부 반복식, 캡처 준위비, 구간·분기·직접시험·manifest |
| `/gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007/tools/run_1206mech.slurm` | 오프라인 러너 — 신설 | 환경·신원·비용 기록, 시험·등록 분석·기계 게이트, 로그·rc 보존 |

### 변경집합 끝

새 함수의 접두는 `det_1206mech_`로 한다. CLI 진입도 이 접두 함수로 연결한다. 기존 판독기의 필요한 순수 보조 함수는 §1.2의 신원 기록 후 import할 수 있다.

생산 소스·바이너리·원자자료·봉인 입력·기존 판독기·기계 게이트·선행 결과는 변경하지 않는다. 계약 파일화와 판정·감사 문서는 운전자의 기록 업무이며 생산 변경집합에 추가하지 않는다.

| 필수 산출 | 내용 |
|---|---|
| `history.csv` | 1206.50 Å의 양 상태·다섯 셸·call 이력, 세대·operator, 원값·구간, 씨앗·변화량·local 몫, 예측·잔차 |
| `census.csv` | 고정 두꺼운 선·셸의 선택 키, 씨앗·최대 이탈·해당 call, 두께 변화, 반증·결손·상태 |
| `excitation.csv` | 다섯 셸의 세대 7 인구비·구간·여기온도, 씨앗·Boltzmann 비교 |
| `result.json` | 셸별 코드·구성 예측, 센서스·중간 두께 대조 집계, 사실 명제 범위, 내부 게이트·미해소 질문 |
| `selftest.json` | 실제 analyze 경로의 정답·구간·결함 주입 결과와 도구 신원 |
| `manifest.json` | 계약·입력·의존·소스·명령·산출 신원, 환경·작업·비용·이탈 |

원자료 전체를 결과에 복사하지 않는다. census의 최대 이탈 call과 원행 키로 원 W5에 돌아갈 수 있게 한다. 중간 두께 집합은 집계만 저장하며 개별 전체 이력 출력은 추가하지 않는다.

새 결과 경로가 이미 존재하면 덮어쓰지 않고 새 실행 식별자를 사용한다. 실패 기록도 보존한다. 롤백은 새 도구의 채택을 철회하는 것으로 한정하고 원 입력·선행 산출을 수정하지 않는다.

## §8. 게이트·직접시험

| 게이트 | 막는 실패 | 통과 조건 |
|---|---|---|
| G-LINK | 상태·선·셸·call·세대 혼합, INIT 칸의 허위 보충, 센서스 구성 변화, 결손 은폐 | §1–§3의 입력 권위·고정 키·상태 처리가 실제 산출까지 유지됨 |
| G-MECH | 정규화 비 생략, 시점 오류, 관측 L을 예측으로 재사용, 잘못된 구간·분기 | §4–§6의 독립 정답·구간 포함·반증 분기가 실제 analyze 출력에서 확인됨 |

과학 예측이 `REFUTED`여도 판독기가 이를 정확히 산출하면 내부 게이트는 통과할 수 있다.

### 8.1 직접시험 원칙

작은 생산 형식의 가상 입력에서 **실제 `det_1206mech_analyze`를 호출하고 저장된 산출을 검사**한다. 보조 함수만 시험하거나 기대 코드 상수를 직접 출력하는 시험은 인정하지 않는다.

- 대수 부분의 정답은 입력 binary64 값을 정확한 유리수로 해석해 독립 계산한다.
- exp·log가 필요한 부분은 독립 고정밀 상·하한으로 검사한다.
- 반환 구간은 독립 정답 또는 독립 정답 구간 전체를 포함해야 한다.
- 과도하게 넓은 구간으로 모든 시험을 통과시키지 않도록, 분리 가능한 예제에서 실제 분기가 분리되는지도 검사한다.
- 결함 주입 위치·기대 사유·실제 사유·rc를 남긴다.

### 8.2 필요한 시험 내용

시험 수를 별도 목표로 삼지 않고 다음 실패를 작은 가상 입력으로 묶어 확인한다.

1. **INIT·이력 연결**
   - INIT의 Sobolev 전용 칸이 없는 정상 입력.
   - call·셸·세대 교환, 모순 중복, 중간 call 결손.
   - call 7이 call 1로 돌아와도 중간 이탈이 있으면 유지 가설 반증.
   - 최종 두께가 바뀌어도 call 2 고정 집합에서 빠지지 않음.

2. **씨앗·프로파일**
   - 비상수 프로파일에서 중심값과 이산 평균을 구별.
   - inner_bb_scale·주파수·격자 가중치 누락을 검출.
   - 흑체와 다른 씨앗을 유지하는 경우 코드 1.

3. **조건부 사상**
   - \(\kappa=1,\Delta B=0\)의 이상화와 비단위 \(\kappa\)의 대수 정답.
   - \(\Delta B\ne0\)의 일반식·미분 정답.
   - β 생략, \(\kappa=1\) 강제, 관측 L의 예측 대입, R′ INIT부터 armed 처리의 결함.
   - 작은 변화량·상쇄·0과 겹치는 분모의 구간 처리.

4. **캡처·분기**
   - 알려진 양의 \(n_0,n_4\)에서 생산 τ·S를 만든 뒤 역산한 인구비의 정답 포함.
   - 씨앗 추종 성공·실패와 온도 판별 불가.
   - 세대 8 대입 거부, source fallback·소비 부적격 처리.
   - §6 코드, 경계 미분리, 전칭 반례와 부분집합 결과의 동시 보존.

기존 판독기를 수정하지 않으므로 선행 전체 율 예산 시험을 새 과제의 필수 배터리로 재실행하지 않는다. import한 역산·구간 기능은 이번 실제 analyze 시험에서 검증한다.

### 8.3 기계 프리플라이트

기존 도구를 그대로 사용한다.

- `scripts/check_prereg_preflight.py`
- `scripts/check_gate_referents.py`
- `scripts/check_doc_canonical.py`
- `scripts/check_dispatch_preflight.py`

검사 문서·발주서의 신원과 stdout·stderr·rc를 보존한다. 발주 단계에서 dispatch가 호출한 prereg 결과를 해당 단계의 증거로 사용하고 같은 문서 신원에 대해 중복 호출하지 않는다.

기계 게이트는 과학 예측·독립 사전감사·직접시험을 대신하지 않는다.

## §9. 환경·비용·재현 명령

### 9.1 환경

등록 환경은 **a10 CPU Slurm, GPU 예약 0**이며 수동 실행 장소는 lageunha다. 로그인 노드에서는 파일 조회·소스 판독만 수행한다. 직접시험·게이트·수치 분석은 계산 장소에서 실행한다.

러너는 CPU·메모리·시간·GPU 0을 명시하고 현행 노드 제외 규칙을 적용한다. module 초기화·실제 Python과 의존 버전·스레드 수·노드·작업 ID를 기록한다.

lageunha 수동 실행은 부하를 확인한 뒤 같은 자원 상한을 적용한다. 시간 측정은 Bash `time`을 사용하며 `/usr/bin/time`의 존재를 가정하지 않는다.

### 9.2 비용 상한

직전의 실제 규모를 이번 상한으로 채택한다. 직전 계약의 60분·256 MiB 여유를 승계하지 않는다.

| 항목 | 상한 |
|---|---:|
| 동시 계산 작업 | 1개 |
| CPU | 4개 |
| 작업 메모리 | 16 GiB |
| 시험·게이트·분석·재판독의 누적 실행 경과 | 60초 미만 |
| CPU 예약량 | 4 CPU×60초 미만, 약 0.067 core·h 미만 |
| 새 도구·시험·결과·로그 합계 | 22 MiB 이하 |
| GPU 예약·생산 실행 | 0 GPU·h·0회 |

Slurm과 수동 실행, 실패·수정 재판독을 모두 합산한다. 대기열 체류·문서 작성·사람 감사 시간은 계산 경과와 구별해 기록한다.

W5는 상태별로 제한된 순차 판독을 하고, 필요한 이력을 메모리에 모아 재사용한다. 선마다 전체 파일을 다시 탐색하지 않는다. 프로파일 비교는 필요한 선 주변 격자만 계산한다.

남은 상한 안에서 완료할 수 없으면 부분 산출·실제 비용·남은 작업을 보고한다. 자동 재제출이나 비용 확대는 하지 않는다. 상한 소진은 과학 예측 반증과 다르다.

### 9.3 계획 CLI

다음은 구현할 인터페이스다.

```bash
python3 -B "$P/scripts/det_siiii_1206mech_readout.py" selftest \
  --production-source "$T2" \
  --out "$D/results/$RUN_ID/selftest"

python3 -B "$P/scripts/det_siiii_1206mech_readout.py" analyze \
  --e-run "$E" \
  --r-run "$R" \
  --production-source "$T2" \
  --selftest "$D/results/$RUN_ID/selftest" \
  --out "$D/results/$RUN_ID/analysis"
```

표적·셸·집합·임계값은 본 계약대로 고정한다. 결과에 맞추는 CLI 옵션을 추가하지 않는다.

```bash
python3 -B "$P/scripts/check_prereg_preflight.py" \
  "$P/docs/PREREG_DET_SIIII_1206_MECH_2026-10-07.md" --root "$P"

python3 -B "$P/scripts/check_gate_referents.py" \
  "$P/docs/PREREG_DET_SIIII_1206_MECH_2026-10-07.md" --root "$P"

python3 -B "$P/scripts/check_doc_canonical.py" \
  "$P/docs/PREREG_DET_SIIII_1206_MECH_2026-10-07.md"

python3 -B "$P/scripts/check_dispatch_preflight.py" "$ORDER" --root "$P"
```

위 명령의 중복 집행은 §8.3에 따라 생략한다. 실제 argv·출력 경로·rc를 기록한다. 재현 실행도 새 결과 경로와 같은 누적 비용 상한을 따른다.

## §10. 처분·완료 조건·후속 결정

### 10.1 필수 완료 내용

판정문 첫 문단에는 다음을 명시한다.

- 스펙트럼 잔차 감소 0.
- 공개 CMFGEN 잣대는 UV2500 스펙트럼 비교에 있으며 이번 준위·반복식의 공개 정답이 아님.
- 생산 해 폐합 미달 승계.
- 발주 전 실측에 노출된 계약이라는 사실.
- 주요 셸의 기전 코드와 확정 반증·미해소 항목.

필수 본문은 다음을 포함한다.

- 1206.50 Å의 다섯 셸에 대한 씨앗·전체 이력·조건부 변화 예측·최종 준위비.
- β만의 예측과 생산 정규화 비를 포함한 예측의 구별.
- R′ INIT 비적용과 후속 armed의 시간 순서.
- λ<2000 Å 두꺼운 선의 고정 모집단·반례·부분집합·결손.
- 시험·게이트·신원·원명령·비용·보존·분장·이탈.
- 등록 질문마다 해소·부분 해소·미해소의 구별.

결손 목록만 작성하고 기전 시험 전체를 완료했다고 쓰지 않는다.

### 10.2 집행 처분과 과학 결과의 분리

| 처분 | 기준 |
|---|---|
| `PASS` | 제한 없는 전체 완료 처분. 본 단은 선행 생산 해 폐합 미달을 승계하므로 전체 처분으로 부여하지 않음 |
| `PASS_LIMITED` | 등록 판독·시험·산출이 적격하게 완료됨. 생산 해 미재현과 남은 미분리·결손의 영향을 명시 |
| `REWORK` | 계약과 다른 선택·연결·예측식·구간·분기·시험 구현, 결손 은폐 또는 주장 과장 |
| `BLOCKED` | 필수 입력·권위·등록 자원 등의 실제 문제로 최소 판독·증거 산출을 완료하지 못함 |

**전체 처분 상한은 `PASS_LIMITED`다.** 이번 질문을 좁혔다는 이유로 선행 미달을 삭제하지 않는다.

과학 예측의 `REFUTED`는 `REWORK`가 아니다. 올바른 입력·도구로 예측 실패를 분리했다면 그것이 등록 질문의 정당한 답이며 `PASS_LIMITED`로 완료할 수 있다.

### 10.3 결과 코드별 다음 걸음

| 주요 셸 코드 | 등록 후속 경로 |
|---|---|
| 0 `MECHANISM_SURVIVES` | 관측된 씨앗 유지와 조건부 기전의 설명 범위를 제시하고, 그 범위에 대한 수리안 오프라인 검증을 설계할지 사용자에게 결정 요청 |
| 1 `SEED_NOT_BLACKBODY` | 초기값 유지와 흑체 기원 가설을 분리. 흑체 씨앗 서술을 철회하고 실제 INIT 출력 형성의 오프라인 조사를 우선할지 사용자 결정 |
| 2 `SEED_RETENTION_REFUTED` | 해당 셸의 씨앗 고정 가설을 철회. 변화가 나타난 call·조건과 다른 셸의 결과를 보고하고 자동 반복 연장 없이 종료 |
| 3 `RETAINED_BUT_MECHANISM_INCOMPLETE` | 유지 사실은 보존하고 실패한 M·X·R을 지목. 다준위 교환·물질 갱신 등 어떤 누락을 후속으로 구별할지 사용자 결정 |
| 4 `UNRESOLVED` | 경계를 가로지르는 양과 필요한 분리 능력을 보고. 기존 입력의 수치 판독 개선이 상한 안에서 가능한지 먼저 판단 |
| 5 `NOT_IDENTIFIABLE` | 필요한 레코드·권위·정의의 결손을 구체적으로 보고. 자료 추가 여부는 사용자 결정이며 새 생산 캡처를 자동 발주하지 않음 |

보조 셸과 센서스는 후속 범위를 제한한다. 주요 셸 코드 0이라도 황 선의 결과가 다르면 황 전선까지 동일 기전으로 승격하지 않는다. 바닥 인구 항의 원인도 남겨 둔다.

**이번 단에는 GPU 판정런이 없다.** 후속 판정런은 다음 세 요건을 갖춘 별도 사용자 결정 뒤에만 검토한다.

1. 기전의 오프라인 특정과 반증 시험 결과.
2. 구체적인 수리안의 오프라인 검증.
3. 바뀔 관측량·방향·크기 또는 범위를 정한 기대치 사전등록.

그 뒤의 생산 검증도 등록된 판정런 1회로 설계한다. 본 단의 코드 0이나 `PASS_LIMITED`가 두 번째·세 번째 요건 또는 런 승인을 대신하지 않는다.

## §11. 집행 기록·분장

| 업무 | 담당 | 한계·규칙 |
|---|---|---|
| 기안·개정·종료감사 | astra | 실제 모델 ID 기록. 계약 저자가 이행을 감사하는 한계 명시 |
| Q-GOAL·Q-LEAN 사전감사 | Fable fresh | 독립 자기답. 호출 실패 시 gpt-6.1-sol, 실패 증거·실제 ID 보존 |
| 구현·직접시험·기계 게이트·실행·판독·판정 | 운전자 | 구현자·판정자 겸임 명시 |
| 코드 감사 | 해제 | 별도 정례 코드 감사 추가 없음 |
| 범위·비용 확대·후속 수리·생산 실행 | 사용자 | 이번 승인에서 자동 발생하지 않음 |

| 집행 항목 | 현재 상태 |
|---|---|
| 사용자 승인 | D의 사용자 결정문 직접 확인 |
| 선행 계약·판정·종료감사 | 직접 판독 |
| 발주 전 실측 | D의 w5_1206_history_2026-10-07.txt 직접 판독 |
| T2 초기 장·프로파일·bb 율·Sobolev 경로 | §1.5의 소스 직접 판독 |
| 이번 수치 분석·전체 센서스 재현 | 미집행 |
| 파일화·독립 사전감사·기계 게이트 | 미집행 |
| 구현·직접시험·실행·판정 | 미집행 |
| 종료감사 | 미집행 |

기안자는 파일 수정·수치 분석·시험 실행·작업 제출을 하지 않았다. 미래 집행 항목을 예상 PASS로 채우지 않는다. 과거 원로그가 없으면 그 사실을 기록하고 새 실행을 과거 증거로 표시하지 않는다.

## §12. 기계 선언

아래 선언은 §7의 변경집합·참조 실물 및 §6.2의 **결과 코드 부호화**를 검사한다. 과학 예측·구간 계산·시점 연결·코드 선택 알고리즘의 적합성은 직접시험과 판정에서 확인한다.

`v`는 선언 검사에 사용하는 결과 코드 하나다. 코드 0–5 외 값, 결손, 복수 값은 `X`다. 코드 9는 음성대조용이다. 과학 예측 상태와 집행 처분은 이 부호화에 섞지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-SIIII-1206-MECH (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_siiii_1206mech_readout\\.py|/gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007/tools/run_1206mech\\.slurm)",
    "symbol": "det_1206mech_",
    "roots": [
      "scripts/det_siiii_1206mech_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007/tools/run_1206mech.slurm"
    ],
    "expected_extra": [
      "scripts/det_siiii_1206mech_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007/tools/run_1206mech.slurm"
    ]
  },
  "branches": {
    "regimes": [
      [0.0, 0.0],
      [1.0, 1.0],
      [2.0, 2.0],
      [3.0, 3.0],
      [4.0, 4.0],
      [5.0, 5.0],
      [9.0, 9.0]
    ],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,2,3,4,5) for x in v)",
      "c0": "sum(1 for x in v if x == 0)",
      "c1": "sum(1 for x in v if x == 1)",
      "c2": "sum(1 for x in v if x == 2)",
      "c3": "sum(1 for x in v if x == 3)",
      "c4": "sum(1 for x in v if x == 4)",
      "c5": "sum(1 for x in v if x == 5)"
    },
    "rules": [
      {
        "name": "X",
        "predicate": "not valid"
      },
      {
        "name": "MECHANISM_SURVIVES",
        "predicate": "valid and c0 == 1"
      },
      {
        "name": "SEED_NOT_BLACKBODY",
        "predicate": "valid and c1 == 1"
      },
      {
        "name": "SEED_RETENTION_REFUTED",
        "predicate": "valid and c2 == 1"
      },
      {
        "name": "RETAINED_BUT_MECHANISM_INCOMPLETE",
        "predicate": "valid and c3 == 1"
      },
      {
        "name": "UNRESOLVED",
        "predicate": "valid and c4 == 1"
      },
      {
        "name": "NOT_IDENTIFIABLE",
        "predicate": "valid and c5 == 1"
      }
    ],
    "adversarial_fixtures": [
      {"name": "mechanism-survives", "mix": [[0,1]]},
      {"name": "non-blackbody-seed", "mix": [[1,1]]},
      {"name": "retention-refuted", "mix": [[2,1]]},
      {"name": "mechanism-incomplete", "mix": [[3,1]]},
      {"name": "unresolved", "mix": [[4,1]]},
      {"name": "not-identifiable", "mix": [[5,1]]},
      {"name": "invalid-code", "mix": [[9,1]]},
      {"name": "missing-code", "mix": []},
      {"name": "multiple-codes", "mix": [[0,1],[2,1]]}
    ]
  },
  "references": [
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/CLAUDE.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PREREG_DET_SIIII_UPSTREAM_2026-10-07.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_DET_SIIII_UPSTREAM_2026-10-07.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/astra_closeout.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/results/t4/identity.sha256"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007/evidence/USER_DECISION_2026-10-07.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007/evidence/w5_1206_history_2026-10-07.txt"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/READOUT_CAUSE1_SII_IONIZATION_CHANNEL_2026-09-25.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/READOUT_CAUSE2_FUV_LINE_LAMBDA_STALL_2026-09-25.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/det_siiii_pop_readout.py",
      "flags_existing": [
        "recover",
        "capture_records",
        "prod_tau_S"
      ]
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/det_siiii_rates_readout.py",
      "flags_existing": [
        "det_siiii_rates_exponx",
        "det_siiii_rates_rad"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c",
      "flags_existing": [
        "cm_planck",
        "cmf_inner_bb_scale",
        "cmfgen_fine_jbar",
        "INIT_SHARED_GAUSSIAN",
        "CMFGEN_NONOVERLAP_SOBOLEV",
        "w5_cont_lines_write"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c",
      "flags_existing": [
        "bb_nrb_arm",
        "nlte_assemble_rate_matrix_impl",
        "nlte_bb_jbar_canonical",
        "line_net_bracket_rates"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_net_rate.c",
      "flags_existing": [
        "line_net_sobolev_material",
        "line_net_sobolev_radiation",
        "line_net_cmfgen_exponx",
        "line_net_einstein_opacity_ratio",
        "line_net_bracket_rates"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_jbar.c",
      "flags_existing": [
        "line_jbar_gaussian_discrete_shells"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_jbar.h",
      "flags_existing": [
        "LINE_JBAR_PROFILE_NDOPPLER",
        "LINE_JBAR_BB_LAMBDA_MIN_ANGSTROM",
        "LINE_JBAR_BB_LAMBDA_MAX_ANGSTROM"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina.h",
      "flags_existing": [
        "SOBOLEV_COEFF"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_atomic.c",
      "flags_existing": [
        "load_atomic_data",
        "line_A_ul",
        "line_B_lu",
        "line_B_ul"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/opacity_publication.c",
      "flags_existing": [
        "a208_signed_sobolev_counted",
        "a208_line_source_counted"
      ]
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/check_prereg_preflight.py"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/check_gate_referents.py"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/check_doc_canonical.py"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/check_dispatch_preflight.py"
    }
  ]
}
```