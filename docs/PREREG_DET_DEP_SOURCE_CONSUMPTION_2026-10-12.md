# 사전등록 - DET-DEP-SOURCE-CONSUMPTION: 고정 상태의 침적 소비와 직접 원천 연결 대조

2026-10-12 · astra 읽기 전용 기안  
상태: 기안. 소스·저장 자료 조회 완료. 구현·직접시험·빌드·계산·제출·판정 미집행.

## §0. Q-GOAL·Q-LEAN 자기답

**Q-GOAL: PASS.** 최종목적은 공개 CMFGEN과 비교 가능한 SN Ia 이온화 상태와 스펙트럼을 얻는 것이다. 이번 단은 요청됐지만 결정론 경로에 연결되지 않은 직접 침적 원천을 같은 물질 상태에서 연결하여, 실제 소비·광도 응답·침적의 중복 청구를 구별한다. 이를 생략하면 원천 요청량을 탈출 광도로 간주하거나, 공개 온도에서 유지되는 기존 방출에 침적을 더한 결과를 물리적 수리로 오인할 수 있다.

관측 가능한 종료 산출은 **가지별 소비 장부·경계·관측 광도·공개 대비를 담은 표 하나**다. 이 표는 연결이 소비처까지 도달했는지, 응답이 어느 파장에 나타나는지, 명시적 원천 분할이 지켜졌는지, 물리적 중복 여부가 어디까지 남는지를 판단하게 한다.

**Q-LEAN: APPROPRIATE.** 상태 재현은 GPU 작업 한 번으로 제한하고, 그 상태에서 기준·연결 전달만 분기한다. 기준과 연결을 각각 처음부터 진화시키지 않는다. O는 기존 기준점으로 재사용하지만, 저장되지 않은 같은 상태의 물질 장부와 순경계 광도를 갖춘 것으로 취급하지 않는다.

저장 포획의 관측 재생 경로는 있으나 전달장 재수렴과 물질 장부 재평가를 함께 수행하는 준비된 재시작 경로는 확인되지 않았다. 새 재시작 체계 대신 기존 실행의 물질13·복사13 지점에서 분기한다.

저장 산출 재검 장치·쓰기 결함 모사·결정표 재계산 장치·새 전 원소 순위표를 만들지 않는다. 보조 산출의 소비처는 주 표의 수치·신원·미정의 사유·직접시험 적격성·비용으로 한정한다.

## §1. 권한·기반·봉인

기안·개정·모든 감사는 astra, 구현·직접시험·게이트·빌드·스테이징·제출·판정은 운전자다. 코드 감사는 없다. GPU·생산 변경 단이므로 빌드·구현 발주 전에 astra의 Q-GOAL·Q-LEAN 경량 사전감사 한 번을 받는다. 감사자는 계약 기안자라는 이해관계를 명시한다. 종료감사도 astra가 수행한다.

| 약칭 | 위치 |
|---|---|
| P | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn` |
| B | `/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012` |
| O | `/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1` |
| T | `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1` |
| S | `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src` |
| D9 | `/gpfs/kjhan/lumina/cmfgen_map/diag9_absdef_20261011` |
| D10 | `/gpfs/kjhan/lumina/cmfgen_map/diag10_energybudget_20261012` |
| C | `/gpfs/kjhan/cmfgen_src/cur_cmf` |
| U | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06` |

사용자 선택은 B의 `evidence/USER_DECISION_2026-10-12.txt`다. 선택지 설명의 구현 문장을 수용 기준으로 승계하지 않는다.

생산 변경 기반은 S를 포함한 봉인 소스 판본이다. 저장소 작업본을 그대로 빌드하지 않는다. 기안자의 `cmp -s` 조회에서 `radeq_publication.c`는 같았고 `lumina_cuda.cu`·`lumina_cmfgen.c`·`lumina_plasma.c`·`physics_comparison.c`·`env_universe.h`는 달랐다.

운전자는 봉인 빌드 목록에 따라 필요한 파일을 B의 독립 작업본으로 복사하고 변경집합 밖 차이가 없음을 확인한다. 봉인 입력·O·T·기존 결과를 수정하지 않는다. 하드링크와 파일시스템 고유 기능을 사용하지 않는다.

## §2. 과학 질문·사전 확인·비용 절감 선택

> 물질13 상태와 경계·기존 계수를 고정한 결정론 전달에서 요청된 직접 침적 원천을 연결하면, 실제 조립과 관측 소비량은 요청량에 대응하는가? 그 변화는 순경계·관측 광도와 대역 광도에 어떻게 나타나는가? 직접 원천에 배정한 에너지를 물질 가열 또는 새 비열적 소비에 다시 청구하는가?

기안 조회에서 다음을 확인했다. 수치는 저장 원자료 인용이며 독립 재계산 결과가 아니다.

- O의 직접 원천 요청과 침적 게이트는 1이다. lumina_cuda.cu 약 8093행의 DET 호출은 반환으로 끝나며 setter는 약 8302행의 우회된 경로에 있다.
- lumina_cmfgen.c 약 2169행의 setter 기본 상태는 NULL이다. 약 2864행의 침적 정규화와 약 3023행의 조립 주입은 이 상태에서 직접 원천을 만들지 않는다.
- O 포획은 물질13·복사13·iter11·조립13·요구 및 계산 tau26이다. 직접 침적 레인은 50셸 모두 0이라는 D10 원자료와 대응한다.
- D10의 복원 전체 순교환은 약 `-2.018984e42 erg/s`, 입력 침적은 약 `7.787639e42 erg/s`다. 두 값의 합은 약 `5.768655e42 erg/s`다.
- 저장 물질 장부 A10은 물질13·복사12, A11은 물질14·복사13이다. 이를 C13과 같은 상태의 물질 장부로 쓰지 않는다.
- T의 기존 판정에는 O의 물질13 비교 덤프 비트 재현이 기록돼 있다. 이것은 새 변경본의 무영향 증명을 대신하지 않는다.
- O 마지막 구간 전달 기록은 `max_rel=9.987495e-04`다. 이것만으로 이번에 등록하는 엄격한 전달 수렴을 입증하지 않는다.

**싼 경로 검토.** lumina_cmfgen.c 약 6033행의 `det_obs_capture_replay`는 저장 J·계수·선 포획으로 관측 전달을 재생한다. J를 다시 수렴시키거나 물질 장부를 재평가하지 않는다. 저장 최종 물질 산출은 관측 C13 이후 물질14와도 구별해야 한다. 따라서 기존 관측 재생만으로 이번 질문 전체를 대체하지 않는다.

**채택 설계.** O를 기준점으로 재사용하고 상태 재현 한 번 뒤 기준 A와 연결 B를 같은 실행 안에서 순차 평가한다. O와 A의 차는 수렴 절차 차이고 B와 A의 차가 등록 원천 연결 효과다. A는 추가 생산 런이 아니다.

## §3. 가지·고정 상태·실행 위치

새 게이트는 `LUMINA_DET_DEP_SOURCE_TEST`이며 미설정 또는 0은 기존 경로, 1은 본 시험이다. 다른 문자열은 실행 전에 거부한다. 기존 `LUMINA_CMF_DEP_SOURCE=1`의 전역 의미를 조용히 바꾸지 않는다.

게이트 1에서도 O의 상태 생성 동안 직접 원천은 연결하지 않는다. 마지막 R6 복사13 commit과 기존 관측 hook 완료 뒤, 마지막 R7 물질 갱신 전에 시험에 진입한다. lumina_cmfgen.c 약 11790행의 이 위치를 사용한다.

| 가지 | 역할 | 직접 원천 | 새 물질 갱신 |
|---|---|---:|---|
| O | 기존 저장 기준점과 실행 중 재현 기준점 | 0 | 기존 이력만 사용 |
| A | 고정 물질의 수렴 기준 전달 | 0 | 없음 |
| B | 같은 물질의 직접 원천 연결 전달 | 요청 분율 1 | 없음 |

고정 목록은 T·전자밀도·밀도·기하·시각·원소 및 이온 밀도·NLTE 준위 인구·원자 자료·BF/FF 계수·선 tau·선 원천·매핑·epsilon·미세 선율 장·안쪽 경계다. 분기 전후 바이트 비교로 보존한다. 새로운 물질 인구·T·전자밀도 풀이, 선율 장 재생산, 마지막 R7 호출은 금지한다.

구간 J와 그 J를 소비하는 산란 원천만 가지별로 재수렴한다. 미세 선율 장을 고정했다는 사실과 구간 J가 변했다는 사실을 서로 다른 열로 기록한다. B를 새 복사13 생산 commit으로 가장하지 않고 `parent_material=13`, `parent_radiation=13`, `branch=A/B`, `solve_index`로 구별한다.

원래 최종 writer의 물질14 결과를 시험 산출로 내보내지 않는다. 시험 전용 종료 표식 뒤 정상 종료하며, 운전자는 원래 생산 완료와 구별한다.

## §4. 연결·원천 분할·변경집합

셸 입력 침적 밀도율을 \(d_s\), 체적을 \(V_s\), \(D_s=V_sd_s\)라 한다. B의 직접 배정은 \(D_{{\rm dir},s}=D_s\)이며 A에서는 0이다.

봉인 조립의 Planck 모양과 실제 주파수 지원을 사용한다.
\[
b_s=\sum_b B_\nu(\nu_b,T_s)\Delta\nu_b,\qquad
\eta_{{\rm dep},sb}=\frac{d_s}{4\pi b_s}B_\nu(\nu_b,T_s).
\]
정규화와 입력은 유한하고 \(b_s>0\)이어야 한다. 음의 침적·지원 누락·비유한·표현 소실을 0으로 대체하지 않는다.

B에서는 setter에 시험 소유의 직접 배정 배열을 연결한다. 실제 조립의 `S_fixed`와 관측 연속 원천에 들어간 값을 각각 기록한다. setter 호출이나 요청량 기록만으로 소비를 인정하지 않는다.

**원천 분할은 실제 평가 입력에 적용한다.** A의 물질 침적 가열 입력은 \(d_s\), B는 0이다. B에서 직접 배정분을 `A210_GAMMA` 또는 별도 비열적 열 항에 다시 넣지 않는다. 원본 `GammaDeposition`과 그 이력은 보존하고 시험 소유 배열을 물질 장부 평가기에 전달한다. 출력 단계에서 gamma 열만 빼는 방식은 허용하지 않는다.

분기 후 비열적 인구 풀이를 호출하지 않으므로 **새 비열적 소비 실행량은 0**이다. 그러나 물질13을 만든 비열적 율과 이력은 남는다. 봉인 코드의 `nonthermal_heat=0`은 비열적 전리 부재를 뜻하지 않는다. lumina_plasma.c 약 23923행은 침적으로부터 비열적 전리율을 만든다.

그 율에 `W_ION_EV`를 곱한 값은 처방의 에너지 척도로만 표시한다. 실제 전리·여기 에너지 소비로 대입하지 않는다. 실제 소비 에너지의 전 채널 대응이 없으면 그 값과 물리적 분할 닫힘은 미정의다. 기존 방출도 이미 지출된 침적 몫으로 자동 배정하지 않는다.

### 변경집합 - DET-DEP-SOURCE-CONSUMPTION (표)

| 파일 | 허용 변경 |
|---|---|
| `/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/source/src/lumina_cuda.cu` | 시험 전용 종료 전달, 기존 최종 writer 우회. 일반 DET setter 연결 금지 |
| `/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/source/src/lumina_cmfgen.c` | 게이트·C13 분기·setter 연결·고정 전달·소비·경계·관측 계측 |
| `/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/source/src/lumina_plasma.c` | 인구 풀이 없는 고정 상태 장부 평가와 침적 소비 이력 추출 |
| `/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/source/src/env_universe.h` | 새 게이트 등록과 등록 개수 갱신 |
| `/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/source/src/det_dep_source.h` | 시험 인터페이스·상태·성분별 유효성 선언 |
| `scripts/det_dep_source_readout.py` | 입력 판독·주 표 작성·미정의 전파·새 판독 코드 음성 대조 |
| `/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/tools/det_dep_source_negative.c` | 실제 새 실행 경로를 호출하는 음성 대조 |
| `/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/tools/run_det_dep_source_offline.slurm` | 제한 빌드·직접시험·스테이징·판독 |
| `/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/tools/run_det_dep_source_gpu.slurm` | 상태 재현과 가지 평가 한 번의 제한 실행 |

### 변경집합 끝

`radeq_publication.c`·`physics_comparison.c`와 원자·충돌·비열적 처방은 변경하지 않는다. 장부의 기존 평가 함수를 소비하되 `a210_production_bundle_ledger`처럼 새 인구 후보를 만드는 함수를 고정 상태 평가 대신 호출하지 않는다.

새 clamp/floor·소스 재정규화 보정·실패 시 다른 풀이로의 대체를 금지한다. 기존 코드의 보정 분기가 시험 가지에서 발동하면 보정 전 상태를 기록하고 그 가지의 수렴 주장을 중단한다. 원래 상태 생성 경로의 처방을 이번 단에서 수리하지 않는다.

## §5. 소비 장부·세대·부호

모든 체적 적분은 셸 공이동계, 광도 단위는 `erg/s`다.
\[
V_s=\frac{4\pi}{3}(r_{{\rm out},s}^3-r_{{\rm in},s}^3).
\]
양의 \(Q\)는 복사에서 물질로 향하는 순교환이다.

각 가지에서 직접 배정 \(D_{\rm dir}\), 조립 소비 \(E_{\rm dep}^{asm}\), 관측 원천 구성 소비 \(E_{\rm dep}^{obs-src}\)를 구별한다.
\[
E_{{\rm dep},s}^{x}=4\pi V_s\sum_b\eta_{{\rm dep},sb}^{x}\Delta\nu_b.
\]
관측 원천 적분은 탈출 광도가 아니다. 실제 저장된 가산 결과에서 직접 항이 표현됐는지도 확인하며 별도 명목 레인만 양수인 경우를 소비 성공으로 판정하지 않는다.

기존 방출은 BF/FF 고정 방출·매핑 선 고정 방출·비매핑 선 고정 방출·산란 방출로 나눈다. 비매핑 선에는 실제 \(\epsilon S+(1-\epsilon)J\)를 적용한다. 산란 양쪽을 물질의 순 에너지 생성으로 세지 않는다.

\[
Q_{{\rm tr},s}=Q_{{\rm material,bin},s}-E_{{\rm dep},s}^{asm}.
\]
\(Q_{\rm material,bin}\)은 구간 전달이 사용하는 기존 흡수와 방출의 순값이다. 직접 항을 뺀 \(Q_{\rm tr}\)에 직접 침적을 다시 더해 복사 수지를 만들지 않는다.

**물질 장부는 별도 정의로 유지한다.** 고정 인구·T·전자밀도·계수와 고정 선율 장, 가지별 연속 복사장을 사용해 기존 물질 평가식의 가열·냉각·Compton·단열·잔차를 평가한다. 물질 세대, 구간 J 가지, 선율 장의 생산 물질·복사 세대를 각각 기록한다. 같은 물질에서 평가해도 선 장부와 구간 \(Q\)의 연산자는 같지 않다.

\(H_s\)는 실제 물질 장부에 전달된 침적 가열, \(N_s^{new}\)는 분기 이후 새 비열적 소비다.
\[
D_s=D_{{\rm dir},s}+H_s+N_s^{new}+U_s^{alloc}.
\]
A는 \((D_{\rm dir},H,N^{new},U^{alloc})=(0,D,0,0)\), B는 \((D,0,0,0)\)를 요구한다. 이는 **시험의 명시적 배정 항등**이다. 물질13 생성 이력의 비열적 소비와 기존 방출까지 설명하는 물리적 에너지 항등으로 승격하지 않는다.

물질 잔차는 가열 합에서 냉각 합을 뺀 값으로 기록한다. 그 음수를 고정 T 유지에 필요한 조건부 외부 열원 척도로 표시할 수 있으나, 실제 시간 미분이나 측정된 열원으로 부르지 않는다. 실제 물질·복사 시간 항은 미정의이며 0을 대입하지 않는다.

## §6. 전달 수렴·순경계·관측 광도

A와 B는 동일한 분기 전 구간 J에서 시작한다. 기존 전달 연산자를 유지하고 각 가지의 고정 원천과 산란 응답을 수렴시킨다. 최대 반복은 가지별 4096이며 상한 도달은 미수렴이다.

연속 두 반복에서 모든 셀의
\[
\frac{|J^{k+1}-J^k|}{|J^{k+1}|+|J^k|}\le10^{-8},\qquad
\frac{|J-\Lambda[S_{\rm fixed}+rJ]|}{|J|+|\Lambda[S_{\rm fixed}+rJ]|}\le10^{-8}
\]
을 요구한다. 두 항이 모두 정확히 0인 셀만 잔차 0으로 정의한다. 그 밖의 영 분모·비유한·표현 소실은 미정의다. 반복 변화량만으로 수렴을 선언하지 않는다.

**순경계.** 실제 안쪽·바깥쪽 표면에서 양방향 강도를 평가한다.
\[
F_\nu(r)=2\pi\int_{-1}^{1}\mu I_\nu(r,\mu)\,d\mu,\qquad
L_{\nu,\rm net}(r)=4\pi r^2F_\nu(r).
\]
\(L_{\rm in,net}\)은 안쪽의 바깥방향 주입에서 되돌아오는 광도를 뺀 값이다. \(L_{\rm out,net}\)은 바깥쪽 탈출에서 외부 유입을 뺀 값이다. Planck 주입 규모나 마지막 셸 중점 강도를 순경계로 대체하지 않는다.

봉인 구간 형식해는 셸 중점 표본을 사용하므로, 물리 표면까지의 경로를 같은 고정 계수와 원천으로 평가하는 계측을 둔다. 실제 표면 위치·충격매개변수·적분 가중·지원 범위를 기록한다.
\[
R_{\rm boundary}=\sum_sQ_{{\rm tr},s}-(L_{\rm in,net}-L_{\rm out,net}).
\]
이 값은 중점 교환과 표면 플럭스의 이산 대응 차다. 0을 강제하거나 사후 \(U/t\)를 넣어 닫지 않는다.

**관측 탈출.** 기존 결정론 observer의 실제 소비 경로와 광선 가중으로
\[
L_{\nu,\rm obs}=8\pi^2\int I_{\nu,\rm obs}(p)\,p\,dp
\]
를 구한다. 직접 원천의 탈출 레인은 기존 성분 분해 연산을 재사용한다. 이 레인과 B-A 전체 응답은 다르다. 후자는 산란 J의 변화도 포함한다.

좌표 변환은 실제 경로의 \(q=\nu_{\rm cmf}/\nu_{\rm obs}\), 연속 광학깊이 \(q\chi ds\), 연속 원천 \(q^{-2}\eta ds\), 선 도약 \(q^{-3}S\)를 기록한다. 바깥 표면에서는 \(I_{\nu,0}=q^3I_{\nu,\rm obs}\), 주파수·방향 변환과 그 적분 가중을 함께 적용해 같은 표면 강도의 공이동 모멘트를 얻는다.

관측계와 공이동계에서 지원 밖으로 이동한 몫을 별도 기록한다. 변환 비교를 위해 속도를 0으로 둔 추가 물리 가지를 만들지 않는다. 구간 경계와 observer의 차에는 좌표뿐 아니라 전달 표현·각도 적분의 차도 있으므로 이를 전부 좌표 효과로 귀속하지 않는다.

**파장 적분.** \(L_\lambda=L_\nu c/\lambda^2\)에 `cm`에서 `A`로의 단위 변환을 적용한다. 각 파일의 실제 지원 적분과 공통 지원 적분을 구별한다. 공개 19.480일 스펙트럼과 공통 지원은 두 지원의 교집합이며 외삽하지 않는다.

공통 지원에서 각 스펙트럼을 자기 절점과 대역 끝점을 포함한 구간별 선형 함수로 적분한다. 대역은 `2500-4000`, `4000-6000`, `6000-8000`, `8000-10000`, `10000-25500 A`, 표적 합은 `2500-6000 A`다. 대역 전체가 지원되지 않으면 전체 대역값은 미정의다. 광학 정규화나 임의 배율을 적용하지 않는다.

## §7. 주 표·허용량·미정의

주 산출은 `dep_source_consumption.csv`이며 `dep_source_consumption.md`는 같은 표의 표시본이다. 행은 가지·범위·셸·성분 또는 대역으로 구별한다. O의 미저장 항목은 미정의로 남긴다.

| 열 묶음 | 필수 내용 |
|---|---|
| 신원 | O/A/B·셸 또는 전역·원본 파일·물질·복사·선율 장·계수 세대·가지 반복 |
| 소비 | D·직접 배정·조립 소비·관측 원천 소비·물질 가열·새 비열적 소비·이력 비열적 척도·미배정 |
| 교환 | 기존 고정 방출·산란·구간 Qmaterial·Qtr·물질 장부 항·잔차·두 정의의 차 |
| 경계·관측 | 안쪽 양방향·순경계·바깥 순경계·경계 잔차·관측 탈출·직접 탈출 레인·좌표 및 지원 차 |
| 공개 비교 | 공통 지원·각 대역·공개값·가지/공개·공개와의 차·B-A·A-O |
| 해석 | 수렴·성분 유효성·중복 청구 상태·미정의 원인·분기·허용 주장 |

전역 장부에는 부호합과 셸 절댓값 합을 함께 둔다. 결측 셸을 제외한 합을 전체 합으로 표시하지 않는다.

배정·단위 변환·성분 합 항등의 허용량은 피연산자 절댓값 합의 `2e-10`이다. 합이 정확히 0이면 차도 정확히 0이어야 한다. 이는 산술 허용량이며 물리 보존 합격선이 아니다. O 재현의 등록 비교는 허용오차 없이 바이트 동일성을 요구한다.

각 값은 `value`, `status`, `reason`, `dependencies`를 가진다. 숫자 0은 확인된 영값에만 쓴다. 성분·진술·셸의 미정의를 독립적으로 유지하며 미정의 피연산자를 쓰는 차·합·비·분류에 그대로 전파한다.

파생값도 유한성·범위·표현 가능성을 검사한다. 오버플로·언더플로에 의한 비영값 소실·양의 직접 항 가산 소실·비율 또는 잔차의 표현 소실은 미정의다. 중간 계산의 보정값이나 작은 양의 분모를 넣지 않는다.

한 대역의 공개값 결함은 그 비교와 의존 합만 무효화한다. 한 셸의 비열적 이력 미확인은 그 물리 분할 주장만 제한한다. 단위·기하·공통 격자·분기 신원·고정 상태 연결의 결함은 해당 공통 입력에 의존하는 상태 전체를 정지한다.

보조 산출은 원천 소비·고정 상태·경계·스펙트럼의 압축 원자료, 입력과 코드 신원, 직접시험 로그, 자원 기록뿐이다. 완성된 표를 다시 읽어 결정값을 만드는 두 번째 판독기를 만들지 않는다.

## §8. 기대치·사전 해석 분기

예상 가능한 것은 B의 셸별 직접 배정·조립·관측 원천 소비가 D에 대응하고, B의 물질 침적 가열과 새 비열적 소비는 0이라는 점이다. 배선 또는 소비 결함이 있으면 이 기대가 실패한다.

관측 탈출량·대역 분포·공개 대비 개선량은 사전에 예측하지 못한다. `O+D`를 예상 광도·상한·하한으로 사용하지 않는다. 모든 대역의 증가나 단조 개선을 수용 기준으로 두지 않는다.

D9·D10 원자료의 기준 척도는 공개 Lbol 약 `1.278780e43`, O 파일 적분 약 `3.946043e42`, 그 차 약 `8.841757e42 erg/s`다. 마지막 차는 지원이 일치하는 볼로메트릭 결손 측정으로 승격하지 않고 기존 절대 규모 비교로 표시한다. 공통 지원과 대역의 새 비교가 우선한다.

등록 응답 분해능은 해당 공개 광도의 1%다. 0.5%와 2%에서도 범주를 함께 표시한다. 이는 과학적 정확도나 수렴 오차 상한이 아니다. 계산 불확실성이 경계에 걸리면 그 범주를 미정의로 둔다.

| 분기 | 조건과 해석 |
|---|---|
| 0 `SOURCE_NOT_CONSUMED` | 유효한 실제 소비 계측이 직접 배정과의 불일치를 확인. setter·조립·관측 구성 중 최초 불일치 지점을 보고하고 광도 결손 원인으로 확대하지 않음 |
| 1 `RESPONSE_RESOLVED` | 실제 소비가 확인되고 공통 지원 또는 등록 대역 중 하나의 B-A가 분해능을 넘음. 부호·규모·파장과 공개 대비 거리 변화를 보고 |
| 2 `NO_RESPONSE_RESOLVED` | 실제 소비가 확인되고 모든 등록 광도 응답이 정의되며 분해능 이내. 원천 무효나 물리적 무영향으로 단정하지 않음 |
| 3 `INDETERMINATE` | 공통 신원·수렴·필수 소비 또는 응답 미정의로 위 조건을 결정할 수 없음. 정의된 독립 성분은 보존 |

중복 여부는 위 분기와 독립된 판정 축이다. 실제 평가 입력에서 직접 배정과 물질 가열 또는 새 비열적 소비가 겹치면 **명시적 중복 청구 확인**이다. 분할이 지켜지면 **시험 분기 내 명시적 중복 없음**까지만 인정한다.

공개 T와 같은 고정 T 및 그 상태의 기존 방출은 침적을 반영한 계산의 결과다. 따라서 명시적 분할 성공과 광도 증가가 함께 나와도 기존 상태에 내재한 에너지와의 물리적 중복은 남을 수 있다. 기존 방출의 침적 귀속·실제 비열적 에너지·시간 항을 닫지 못하면 **물리적 중복 미확인, 고정 상태 감도 시험**으로 제한한다.

공개 대비 거리가 줄면 해당 지원의 수치적 접근, 커지면 이탈, 일부 대역만 줄면 혼합 응답으로 보고한다. 어느 경우에도 연결 가지의 증가를 결손 해소나 정당한 물리로 단정하지 않는다.

## §9. 게이트·새 코드 직접시험

| 게이트 | 통과 조건 | 막는 실재 실패 |
|---|---|---|
| G-PLAN | 기존 prereg·참조·정본·발주 검사와 경량 사전감사 통과 | 미등록 변경·폐기 계약·과도한 계측을 발주 |
| G-SCOPE | 봉인 기반·변경집합·게이트 해석·OFF 동일성 확인 | 다른 작업본이나 게이트 OFF에서 바뀐 생산식을 사용 |
| G-STATE | O 재현·C13 진입·고정 목록 보존 | 물질14 또는 가지별로 달라진 인구를 같은 상태로 비교 |
| G-CONSUME | 소비처 계측·실제 분할 입력·산술 유효성 확인 | 요청량을 소비량으로 기록하거나 직접 침적을 열에 재청구 |
| G-SOLVE | 가지별 수렴·실제 표면·좌표·지원 정의 충족 | 반복 횟수만으로 수렴 선언, Planck 규모를 순경계로 대체 |
| G-READ | 성분별 판독·파생값 검사·의존 전파 수행 | 미정의가 전체 합·대역비·분기에서 유한한 값으로 부활 |
| G-RUN | 등록 한 번의 실행·종료·자원·입력 보존 | rc=0만으로 완료 선언, 자동 재실행 또는 비용 초과 |

직접시험은 아래 새 코드의 음성 대조로 한정한다. 성한 대조를 함께 실행하고, 주입값이 실제 판독 또는 실행에서 원래 값과 달라졌음을 확인한다. 기존 전체 시험 묶음을 복제하지 않는다.

1. **게이트·OFF:** 미설정·0에서 봉인 실행 함수와 새 빌드의 조립·전달·writer 결과가 바이트 동일해야 한다. 잘못된 게이트, C13 이전 주입, 잘못된 세대를 실제 진입 함수에 넣어 거부를 시연한다.
2. **고정 상태:** 고정 목록의 각 배열·세대 연결을 하나씩 변경하여 G-STATE가 해당 분기를 거부함을 시연한다. 문자열 표식만 바꾸는 시험으로 대체하지 않는다.
3. **소비·분할:** setter 누락, 조립 레인 누락, 관측 레인 누락, B의 물질 가열 재청구, 비열적 소비 재진입을 실제 소비 함수에서 각각 시연한다.
4. **수치 정의:** 음의 침적, 영 정규화, 비유한 입력, 유한 피연산자의 파생 오버플로, 비영 적분·직접 가산의 표현 소실을 주입하여 독립 성분과 의존값의 미정의를 확인한다.
5. **수렴·경계·좌표:** 미수렴 정지, 잘못된 표면 위치·플럭스 부호·광선 가중·q 거듭제곱·파장 단위·지원 밖 보간을 작은 실행 입력으로 각각 시연한다. 상수 강도·순수 주입 등 분석적으로 알려진 대조값을 사용한다.
6. **판독·전파:** 중복·누락 셸, 세대 혼합, 한 대역 결함, 한 셸 성분 결함, 공통 격자 결함을 실제 입력 판독부터 통과시켜 영향 범위와 전역 합·비·분기 전파를 확인한다.

O의 iter0010 비교 덤프와 C13 포획·결정론 스펙트럼을 새 실행의 주입 전 산출과 비교한다. 이것은 새 계측의 대조 자격 검사이며 새 저장 산출 재검 체계가 아니다. 불일치하면 B 실행을 시작하지 않는다.

## §10. 비용·실행·보존

**GPU 상한은 48 GPU·h, GPU 작업 1개, 모델 진입 1회, 재제출 0회**다. 기준·연결은 같은 작업 안의 고정 전달 평가다. 상한에는 상태 재현·기준 확인·두 가지 평가·계측을 모두 포함한다.

GPU는 h200 두 개를 우선 사용하고 할당 전에만 h100 두 개로 바꿀 수 있다. 각 GPU 메모리는 80 GB 이상, a40은 제외한다. `--ntasks=1`, `--cpus-per-task=16`, `--mem=96G`, `--time=24:00:00`, `--exclude=syn102,syn03`, `--no-requeue`, 작업명 `LUMINA_` 접두를 사용한다.

근거는 O의 약 18.8시간과 T의 약 38.97 GPU·h 선례다. 48 GPU·h는 완료 예측이 아니라 비용 상한이다. 대기 중 자원 변경은 모델 진입 전 같은 등록 실행으로 한정하며 중복 제출하지 않는다.

오프라인은 a10 CPU Slurm만 사용한다. 작업당 CPU 최대 4개·메모리 16G·벽시계 30분, 전체 **7200 할당초와 4 core·h** 중 먼저 닿는 상한을 적용한다. GPU 배정은 0이고 `--gres=gpu:0`을 명시한다. 빌드·음성 대조·기계 검사·판독·수정 실행·감사 재현을 모두 포함한다.

신규 저장은 소스·빌드·스테이징·원자료·실패 기록을 합쳐 **32 GiB**다. 스테이징은 파일 목록과 크기로 상한을 확인한다. 대형 선별 기록을 반복 저장하지 않고 고정 자료를 공유 입력으로 한 번만 보존한다.

각 작업에서 `module purge` 후 O 빌드 처방의 `gnu13/13.2.0`, 필요 시 `cuda/13.0.2`를 명시적으로 적재한다. 실제 도구·라이브러리·CPU·GPU 신원과 스레드 수를 기록한다. 로그인 노드에서는 조회만 한다.

런 루트는 B의 `PAIR/run1`이며 `OWNER.txt`에 단·운전자·계약·실행 목적을 기록한다. 이미 존재하는 실행 산출을 덮어쓰지 않는다. GPU 실패 후 자동 재개·다른 기종 재실행은 없다.

운전자는 아래 인터페이스를 구현하며 모든 연산 명령을 해당 Slurm 작업 안에서 실행한다.
```bash
python3 -B scripts/det_dep_source_readout.py selftest
bash /gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/tools/run_det_dep_source_offline.slurm prepare
bash /gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/tools/run_det_dep_source_gpu.slurm /gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/PAIR/run1
python3 -B scripts/det_dep_source_readout.py analyze --run /gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/PAIR/run1 --out /gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/results/run1
```
정확한 컴파일·링크·모델 명령, 유효 exports, 입력 목록·크기·SHA-256, 작업 번호·종료·할당 비용을 보존한다. 재현은 같은 명령과 입력의 새 출력 경로를 사용하며 등록 상한 안에서만 한다.

## §11. 판정·종료감사·주장 상한

운전자는 소비 연결, 고정 상태 보존, 수렴 기준·연결 차, 경계와 관측의 관계, 공개 대비, 명시적 중복과 물리적 중복의 구별을 한 판정에 담는다. 미소비·미수렴·미정의는 목표 미달 범위를 명시하며 중간 게이트 통과로 단을 성공 처리하지 않는다.

종료감사는 주 표의 원자료 연결·원천 분할·부호·세대·좌표·파장 지원·미정의 전파·주장 상한을 우선한다. 기안자 이해관계를 밝히고 저장 자료 조회와 독립 실행 재현을 구별한다. 폐합에는 주 표·필수 증거·운전자 판정·astra 종료감사가 필요하다.

생산 승격·물리적 정답·수리·새 T 계산·자기일관 효과·철족 일반화를 선취하지 않는다. 어느 분기도 후속 실행을 자동 승인하지 않는다.

기안자는 공유 문맥 전체, 현재 디렉터리와 프로젝트 신원, 지정 진단·감사·폐합 기록, 원자료 표·실행 설정·관련 소스를 조회했다. 대표 조회 명령은 다음과 같다.
```bash
cat /home/kjhan/.codex/SHARED_CONTEXT.md
pwd
cat CLAUDE.md
sed -n '8090,8120p' /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cuda.cu
sed -n '6020,6255p' /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c
sed -n '11546,12090p' /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c
```
파일 변경·새 수치 계산·해시 재산출·빌드·직접시험·제출은 미실행이다. 사전등록 검사 통과도 아직 주장하지 않는다.

집행 기록은 이 절에 둔다. 운전자는 사전감사, 기계 게이트 rc, 미실행 항목의 실제 결과, 구현·빌드 신원, 직접시험, 제출·비용, 판정, 종료감사와 이탈 사유를 기록한다. 다른 문서는 이 장부를 지목한다.

## §12. 기계 선언

참조는 기존 실재 파일만 지정한다. 분기 예시는 기존 프리플라이트의 문면 검사 입력이며 새 결정표 재계산 장치가 아니다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 - DET-DEP-SOURCE-CONSUMPTION (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_dep_source_readout\\.py|/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/(?:source/src/(?:lumina_cuda\\.cu|lumina_cmfgen\\.c|lumina_plasma\\.c|env_universe\\.h|det_dep_source\\.h)|tools/(?:det_dep_source_negative\\.c|run_det_dep_source_(?:offline|gpu)\\.slurm)))",
    "symbol": "det_dep_source_",
    "roots": [
      "scripts/det_dep_source_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/source/src/lumina_cuda.cu",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/source/src/lumina_cmfgen.c",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/source/src/lumina_plasma.c",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/source/src/env_universe.h",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/source/src/det_dep_source.h",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/tools/det_dep_source_negative.c",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/tools/run_det_dep_source_offline.slurm",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/tools/run_det_dep_source_gpu.slurm"
    ],
    "expected_extra": [
      "scripts/det_dep_source_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/source/src/lumina_cuda.cu",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/source/src/lumina_cmfgen.c",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/source/src/lumina_plasma.c",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/source/src/env_universe.h",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/source/src/det_dep_source.h",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/tools/det_dep_source_negative.c",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/tools/run_det_dep_source_offline.slurm",
      "/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/tools/run_det_dep_source_gpu.slurm"
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
      {"name":"SOURCE_NOT_CONSUMED","predicate":"valid and c0 == 1"},
      {"name":"RESPONSE_RESOLVED","predicate":"valid and c1 == 1"},
      {"name":"NO_RESPONSE_RESOLVED","predicate":"valid and c2 == 1"},
      {"name":"INDETERMINATE","predicate":"not valid or (c0 == 0 and c1 == 0 and c2 == 0)"}
    ],
    "adversarial_fixtures": [
      {"name":"not_consumed","mix":[[0,1]]},
      {"name":"response","mix":[[1,1]]},
      {"name":"no_response","mix":[[2,1]]},
      {"name":"indeterminate","mix":[[3,1]]},
      {"name":"invalid","mix":[[9,1]]},
      {"name":"missing","mix":[]},
      {"name":"multiple","mix":[[0,1],[1,1]]}
    ]
  },
  "references": [
    {"path":"docs/PREREG_DET_LINE_NET_PROVENANCE_2026-10-11.md"},
    {"path":"docs/PREREG_DET_COLL_PRESC_2026-10-09.md"},
    {"path":"docs/PREREG_DET_SI_CONSUMPTION_TRACE_2026-10-10.md"},
    {"path":"docs/VERDICT_DET_SI_CONSUMPTION_TRACE_2026-10-11.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/depsrc_20261012/evidence/USER_DECISION_2026-10-12.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag10_energybudget_20261012/DIAGNOSIS.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag10_energybudget_20261012/DRIVER_AUDIT_r1.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag10_energybudget_20261012/DRIVER_AUDIT_r2.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag10_energybudget_20261012/LOOP_CLOSED.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag10_energybudget_20261012/out/r1_budget_419905/summary.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag10_energybudget_20261012/out/r2_material_419922/summary.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag9_absdef_20261011/out/r2_energy_419677/scales.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/STAGE_IDENTITY.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/job.slurm"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/SUBMIT_COMMAND.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/RUN_FOOTER.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/resolved_lumina.exports"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/work/det_obs_capture/meta.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c","flags_existing":["cmfgen_set_deposition","det_obs_capture_replay","det_obs_final_pass_hook"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c","flags_existing":["a210_production_bundle_ledger","gamma_deposition_compute_nonthermal"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/radeq_publication.c"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f"},
    {"path":"data/standart_data1/toy06/spectra_toy06_cmfgen.txt"},
    {"path":"data/standart_data1/toy06/lbol_edep_toy06_cmfgen.txt"},
    {"path":"data/standart_data1/toy06/phys_toy06_cmfgen.txt"}
  ]
}
```