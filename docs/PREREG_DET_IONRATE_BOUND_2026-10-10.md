# 사전등록 — DET-IONRATE-BOUND: 철족·Si 전리율의 공개 CMFGEN 조건부 비교와 식별 한계

2026-10-10 · 읽기 전용 기안  
운전자 파일화 대상: `docs/PREREG_DET_IONRATE_BOUND_2026-10-10.md`  
상태: **기안. 구현·직접시험·게이트·계산·판정·종료감사는 미집행.**

## §0. Q-GOAL·Q-LEAN 자기답과 과학 질문

**Q-GOAL — 최종목적에 직접 기여하는가? 자기답: PASS.**

최종목적은 Monte-Carlo 수송으로 CMFGEN 수준의 SN Ia 스펙트럼을 재현하는 것이다. 이번 질문은 **「O의 Fe·Co·Ni·Si II 과잉과 관련하여, 공개 CMFGEN의 전자온도·전자밀도로 바꾸면 재결합 율의 방향과 크기가 얼마나 달라지는가? 공개 III/II 비를 유지하는 데 필요한 조건부 총 전리율은 얼마이며, 저장된 O 율과 비교하여 어느 채널까지 구분할 수 있는가?」**다.

공개 자료에는 채널별 율이 없다. 따라서 **실제 CMFGEN의 광이온화·비열적 전리·충돌 전리·재결합 율을 모두 복원하거나 공통 원인을 확정하는 것은 이번 입력만으로 불가능**하다. 이 한계를 숨기고 자체 CMFGEN 런이나 Lumina 처방값을 공개 CMFGEN의 실제 율로 대신하지 않는다.

새로 채우는 칸은 **셸·원소별 공개 III/II 비, 공통 원자 처방의 열적 상태 치환에 따른 재결합 율 비, 조건부 필요 총 전리율, 채널별 비교 가능 여부**다. 이 단을 생략하면 열적 상태 차이와 공개 자료로 식별할 수 없는 복사장·비열적 분배 문제를 구분하지 못한 채 다음 계산을 선택하게 된다.

**Q-LEAN — 불필요한 측정·게이트를 추가하는가? 자기답: APPROPRIATE.**

| 유지 항목 | 소비하는 비교·결정 |
|---|---|
| 공개·O의 II·III 분율과 속도 | 전리비 차이 및 셸 대응 |
| 공개·O의 \(T_e,n_e\) | 재결합 율의 열적 상태 치환 |
| 고정 원자 계수·단면적의 출처와 정의 | 조건부 재결합 비교의 의미 |
| 식별 가능한 O 저장 채널 율 | 필요 총 전리율과의 제한된 비교 |
| 입력·세대 신원과 실제 비용 | 다른 상태 혼합 및 자원 초과 방지 |

새 수송·평형해·준위 개체수 역산·스펙트럼 재적분·민감도 사다리는 만들지 않는다. 저장 산출 재검·쓰기 결함 모사·결정표 재계산 장치도 만들지 않는다. 종료감사는 과학 표의 수치·대응·주장 한정을 먼저 본다.

## §1. 기준선·작업 위치·범위

| 식별자 | 고정 대상 |
|---|---|
| 프로젝트 P | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn` |
| 기준 O | `/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1` |
| 작업 기반 B | `/gpfs/kjhan/lumina/cmfgen_map/ionrate_20261010` |
| 공개 자료 C | `data/standart_data1/toy06` |

사용자 결정은 B의 `evidence/USER_DECISION_2026-10-10.txt`에 기록된 전리율 비교 승인과 「감사는 astra로」다. O는 N에 `LUMINA_OMEGA_CMFGEN=1`을 추가한 12회 계산이다.

대상은 **Z=14,26,27,28, II↔III, shell_id=0…49**다. 총 200개 원소·셸 쌍을 유지한다. 철족 부재 셸도 삭제하지 않고 원소량 0을 표시한다.

II 과잉을 전체 저전리와 동일시하지 않는다. 직전 철족 표에서 안쪽 셸은 II와 III가 모두 공개보다 높았다. 이번 II↔III 비교는 IV 이상으로의 흐름이나 전체 평균 전하를 대표하지 않는다.

**생산 소스·바이너리·덱·환경변수 변경 0, 생산 모델 진입 0, GPU 0**이다. 자체 CMFGEN 실행, 기존 자체런 율의 참조값 채택, 침적률에서 비열적 전리율을 임의 배분하는 작업은 제외한다.

## §2. 잣대의 등급과 식별 한계

| 등급 | 정의 | 허용하는 주장 |
|---|---|---|
| `PUBLIC` | 공개 분율·\(T_e,n_e\) | 공개 상태와 O 상태의 차이 |
| `COMMON_ATOM` | 동일 원자 처방을 O·공개 열적 상태에서 평가 | 그 처방 안에서의 상태 치환 효과 |
| `BALANCE_CONDITIONAL` | 공개 III/II 비에 조건부 재결합 율을 곱함 | 가정한 평형에서 필요한 총 전리율 |
| `O_RECORDED` | 세대·대상·단위·생산 경로가 확인된 저장 율 | 해당 O 계산 경로의 기록값 |
| `NOT_IDENTIFIABLE` | 필요한 자료나 대응이 없음 | 해당 채널은 비교하지 못함 |

**`COMMON_ATOM`은 CMFGEN의 실제 율이 아니다.** CMFGEN 기원의 원자 자료를 사용해도 공개 계산의 원자 모형·준위 개체수·복사장·유효 재결합 처방이 같다는 증거가 되지 않는다.

공개 III/II 비는 정상상태에서 인접 단계 사이의 순전류가 0이고 다른 경로가 없을 때만 총 전리/재결합 율 비와 같다. 시간의존·전하교환·비인접 전이·경계 흐름을 공개 자료로 배제할 수 없으므로 이 등식을 관측 사실로 사용하지 않는다.

이전에 자체 CMFGEN 런은 공개 비교의 잣대로 부적격 판정을 받았다. 일부 자체런에서 반복 쌍 정합을 확보한 이력도 공개 해에 대한 권위를 회복하지 않는다. 이번 단은 그 자료를 분모로 사용하지 않는다.

## §3. 입력·시각·세대·공간 대응

고정 입력은 다음과 같다.

- O의 `work/physics_comparison/physics_DET_iter0010.ion.csv`, 대응 shell CSV·manifest.
- O의 `work/lumina_ion_channels.csv`, `work/lumina_plasma_state.csv`, `input/resolved_lumina.exports`, `STAGE_IDENTITY.txt`.
- O 덱의 `levels.csv`, `ionization_energies.csv`, `cmfgen_sigma_bf.bin`, 필요한 상위 이온 준위 자료 및 원자 자료 출처 기록.
- 공개 `ionfrac_{fe,co,ni,si}_toy06_cmfgen.txt`와 `phys_toy06_cmfgen.txt`.
- 재결합 계수 정의를 확인할 고정 소스 `src/lumina_plasma.c`.

공개 시각은 **19.480 d**, O의 epoch는 **1683072 s**다. 다른 시각·시간 보간으로 대체하지 않는다.

O 분율은 물질 세대 **13**, 복사 세대 **12**, iteration **10**인 고정 이온 덤프를 사용한다. 덤프의 NLTE 단계 밀도를 같은 행의 원소 총밀도로 나눈다.

`lumina_ion_pops.csv`의 확인된 헤더 다음 자료는 population_generation=14이므로 이 파일을 세대 13 분율의 대체물로 사용하지 않는다.

율 장부의 `gen`은 끝부분에서 274이며 **물질 세대 번호로 간주하지 않는다.** 최대 `gen`, `gen=13`, 마지막 행을 임의 선택하지 않는다. 생산 기록·작성 경로로 관측 상태에 대응하는 호출을 특정할 수 있을 때만 `O_RECORDED`로 채택한다. 대응이 없으면 해당 칸은 미식별이다.

율 장부에 Fe·Co·Ni가 있다고 선취하지 않는다. 실제 Z·stage 집합과 호출 의미를 확인한다. Si·S·Ca의 기록을 철족 율로 복제하지 않는다.

속도는 shell CSV의 경계 중간값을 km/s로 환산한다. 공개 각 파일에서 최근접 속도, 동률이면 낮은 속도를 선택하고 범위 밖은 미정의로 남긴다. 양쪽 속도와 차이를 기록한다. 공개 분율과 물리 자료는 같은 공개 속도에 있어야 하며 서로 다른 행을 묵시적으로 결합하지 않는다.

이 비교는 같은 질량 셸의 비교가 아니다. 중복 대응 공개 행을 독립 표본으로 세지 않는다. 입력·새 코드·재사용 함수의 경로와 SHA-256을 실행 증거에 남긴다.

## §4. 공개 전리비와 공통 원자 재결합 율

공개 열의 `si1,fe1,co1,ni1`은 II, 끝자리 2는 III라는 기존 명명 가정을 승계한다.

\[
x^O=\frac{f^O_{\rm III}}{f^O_{\rm II}},\qquad
x^C=\frac{f^C_{\rm III}}{f^C_{\rm II}},\qquad
d_x=\log_{10}(x^O/x^C).
\]

원소 총량 0인 O 셸에서는 형식적인 단계 밀도 비를 계산하지 않는다. II+III 합으로 분율을 재정규화하지 않는다.

**공통 원자 방사 재결합 진단**은 고정 덱의 결합–자유 단면적을 이용한 자발 Milne 계수로 정의한다. 상위 이온은 III 바닥상태 핵으로 고정하며, 그 핵으로의 전이가 식별된 II의 결합 준위만 합산한다.

\[
\alpha_{\rm M}(T)=
\frac{4\pi}{c^2}\left(\frac{h^2}{2\pi m_e kT}\right)^{3/2}
\sum_i\frac{g_i}{g_{\rm III,0}}
\int_{\nu_i}^{\infty}\nu^2\sigma_i(\nu)
e^{-(h\nu-\chi_i)/(kT)}\,d\nu .
\]

실제 적분은 저장 단면적 격자의 지원 범위로 제한하고, 구간폭·문턱·통계가중치·포함 준위 수를 기록한다. 범위 밖 단면적을 외삽하거나 누락 준위를 수소형으로 채우지 않는다. 유도 재결합·들뜬 상위 핵·미저장 고준위 기여는 포함하지 않는다.

핵 대응·가중치·단면적 정체성을 정할 수 없으면 해당 계수는 미식별이다. 공명이 포함된 단면적의 역변환을 순수 RR로 단정하지 않고 `MILNE_WITH_RESONANCES`로 표시한다.

**공통 원자 DR 진단**은 고정 소스의 해당 III→II 계수에서
\[
\alpha_{\rm D}(T)=T^{-3/2}\sum_j c_j e^{-E_j/T}
\]
로 평가한다. 계수·에너지 단위·출처·적용 범위를 표에 연결한다. 경험적 배율은 1, 추가 floor는 0이다. 이 정의는 O의 실제 활성 처방과 구별한다.

`DR_TABLE`이라는 저장 위치만으로 순수 DR임을 인정하지 않는다. 원자료가 RR+DR 총계수이면 `TOTAL_RECOMB`로 표시하고 RR·DR 분리를 미식별로 남긴다. 누락 계수는 0으로 바꾸지 않는다.

각 적격 계수 \(k\)에 대해
\[
R_k^{O,*}=n_e^O\alpha_k(T_e^O),\quad
R_k^{C,*}=n_e^C\alpha_k(T_e^C),\quad
d_k=\log_{10}(R_k^{O,*}/R_k^{C,*}).
\]
별표는 공통 처방의 진단값이며 실제 코드 율이 아님을 뜻한다. 단위는 s⁻¹, 재결합하는 III 이온 하나당이다.

## §5. 채널 비교와 조건부 필요 총 전리율

주 표의 채널은 광이온화·비열적 전리·충돌 전리·방사 재결합·이전자 재결합이다.

광이온화·비열적 전리의 **실제 CMFGEN 분모는 미식별**로 사전등록한다. 공개 탈출 스펙트럼을 셸별 \(J_\nu\)로 사용하지 않으며, 공개 침적 에너지를 특정 이온의 전리 사건률로 변환하지 않는다.

충돌 전리도 공개 준위 개체수·유효 계수가 없으므로 실제 CMFGEN 비는 미식별이다. 새로운 근사 충돌 처방을 도입하지 않는다. 식별된 O 저장 율만 원량으로 남긴다.

O의 율은 하위 II 이온당 전리 s⁻¹과 상위 III 이온당 재결합 s⁻¹을 구분한다. `c_rec`의 물리적 정의를 확인하기 전에는 RR 또는 DR에 합산하지 않는다. 단계 평형용 보조 장부를 NLTE 준위 행렬의 개체수 가중 율로 표시하지 않는다.

RR와 DR의 분리가 적격이고 중복이 배제된 경우만
\[
R^{C,*}=R_{\rm M}^{C,*}+R_{\rm D}^{C,*},\qquad
U_{\rm req}^{C,*}=x^C R^{C,*}
\]
를 계산한다. \(U_{\rm req}^{C,*}\)는 **공통 원자 처방·정상 인접단계 평형에서 필요한 총 전리율**이며 실제 CMFGEN 총 전리율이 아니다. 분리 부적격이면 이 합과 파생 비교도 미식별이다.

동일 O 상태·경로에 속한 세 전리 채널이 모두 식별된 경우에만
\[
U^O=\Gamma_{\rm phot}^O+\Gamma_{\rm nt}^O+C_{\rm ion}^O,\qquad
d_U=\log_{10}(U^O/U_{\rm req}^{C,*})
\]
를 보고한다. 빠진 채널을 0으로 간주하지 않는다.

\(U_{\rm req}^{C,*}-C_{\rm ion}^O\) 같은 잔차를 CMFGEN 광이온화 또는 비열적 전리로 명명하지 않는다. O 분율에 재결합 율을 곱한 값을 O의 측정 전리율로 대신하지 않는다.

## §6. 주 표·기대치·공통성 판단

**주 산출은 CMFGEN 대비 표 하나**다. 200개 원소·셸 쌍마다 다음 7행, 총 **1400행**을 유지한다.

| 행 | 비교값 |
|---|---|
| `ION_RATIO` | \(x^O,x^C,d_x\) |
| `PHOTOIONIZATION` | 식별된 O 원량, 실제 CMFGEN 비는 미식별 |
| `NONTHERMAL_IONIZATION` | 같은 규칙 |
| `COLLISIONAL_IONIZATION` | 같은 규칙 |
| `RADIATIVE_RECOMBINATION` | 공통 Milne 율·dex, 적격성 표지 |
| `DIELECTRONIC_RECOMBINATION` | 공통 DR 율·dex 또는 분리 불가 |
| `TOTAL_IONIZATION_REQUIRED` | \(U_{\rm req}^{C,*}\), 적격한 경우 \(U^O,d_U\) |

필수 열은 셸·Z·II↔III·공개 대응 속도·\(T_e,n_e\)·비교 등급·원량·단위·비·dex·수치 구간·상태·사유·원자 처방 식별자다. 실제 O 율과 공통 처방 O 율은 별도 열로 둔다.

**기대치**

- 철족 존재 셸의 II 분율 과잉은 선행 사실이다. \(d_x\)의 전 셸 부호와 크기는 계산 전에 단정하지 않는다.
- Si·철족의 \(d_{\rm M},d_{\rm D},d_U\) 부호·크기·방사상 순서는 예측하지 못한다.
- DR의 온도의존성을 단조라고 가정하지 않는다.
- 광이온화와 비열적 전리의 실제 CMFGEN 개별 비는 미식별일 것으로 기대한다.
- O 저장 율의 관측 상태 대응과 철족 수록 여부는 미확인이다. 누락 발견을 물리적 영율로 해석하지 않는다.
- 스펙트럼 개선·수렴·생산 승격은 기대 결과로 등록하지 않는다.

해석은 전리비 차이 → 재결합 상태 치환 → 조건부 필요 총율 → 남는 미식별 채널 순서다.

동일 셸에서 Si와 적어도 한 철족 원소가 모두 적격이면 채널별 dex 부호의 공통·반대·미분리를 나란히 보고한다. 철족 세 원소를 먼저 합치거나 유리한 셸만 고르지 않는다.

같은 부호는 **공통된 열적 상태 반응과 양립**한다는 뜻이다. 반대 부호는 그 셸에서 그 채널의 단순 공통 방향 설명을 지지하지 않는다. 어느 쪽도 실제 공통 원인의 입증·배제가 아니다. 겹치는 적격 셸이 없으면 공간적으로 공통성을 시험하지 못했다고 보고한다.

## §7. 수치 처리·최소 변경집합·출력

공개 토큰의 표시 정밀도는 기존 `printed`·`I`·로그 구간 함수를 재사용한다. 비선형 계수의 구간은 입력 구간에서 평가하며 DR 극값을 끝점만으로 단정하지 않는다. 수치 구간은 원자 자료·준위 절단·공간 대응·물리 모형 오차를 포함하지 않는다.

코드는 0=음의 dex, 1=양의 dex, 4=구간이 0 포함, 5=미정의로 고정한다. 영·음수·비유한 값에 floor를 넣지 않는다. 잘못된 신원·단위·결합은 미정의 과학 행으로 숨기지 않고 비교 부적격으로 처리한다.

### 변경집합 — DET-IONRATE-BOUND (표)

| 파일 | 허용 변경 |
|---|---|
| `scripts/det_ionrate_bound_readout.py` | 신설: 지정 입력 결합, 공통 계수 평가, 표 조립, 음성 대조 |
| `/gpfs/kjhan/lumina/cmfgen_map/ionrate_20261010/tools/run_det_ionrate_bound_offline.slurm` | 신설: 직접시험·판독을 실행하는 얇은 CPU 러너 |

### 변경집합 끝

기존 판독기·생산 소스 수정은 0이다. `det_blue_ion_map_public`, `det_blue_ion_map_nearest`, `det_blue_ion_map_fraction`과 기존 구간 함수를 재사용한다. 원소 상수를 전역 변경하지 않는다.

`det_siiii_rates_readout.py`의 준위 율 예산은 이온 단계 율이 아니므로 전체 분석 경로를 호출하지 않는다. 기존 원자 계산 도구의 전역 입력 로드·다른 런 기본값·clamp도 승계하지 않는다.

출력은 `cmfgen_comparison.csv`, 같은 표와 해석의 `cmfgen_comparison.md`, 입력·시험·게이트·비용의 `evidence.json` 세 파일이다. 별도 결정 파일·보조 진단 표·저장 CSV 재독해 검사기는 만들지 않는다.

## §8. 게이트·직접시험·종료감사

| 게이트 | 막는 실재 실패 |
|---|---|
| G-DOC | 다른 계약·미등록 변경집합으로 집행하는 실패 |
| G-INPUT | 세대 14 개체수나 장부 호출 번호를 관측 세대 13과 혼동하는 실패 |
| G-PHYS | 다른 핵·단계·단위 결합, RR+DR 중복, 조건부 값을 실제 CMFGEN 율로 표기하는 실패 |
| G-TABLE | 1400행 중 불리한 행 누락·중복 또는 공개 셸 오결합 |
| G-COST | 로그인 계산·GPU 사용·등록 누적 비용 초과 |

새 코드 직접시험은 소형 정상 자료와 다음 결함을 짝지은 **음성 대조**로 제한한다.

1. 장부 `gen`을 물질 세대로 바꾸거나 분율 세대를 14로 바꾸면 거부한다.
2. II↔III를 III↔IV로 바꾸거나 공개 물리 행을 이웃 속도로 결합하면 거부한다.
3. cm³ s⁻¹ 계수를 s⁻¹로 취급하거나 \(n_e\)를 두 번 곱한 변형은 등록 수식과 불일치한다.
4. RR+DR 총계수를 순수 DR로 넣어 Milne와 합산하면 거부한다.
5. 미식별 공개 광이온화 칸에 필요 총율을 넣거나 누락 O 채널을 0으로 넣으면 거부한다.

기존 함수 시험을 복제하지 않는다. 행 검사는 저장 전 자료에 적용한다. **종료감사의 1차 대상은 과학 표의 수치·대응·주장 한정**이며 필요한 원입력 재현만 제한적으로 수행한다.

## §9. 환경·비용·재현 인터페이스

a10 CPU Slurm, 동시 1작업, **1 CPU·8 GiB·GPU 0**, 누적 할당 경과 **600초**, 누적 CPU 할당 **1/6 core·h**를 상한으로 한다. 문서 게이트·직접시험·실패 실행·판독·필요한 감사 재현을 모두 포함한다. 대기시간은 별도 기록한다.

로그인 노드에서는 조회·편집만 한다. lageunha로 우회하지 않는다. 러너는 `module purge` 후 필요한 모듈을 명시적으로 로드하고 실제 버전을 남긴다. Python은 `-B`, 수치 라이브러리는 단일 스레드로 실행한다.

```bash
IONRATE_BASE=/gpfs/kjhan/lumina/cmfgen_map/ionrate_20261010
sbatch --partition=a10 --ntasks=1 --cpus-per-task=1 \
  --mem=8G --gres=gpu:0 --time=00:05:00 \
  --job-name=LUMINA_IONRATE_BOUND --exclude=syn102,syn03 \
  "$IONRATE_BASE/tools/run_det_ionrate_bound_offline.slurm" analyze a1
```

이는 구현할 인터페이스이며 기안자가 실행하지 않았다. `analyze a1`은 음성 대조 통과 후 표를 만든다. 추가 작업 시간 요청은 잔여 누적 예산 이내로 제한한다. 일반 파일 입출력을 사용하고 기존 출력 덮어쓰기를 거부한다. 원본은 수정하지 않는다.

## §10. 완료 조건·처분·주장 상한

다음이 모두 충족되면 **조건부 비교 단에 한해 `PASS_LIMITED`**다.

- 1400행과 등급·미식별 사유가 완전하며 공개 전리비가 정의 가능한 곳에서 계산됐다.
- Fe·Co·Ni·Si 각각 적어도 한 존재 셸에서 적격 재결합 상태 치환의 수치 비교를 얻었다.
- 실제 율·공통 처방·평형 조건부 값을 분리했고 비용 상한을 지켰다.
- 어떤 채널과 공통성 질문을 답하지 못했는지 명시했다.

재결합 비교의 필수 입력 부족으로 위 수치 조건을 충족하지 못하거나 예산 안에 끝내지 못하면 `INCOMPLETE`, 잘못된 신원·물리 정의로 만든 표는 `INADMISSIBLE`이다. 미식별 칸만 채운 표를 목표 달성으로 판정하지 않는다.

**사용자가 승인한 실제 채널별 원인 분리는 이번 자료로 부분적으로만 답할 수 있다.** `PASS_LIMITED`도 그 원인 규명의 완료를 뜻하지 않는다. 광이온화 대 비열적 전리의 분리에는 공개 계산과 연결된 내부 율 또는 복사장·준위 개체수·비열적 분배 자료가 추가로 필요하다.

생산 승격·물리적 정답·인과 기여율·수렴을 선취하지 않는다. 조건부 잔차의 크기를 청색 결손 설명률로 바꾸지 않는다. 후속 계측·새 CMFGEN 런·GPU 계산은 이 계약으로 승인되지 않는다.

## §11. 분장·기안 확인 범위

기안·개정·모든 감사는 astra, 구현·직접시험·게이트·실행·판독·판정은 운전자다. GPU 0·생산 변경 0이므로 개정25-ab에 따라 경량 사전감사는 생략한다. 추가 감사가 필요하면 개정25-ac에 따라 astra가 맡는다.

Q-GOAL·Q-LEAN 자기답은 독립 감사가 아니다. 계약 저자의 종료감사 이해관계와 운전자의 구현·판정 겸임을 기록한다.

기안자는 공유 문맥·프로젝트 위치·현행 규약·직전 계약과 판정·사용자 결정·지정 입력 헤더·일부 소스와 재사용 함수를 읽었다. 원자 자료 전체의 적격성·율 장부 대응·수치 결과는 검증하지 않았다. 파일 수정·계산·작업 제출은 하지 않았다.

## §12. 기계 선언

분기는 비교 대상보다 낮음·높음·부호 미분리·미식별을 뜻한다. 조건부 등급을 실제 CMFGEN 율 비교로 승격하지 않는다. 참조에는 실재 파일만 둔다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-IONRATE-BOUND (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_ionrate_bound_readout\\.py|/gpfs/kjhan/lumina/cmfgen_map/ionrate_20261010/tools/run_det_ionrate_bound_offline\\.slurm)",
    "symbol": "det_ionrate_bound_",
    "roots": [
      "scripts/det_ionrate_bound_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/ionrate_20261010/tools/run_det_ionrate_bound_offline.slurm"
    ],
    "expected_extra": [
      "scripts/det_ionrate_bound_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/ionrate_20261010/tools/run_det_ionrate_bound_offline.slurm"
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
      {"name":"BELOW_REFERENCE","predicate":"valid and c0 == 1"},
      {"name":"ABOVE_REFERENCE","predicate":"valid and c1 == 1"},
      {"name":"UNRESOLVED","predicate":"valid and c4 == 1"},
      {"name":"NOT_IDENTIFIABLE","predicate":"valid and c5 == 1"}
    ],
    "adversarial_fixtures": [
      {"name":"below","mix":[[0,1]]},
      {"name":"above","mix":[[1,1]]},
      {"name":"unresolved","mix":[[4,1]]},
      {"name":"undefined","mix":[[5,1]]},
      {"name":"invalid","mix":[[2,1]]},
      {"name":"missing","mix":[]},
      {"name":"multiple","mix":[[0,1],[1,1]]}
    ]
  },
  "references": [
    {"path":"docs/PREREG_DET_BLUE_ION_MAP_2026-10-10.md"},
    {"path":"docs/VERDICT_DET_BLUE_ION_MAP_2026-10-10.md"},
    {"path":"docs/VERDICT_DET_COLL_PRESC_2026-10-09.md"},
    {"path":"docs/VERDICT_DET_NRB_JUDGE_2026-10-08.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/ionrate_20261010/evidence/USER_DECISION_2026-10-10.txt"},
    {"path":"scripts/det_blue_ion_map_readout.py","flags_existing":["det_blue_ion_map_public","det_blue_ion_map_nearest","det_blue_ion_map_fraction"]},
    {"path":"scripts/det_nrb_judge_readout.py","flags_existing":["det_nrb_judge_log10i"]},
    {"path":"scripts/det_siiii_pop_readout.py","flags_existing":["class I","def printed"]},
    {"path":"scripts/det_siiii_rates_readout.py"},
    {"path":"src/lumina_plasma.c","flags_existing":["DR_TABLE","dr_alpha_eval"]},
    {"path":"data/standart_data1/toy06/phys_toy06_cmfgen.txt","flags_existing":["19.480"]},
    {"path":"data/standart_data1/toy06/ionfrac_fe_toy06_cmfgen.txt"},
    {"path":"data/standart_data1/toy06/ionfrac_co_toy06_cmfgen.txt"},
    {"path":"data/standart_data1/toy06/ionfrac_ni_toy06_cmfgen.txt"},
    {"path":"data/standart_data1/toy06/ionfrac_si_toy06_cmfgen.txt"}
  ]
}
```
