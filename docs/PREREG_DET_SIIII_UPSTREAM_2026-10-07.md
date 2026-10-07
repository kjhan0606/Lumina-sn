# 사전등록 — DET-SIIII-UPSTREAM: Si III 준위 4의 율 예산과 상류 인구 차이 귀속

2026-10-07 · 기안 역할: astra · read-only · 개정 1  
운전자 파일화 대상: `docs/PREREG_DET_SIIII_UPSTREAM_2026-10-07.md`  
상태: **86d130b의 Fable 사전감사와 사용자 표적 교체 결정 반영. 결과 판독 전 개정이며 발주 시점 upstream 판독 경로 미구현·본 분석 0회. 개정 1 기계 게이트·직접시험·판정 미집행.**

## §0. Q-GOAL·Q-LEAN 자기답과 과학 질문

### Q-GOAL — 최종목적에 직접 기여하는가?

DET-SIIII-RATES는 Si III 2542.58 Å 상위 준위 5의 E′−R′ 밀도 차이를 주요 셸 19에서 `MIXED`로 판정했다. 복사장 항과 상대 인구 항은 각각 밀도 차이의 약 0.46·0.49이며, 상대 인구 항의 0.999853은 하위 준위 4에서 온다. 이는 **승계 사실**이며 이번 단에서 다시 판정하지 않는다.

이번 목표는 **준위 4 하나의 세대 7 율 예산과 밀도 차이를 분해하여, 준위 5의 MIXED 판정에서 미식별로 남은 상대 인구 차이가 준위 4의 율 계수 항으로 귀속되는지, 다시 다른 상대 준위 인구 항으로 이어지는지 판정하는 것**이다.

여기서 율 계수 항은 NRB 형태·복사 입력·충돌 계수 변화를 구별한다. 잔차 지배·혼합·수치 미분리·자료 결손도 등록된 답이다. 상대 준위가 에너지상 더 높다는 전제를 두지 않는다.

이 결과는 공개 CMFGEN 잣대가 있는 UV2500(2500–3000 Å, E′ N의 OPT 정규화 색비 +0.61 dex) 가지의 후속 조사를 결정하는 데 쓰인다(`docs/VERDICT_DET_OBS_WRITER_2026-10-02.md` §7). 이를 생략하면 준위 5의 상대 인구 항을 설명의 종점으로 오인하거나, 준위 4의 큰 E′/R′ 밀도비 자체를 원인으로 재명명하게 된다. 최종 스펙트럼 차이의 물리적 설명을 위한 진단이며, 이번 작업의 스펙트럼 잔차 감소는 0이다.

황 전선과의 연결: 09-25 1단계는 조사한 셸의 S II 광이온화에서 명시 들뜬 준위 경로가 69–82%이고 그 경로의 광자가 대부분 λ<2000 Å임을 보였으며, 2단계의 두꺼운 선 bb J̄·Λ-정체 진단과 bb 율 ALI/NRB 괄호 시험은 E′의 1206 Å 두꺼운 공명선과 연결된다. 출처는 `docs/READOUT_CAUSE1_SII_IONIZATION_CHANNEL_2026-09-25.md` §5–§6, `docs/READOUT_CAUSE2_FUV_LINE_LAMBDA_STALL_2026-09-25.md` §6–§7, `docs/VERDICT_DET_NRB_E_2026-10-02.md` §7이며, 특정 선의 독점적 인과를 승계하지 않는다.

**기안자 자기답: PASS.**

### Q-LEAN — 불필요한 계측·게이트를 추가하는가?

분석은 표적 준위 4의 주요 셸 19와 보조 셸 24·30으로 제한한다. 직접 연결된 생산 선은 21개이며, 두 상태의 등록 분석 셸에서 전부 소비·mapped이고 W5 call 7에 기록되어 있다. 표적 인구는 이 연결선들의 역산 구간 교집합으로, 상대 인구는 그 상대를 표적과 잇는 선에서 복원하는 기존 규칙을 사용한다.

- 기존 판독기를 일반화하고 러너만 신설한다.
- 새 판독기·생산 solver·복사장 replay·추가 생산 실행을 만들지 않는다.
- 새 직접시험은 대표 선 없이 지정한 준위 표적이 전이의 하위·상위에 놓이는 연결에 한정한다.
- 선행 30개 직접시험은 회귀로 유지한다. 기존 생산식·오차 기준을 새로 감사하는 별도 시험 묶음을 만들지 않는다.
- 기존 G-LINK·G-RATE 안에서 새 연결을 검증한다.
- 셸 10·15의 자료가 존재해도 분석 범위를 확장하지 않는다.
- 이번 상대 준위들의 율 예산을 다시 재귀적으로 계산하지 않는다.

**기안자 자기답: APPROPRIATE.**

위 자기답은 사전감사를 대신하지 않는다. **Fable fresh가 Q-GOAL, 이어서 Q-LEAN을 독립으로 답한다.** 호출 실패 시 사용자 지정 대체 모델 gpt-6.1-sol을 사용하며 실패 기록과 실제 모델 ID를 보존한다.

### 승인 근거와 주장 상한

사용자 결정문:

`/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/evidence/USER_DECISION_2026-10-07.txt`

둘째 결정 항목의 「후보 2로 교체」에 따라 표적은 준위 4 하나다. 후보 1(준위 6·7·8·10·11)은 보류한다.

허용하는 주장은 **세대 7 고정 상태에서 생산 정의를 적용한 율 예산과 정확한 대칭 분해의 대수적 귀속**이다. 다음은 판정하지 않는다.

- 여섯 반복 동안 축적된 밀도 차이의 독립적인 인과 기여율.
- 물리적으로 옳은 NRB 처리, 수렴성, 생산 승격 또는 수리 승인.
- 공개 CMFGEN 인구·율과의 일치.
- 특정 과정의 간접 영향 부재.
- 생산 해 재현 또는 스펙트럼 잔차 감소.

NRB는 선 자기 소스를 새 인구에 연결하는 **반복법 변경**이라는 정의를 승계한다. 고정된 전체 J̄ 행렬의 단순 항등 재작성으로 해석하지 않는다.

## §1. 경로·기준선·승계 사실

### 1.1 경로

```text
P  = /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
C  = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006
T2 = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source
E  = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Eprime/run1
R  = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1
B  = /gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007
D0 = /gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007
D  = /gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007

선행 정본 = D0/results/b2
새 러너 = D/tools/run_siiii_upstream.slurm
새 산출 = D/results/<실행 식별자>
새 집행 증거 = D/evidence/<집행 식별자>
```

### 1.2 승계 기록

- DET-SIIII-RATES 폐합: `0f0e588`.
- 직전 계약: `docs/PREREG_DET_SIIII_RATES_2026-10-07.md`.
- 직전 판정: `docs/VERDICT_DET_SIIII_RATES_2026-10-07.md`.
- 선행 종료감사:
  - `/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/astra_closeout.md`
  - `/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/astra_closeout2.md`
- 기존 판독기: `scripts/det_siiii_rates_readout.py`.
- 변경하지 않는 역산 의존 파일: `scripts/det_siiii_pop_readout.py`.
- 선행 신원 장부: `/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/results/b2/identity.sha256`.

선행 신원 장부에 기록된 판독기 SHA-256:

```text
58722696ddd76d598f59fb16b00f225299d980492191ab45fb2f5894a81134ac
```

역산 의존 파일 SHA-256:

```text
63cbea718eec73a84b21e4cf907e89934b5cbf2011026834cd50cc8e686976be
```

운전자는 수정 전 실제 파일을 위 장부와 대조하고, 수정 후 도구의 새 신원을 별도로 기록한다. 과거 결과·감사·저작 표기를 새 도구의 결과로 바꾸지 않는다.

### 1.3 재검토하지 않는 확정 사실

다음은 새 과학 판정 대상이 아닌 승계 사실이다.

1. 준위 5의 주요 셸 19 분기는 코드 4, `MIXED`다.
2. 그 상대 인구 항의 0.999853은 준위 4이며, 준위 4의 복원 밀도 E′/R′는 셸 19에서 약 168이다.
3. 준위 5의 셸 19 NRB 형태 항은 음수이고 복사장·상대 인구 항은 양수다.
4. Si NLTE 쌍은 (II, III) 하나이며 Si III는 최상위 이온이다. `TOPSTAGE_IV`는 비활성이다.
5. Si III 여기 준위의 직접 과정은 bb 복사·bb 충돌이다. 활성 충돌식은 `NLTE_COLL_FIX`, \(\Upsilon=\max(\Upsilon_{\rm vR},1)\)이다.
6. 그 밖의 직접 항은 `STRUCTURAL_ZERO`다. 간접 영향의 부재를 뜻하지 않는다.
7. 세대 7 캡처는 후속 물질 갱신의 평형해가 아니다. 직전 계약 §5.3의 생산 해 재현 불가와 판정 상한을 승계한다.
8. 선행 30개 직접시험, 생산 연산 순서에 근거한 구간 처리, 종료감사에서 수정된 허용폭·분해·분기 검증을 승계한다.

기존 설정·신원 검사는 **승계 전제가 같은 입력에 적용되는지 확인하는 연결 검사**로 유지한다. 생산 경로 전수조사나 확정 사실의 재감사를 반복하지 않는다.

### 1.4 새 표적의 자료 가용성 근거

- `/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/evidence/level4_feasibility_2026-10-07.txt`

운전자·lageunha 실측에서 E′·R′ 모두 W5 기록 셸 10·15·19·24·30의 연결선 21/21이 소비·mapped이고 W5 call 7에 존재하며, 준위 4와 바닥 0의 복원이 가능하다. 셸 가용성은 이 증거를 인용하고 새 산출로 재기록하지 않는다. **역산 적격성·표적 인구 양립성·율 예산 완전성의 통과 결과는 아니다.**

### 1.5 입력 권위와 보존

E·R 각각에서 다음 입력을 사용한다.

| 입력 | 역할 |
|---|---|
| `work/det_obs_capture/lines.bin` | 소비·mapped 선의 세대 7 τ·S_used |
| 같은 디렉터리의 `manifest.json`, `meta.txt` | 캡처 세대·형식·시점·권위 |
| 같은 디렉터리의 `Te.f64`, `ne.f64` | 세대 7 충돌 계수·super-level 검사 |
| `work/w5_lines.csv` | 해당 선·셸의 call 7, rad 7, material 7 복사 입력 |
| `input/model/line_list.csv`, `levels.csv` | 전이 행·준위·에너지·통계중량·Einstein 계수 |
| `input/deck.sha256`, `input/resolved_lumina.exports`, `input/run_build_identity.json` | 봉인·설정·생산 신원 |

| D0 정본의 분석 산출 | 준위 5 셸 19의 확정 귀속과 준위 4를 연결하는 선행 근거; §6.4의 경로·SHA-256 인용 |

생산식의 권위는 T2다. 현재 작업 트리의 생산 소스로 바꾸지 않는다.

대형 캡처의 기존 봉인 신원을 승계하고 새로 선택한 레코드의 출처를 기록한다. 전체 복제·전체 CSV 변환을 하지 않는다. 새 결과 경로가 이미 있으면 거부하고 새 실행 식별자를 사용한다. 기존 결과와 실패 기록은 보존한다. 특정 파일시스템의 스냅숏·reflink·확장 속성 등을 전제로 하지 않는다.

## §2. 표적·셸·생산 범위

### 2.1 준위 표적

표적은 Z=14, ion=2, 준위 ID=4 하나로 지정한다. 대표 방출선 행을 표적의 필수 식별자로 요구하지 않는다.

| 준위 | 명칭 | g | 에너지, eV | 연결선 | 자기 자신을 제외한 상대 준위 | 그중 준위 ≥100 |
|---:|---|---:|---:|---:|---:|---:|
| 4 | 3s3p ¹P₁ | 3 | 10.2763571337 | 21 | 21 | 8 |

실측 기록의 상대 목록 22개는 표적 자신을 포함한 끝점 집합이다. 예산은 자기 자신을 제외한 21개 상대에 대해 합한다. 연결선은 모두 bb domain 안(약 565–2543 Å)이며, 바닥 0과 잇는 1206.50 Å(0→4, f=1.70)와 1206.56 Å(4→15, f≈1.72)는 서로 다른 전이다.

전이 ID는 CSV 헤더를 제외한 **0-based 데이터 행**이다. 별도 `line_id` 열로 치환하지 않는다. 표적이 전이의 하위 준위인 선도 전부 포함한다.

### 2.2 셸의 역할

| 셸 | 등록 역할 | 결론에서의 사용 |
|---|---|---|
| 19 | 주요 셸 | 2542 Å 가지의 선행 주요 셸 17–22 중 W5 기록 셸이며, 준위 5의 MIXED 판정을 이어받는 주 판정 |
| 24·30 | 보조 셸 | 2542 Å 가지의 선행 주요 셸 밖에서 동일 분해 비교; 주요 셸 대체 불가 |

주요 판정 단위는 **준위 4×셸 19**다. 셸 24·30은 각각 보조 결과로 남기며 평균·다수결로 주요 판정과 묶지 않는다. 다른 셸의 분기가 더 분명하다는 이유로 주 결론을 옮기지 않는다. 셸 17–30의 기존 W5 결손 범위는 직전 판정, 셸 10·15의 가용성은 §1.4 증거를 인용하며 이번 러너에서 재기록·보간·분석하지 않는다.

### 2.3 직접 생산 과정

준위 4는 cutoff 100 미만의 개별 solve 준위다. 직접 예산은 bb 복사와 bb 충돌만 계산한다.

직접 광이온화·복사 재결합·충돌 이온화 및 삼체 재결합·DR·비열적 이온화·전하교환은 승계 근거에 따라 `STRUCTURAL_ZERO`로 기록한다. 이를 위한 새 적분기나 반응 계산기를 구현하지 않는다.

근거:

`/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/evidence/production_paths_grep_2026-10-07.txt`

준위 ≥100의 상대 준위는 실제 super-level 100의 구성원으로 취급한다. full-level 전이 계수와 구성원 인구를 사용한 예산에 투영 분율을 다시 곱하지 않는다.

### 2.4 선행 준위로 되돌아오는 연결

표적은 준위 4 하나이며 자기 연결은 예산에 넣지 않는다. 서로 다른 표적 사이의 공유 전이는 0건이다.

준위 5는 2542.58 Å(4→5)로 연결된 실제 상대다. 이번 귀속이 준위 5의 인구 항을 가리키면 §6의 `RETURN_TO_LEVEL_5`로 보고하며, 독립된 새 원인으로 세지 않는다.

## §3. 세대 7 표적·상대 인구 복원

### 3.1 복원식과 적격성의 승계

\[
a=\frac{g_l}{g_u},\qquad
A_\nu=\frac{2h\nu^3}{c^2},\qquad
K=C_{\rm Sob}f_{lu}\lambda t_{\rm exp},
\]

\[
d=\frac{\tau}{K},\qquad
n_l=d\left(1+\frac{S}{A_\nu}\right),\qquad
n_u=\frac{g_u}{g_l}d\frac{S}{A_\nu}.
\]

생산 상수·연산 순서·저장 정밀도와 역산 적격성은 직전 계약·수리된 판독기를 승계한다. source fallback·변형, 비유한 값, 분모 문제, 세대·권위 불일치를 임의 허용폭으로 통과시키지 않는다.

### 3.2 표적 연결선에서의 인구 복원

표적과 필요한 상대 준위 집합은 다음으로 고정한다.

\[
L=\{4\}\cup\{j:\text{준위 4와 생산 선으로 연결됨}\},
\quad U=\{4\}.
\]

복원에는 같은 런·같은 셸·같은 캡처 세대의 **준위 4 연결선 21개**만 사용한다. 표적 인구는 연결선들의 적격 역산 구간 교집합으로, 각 상대 인구는 그 상대를 표적과 잇는 선의 역산 구간으로 얻는다. 이는 기존 판독기의 표적·상대 인구 규칙이며, 다른 소비선으로 복원 범위를 넓히지 않는다.

동일 입력을 선택 판독한 결과를 재사용하며, 표적마다 캡처를 다시 스캔하지 않는다.

### 3.3 표적 인구 구간의 교집합

런 \(X\), 셸 \(s\)마다 표적 연결선에서 얻은 준위 4의 적격 구간을 모은다. \(\mathcal E^X_{4,s}\)는 이 연결선 중 역산이 적격한 선의 집합이다.

\[
\mathcal I^X_{4,s}
=
\bigcap_{r\in\mathcal E^X_{4,s}}
[n^{X,r}_{4,\mathrm{lo}},n^{X,r}_{4,\mathrm{hi}}].
\]

규칙은 다음과 같다.

1. 연결선별 행 ID·끝점 방향·세대·적격성·원 구간을 보존한다.
2. 연결선에서 얻은 표적의 적격 구간을 모두 교집합에 넣는다. 먼저 성공한 선만 고르지 않는다.
3. 비어 있지 않은 교집합을 해당 상태·셸의 유일한 표적 인구 구간으로 사용한다.
4. 표적 인구의 모든 사용처는 같은 교집합을 사용하고, 상대 인구는 해당 연결선의 역산 구간을 사용한다.
5. 대표 선을 지정하지 않고, 교집합 내부의 결정론적 대표값을 정한다. 대표값 선정은 오차폭 확대나 물리 인구의 clamp가 아니다.
6. 선 하나만 적격하면 그 구간을 사용하고 단일선 복원임을 기록한다. 다른 필수 연결선의 결손은 그대로 남긴다.
7. 부적격 선은 기존 생산·역산 기준에 따라 사유를 남긴다. 결과에 맞추어 제외하지 않는다.

표적의 적격 구간 교집합이 비면 `POPULATION_INCONSISTENT`와 해당 예산 코드 6을 기록한다. 평균·중앙값·임의 허용폭·원하는 부분집합으로 덮지 않으며, 충돌한 선과 구간을 남긴다.

표적의 적격 구간이 없거나 필수 상대 인구가 해당 연결선에서 복원되지 않으면 기존 `MISSING_G7_POPULATION` 처리를 따른다. 다른 선·세대의 인구나 투영 유도로 결손·불일치를 우회하지 않는다.

### 3.4 연결선 예산의 완전성

§1.4 실측에 따라 등록 분석 셸의 21개 연결선 전부를 예산에 요구한다. 실측의 존재 확인을 역산·W5 적격성 검사 대신 사용하지 않는다.

전체 예산에는 각 연결선의 양 끝 인구, 원자 계수, 그 선 자신의 W5 입력 및 실제 생산 분기가 모두 적격해야 한다. 하나라도 필요한 입력이 결손이면 기존 규칙대로 부분 예산과 코드 6을 남긴다.

### 3.5 Super-level 검사와 세대 8 배제

기존 super-level Boltzmann 양립성 검사를 필요한 구성원에 적용한다. 구성원 인구는 §3.2의 직접 연결선 역산을 사용하며 별도 super-level 사이를 LTE 비로 채우지 않는다.

이번 upstream 경로에서는 세대 8을 읽거나 보조 비교를 산출하지 않는다. 결손·불일치 세대 7 인구를 세대 8로 채우지 않는 기존 음성대조는 §8.1의 회귀로 유지한다.

## §4. 생산 정의에 따른 율

### 4.1 예산

\[
I^X_{u,p}=\sum_{j\ne u}n^X_jP^X_{ju,p},\qquad
D^X_{u,p}=\sum_{j\ne u}P^X_{uj,p},
\]

\[
O^X_{u,p}=n^X_uD^X_{u,p},\qquad
F^X_{u,p}=I^X_{u,p}-O^X_{u,p}.
\]

\[
I_u=\sum_pI_{u,p},\quad
D_u=\sum_pD_{u,p},\quad
O_u=n_uD_u,\quad
\mathcal R_u=I_u-O_u.
\]

\(P,D\)의 단위는 \({\rm s^{-1}}\), 인구는 \({\rm cm^{-3}}\), 체적당 율은 \({\rm cm^{-3}\,s^{-1}}\)다. 양의 순율은 표적 인구를 늘리는 방향이다.

이번 표적에는 별도 직접 외부 source 항이 없다. 구조적 0을 구현하기 위한 계산 경로를 추가하지 않는다.

### 4.2 복사 계수

\[
P^{\rm off}_{lu}=B_{lu}\bar J,\qquad
P^{\rm off}_{ul}=A_{ul}+B_{ul}\bar J,
\]

\[
P^{\rm on}_{lu}=B_{lu}ct,\qquad
P^{\rm on}_{ul}=\beta A_{ul}+B_{ul}ct,\qquad
ct=\beta J_c.
\]

생산의 `srce_chk`·operator·provenance·domain 조건과 β 계산을 승계한다. E′ 실제 off, R′ 실제 armed 분기를 쓰되, NRB 비적용 선은 해당 생산 off 분기를 유지한다.

W5의 call 7, rad 7, material 7, operator, 선·셸·이온·주파수와 모순 중복을 검사한다. 기존 β 및 J̄ 원장 일치 검사도 유지한다.

A·B는 봉인 원자 입력, J̄·J_c·β·τ_eff와 관련 오차는 **해당 율 선의 W5**가 권위다. coarse J 보간이나 다른 선의 복사 입력으로 결손을 메우지 않는다.

### 4.3 충돌 계수

\[
C_{lu}
=n_e\frac{8.629\times10^{-6}}{g_l\sqrt{T_e}}
\Upsilon e^{-\Delta E/kT_e},
\qquad
C_{ul}
=n_e\frac{8.629\times10^{-6}}{g_u\sqrt{T_e}}\Upsilon,
\]

\[
\Upsilon=
\max\left[
14.5f_{lu}
\left(\frac{13.6057}{\Delta E_{\rm eV}}\right)^2 0.2,\;1
\right].
\]

활성 충돌강도 하한과 비활성 `coll_floor`를 혼동하지 않는다. 별도 충돌표를 새로 적용하거나 생산 하한을 수리하지 않는다.

### 4.4 결손과 부분 예산

계수가 알려져도 인구가 결손이면 체적당 유입·유출은 결손일 수 있다. 가능한 계수와 부분율은 남기되 부분합을 총예산으로 표시하지 않는다.

`STRUCTURAL_ZERO`, 활성 경로의 계산된 0, `MISSING`을 구별한다.

## §5. 폐합과 수치 기준

### 5.1 오프라인 산술 폐합

- 전이별 합과 과정별·표적별 예산이 일치해야 한다.
- 표적 인구는 모든 항에서 같은 교집합을 사용하고, 상대 인구는 해당 연결선에서 복원한 구간을 사용해야 한다.
- 한 표적의 행 안에서 전이를 중복 합산하지 않는다.

- 상대별 인구 항의 합과 §6의 전체 상대 인구 항이 일치해야 한다.
- 밀도 차이 분해는 잔차를 포함하여 폐합해야 한다.
- super-level 구성원 인구에 투영 분율을 중복 적용하지 않는다.

이는 도구의 산술 일관성 검사이며 생산 해 재현의 증거가 아니다.

### 5.2 고정 상태의 평형 잔차

완전한 예산에서

\[
\epsilon_u=\frac{|\mathcal R_u|}{I_u+O_u}
\]

를 보고한다. 분모 0은 별도 상태로 기록한다. 잔차가 0이 아니라고 판독기 오류로 단정하거나, 분해를 닫기 위해 잔차를 제거하지 않는다.

### 5.3 생산 해 재현 불가의 승계

직전 계약 §5.3의 `PRODUCTION_SOLUTION_NOT_TESTABLE`을 그대로 승계한다.

세대 7 캡처는 후속 R7 물질 갱신 이전이고, 후보 전자밀도 폐합·CE 경계 인구·감쇠 이력이 저장되지 않는다. 최종 저장 인구도 특정 단일 Si 선형 solve의 직접 출력으로 연결할 수 없다.

이번에는 이 조건을 재조사하거나 생산 해·동결 세대 7 해를 새로 풀지 않는다. 준위 4의 율 예산을 구해도 이 제한은 해소되지 않는다.

### 5.4 오차 처리

직전 계약과 종료감사 수리판의 구간 처리를 승계한다.

- binary64 및 텍스트 writer 정밀도·생산 연산의 반올림을 반영한다.
- W5의 `Jbar_err_upper`, `J_c_err_upper`를 전파한다.
- 상쇄가 있는 항에는 절대오차를 유지한다.
- 새 경험적 상대 허용폭을 넣지 않는다.
- 공통 인구 구간의 교집합을 구한다고 입력 오차 자체가 독립적으로 줄었다고 주장하지 않는다.
- 결손을 넓은 오차막대로 대신하지 않는다.

## §6. 귀속·분기·사슬 보고

### 6.1 정확한 대칭 분해

\[
\Delta X=X^{E'}-X^{R'},\qquad
\bar X=\frac{X^{E'}+X^{R'}}2.
\]

완전한 예산과 양의 \(\bar D_u\)에서

\[
\Delta n_u
=\sum_p
\frac{\Delta I_{u,p}-\bar n_u\Delta D_{u,p}}{\bar D_u}
-\frac{\Delta\mathcal R_u}{\bar D_u}.
\]

또한

\[
\Delta(n_jP_{ju,p})
=\bar n_j\Delta P_{ju,p}
+\bar P_{ju,p}\Delta n_j.
\]

따라서 상대 준위별 인구 항은

\[
B_{u\leftarrow j}
=
\frac{\sum_p\bar P_{ju,p}\Delta n_j}{\bar D_u},
\qquad
A_{u,\rm population}=\sum_{j\ne u}B_{u\leftarrow j}.
\]

계수 항에는 유입 계수 변화와 표적의 유출 계수 변화가 함께 들어간다. 모든 밀도 귀속 항의 단위는 \({\rm cm^{-3}}\)다.

### 6.2 NRB 형태·복사 입력·충돌 분리

같은 상태에서 on·off 양쪽이 적격한 경우 직전 대칭 분해를 사용한다.

\[
\Delta P_{\rm form}
=\frac{[P_{\rm off}(E)-P_{\rm on}(E)]
+[P_{\rm off}(R)-P_{\rm on}(R)]}{2},
\]

\[
\Delta P_{\rm field}
=\frac{[P_{\rm off}(E)-P_{\rm off}(R)]
+[P_{\rm on}(E)-P_{\rm on}(R)]}{2}.
\]

실제 E=off, R=on인 선에서는 두 항의 합이 실제 계수 차이다. 양쪽 비적용 선은 형태 차이가 없는 생산 분기를 따른다. 필요한 가상 형태 입력이 없으면 `NRB_FORM_NOT_IDENTIFIABLE`로 남긴다.

`field`는 J̄·J_c·β 등 저장된 복사 관련 입력의 상태 차이를 뜻한다. 충돌 계수 변화는 별도 항이다. 상대 인구 변화는 복사 또는 충돌 계수 자체의 변화로 재명명하지 않는다.

### 6.3 지배 기준과 코드

비교 항은 직전과 동일하다.

1. NRB 형태 계수 항.
2. 나머지 복사 입력 계수 항.
3. 충돌 계수 항.
4. 직접 이온화 등 항 — 이번에는 구조적 0.
5. 상대 준위 인구 항.
6. 비평형 잔차 항.

\[
|A_k|_{\rm lo}>
\sum_{m\ne k}|A_m|_{\rm hi}
\]

이고 방향이 분리될 때만 지배로 판정한다. 계수 항들을 사후 합쳐 새 지배 기준을 만들지 않는다.

| 코드 | 분기 | 이번 질문에 대한 의미 |
|---:|---|---|
| 0 | `NRB_FORM_DOMINANT` | 이 표적의 국소 대수 분해에서 NRB 형태 계수 항 지배 |
| 1 | `OTHER_RATE_DOMINANT` | 복사 입력 또는 충돌 계수 항 지배; 항 이름 필수 |
| 2 | `PARTNER_POPULATION_DOMINANT` | 상대 인구 항으로 귀속이 이어짐; 상대별 구성 보고 |
| 3 | `RESIDUAL_DOMINANT` | 고정 상태 비평형 잔차 지배 |
| 4 | `MIXED` | 완전한 예산에서 단일 지배항이 없음을 구별함 |
| 5 | `UNRESOLVED` | 유효 예산의 수치 구간으로 방향·지배 경계가 미분리 |
| 6 | `NOT_IDENTIFIABLE` | 필요한 인구·계수·권위 결손 또는 인구 불일치 |

분리된 반대 부호 항이 있으면 `CANCELLING`을 덧붙인다. 유효 입력의 결손과 구현 오류는 구별한다. 구현 때문에 산술 폐합이 실패하면 처분은 `REWORK`이며, 코드 6 출력만으로 정상 완료하지 않는다.

### 6.4 사슬의 고정 보고 형식

사슬은 **한 단계 확장한 표와 연결 목록**으로 보고한다.

| 층 | 필수 내용 |
|---|---|
| 선행 연결 | 준위 5 셸 19의 MIXED와 상대 준위 4를 식별하는 D0/results/b2/analysis의 경로·SHA-256·행 키; 선행 B 값은 복사하지 않음 |
| 이번 표적 | \(u\), 셸, \(\Delta n_u\), 분해 항·구간, 코드·지배 항·잔차·완전성 |
| 이번 상대 연결 | 각 \(j\)에 대한 \(B_{u\leftarrow j}\), 부호·구간·인구 출처·에너지 관계 |
| 연결의 성격 | 상대가 준위 5이면 `RETURN_TO_LEVEL_5`, 나머지는 `OUTSIDE_TARGET_SET` |

모든 상대별 항을 기계 산출에 남긴다. 본문에서 일부 큰 항만 제시하면 나머지의 합과 전체표를 함께 연결한다.

선행 인용은 D0/results/b2/analysis의 `budget.csv`, `rates.csv`, `populations.csv`, `result.json`, `manifest.json`을 경로·SHA-256으로 연결하고, target=SiIII_2542·shell=19 및 상대 준위 4·전이 행 1097363을 식별한다. 선행 B 수치의 복사·재계산은 하지 않는다. 셸 24·30의 이번 행에는 선행 주요 셸 연결이 없음을 명시하고 셸 19의 인용을 같은 셸의 증거처럼 붙이지 않는다.

표현 규칙:

- 코드 0·1은 **“이번 표적의 등록된 국소 분해에서는 계수 항 지배”**로 쓴다. 인과 사슬의 최종 원인을 찾았다고 쓰지 않는다.
- 코드 2는 **“상대 인구 항으로 이어짐”**으로 쓴다. 그 상대의 인구 원인은 이번 범위 밖이다.
- 준위 5로 되돌아오면 `RETURN_TO_LEVEL_5`로 표시한다.
- 그 밖은 `OUTSIDE_TARGET_SET`으로 표시한다. 에너지가 낮은 상대도 있을 수 있다.

- 순환 연결은 순환으로 남긴다. 재귀 대입·무한 사슬 확장·새 연립해 계산을 하지 않는다.

선행 \(B_{5\leftarrow4}\)와 이번 \(B_{4\leftarrow j}\)는 서로 다른 표적의 분모를 가진다. **몫을 곱하여 준위 5의 인과 기여율이나 캐스케이드 확률로 만들지 않는다.** 보조 비율을 표시할 때는 대수적 항의 구성비라는 분모와 부호를 명시하고, 분모가 수치적으로 0과 분리되지 않으면 표시하지 않는다.

셸 19의 사슬이 주요 결과이고 셸 24·30은 보조 결과다. 어느 셸의 결론도 다른 셸이나 전체 반복 이력으로 확장하지 않는다.

## §7. 최소 변경집합·산출

### 변경집합 — DET-SIIII-UPSTREAM (표)

**기존 판독기 수정으로 구현 가능하다.** 율·분해·구간·분기 계산과 표적·상대 인구 복원 규칙을 재사용하고, 대표 선 없이 준위를 표적으로 지정하도록 일반화한다. 새 판독기 파일은 만들지 않는다.

| 경로 | 함수·항목 | 허용 변경 |
|---|---|---|
| `scripts/det_siiii_rates_readout.py` | `det_siiii_rates_*` | 준위 4 표적 설정, 기존 표적 교집합·연결선 상대 인구 규칙과 예산 연결, 상대별 사슬·선행 근거 인용, 준위 표적 직접시험·회귀·manifest; upstream의 세대 8 비교·가용성 재기록·closure 상수 칸 생략 |
| `/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/tools/run_siiii_upstream.slurm` | 오프라인 러너 — 신설 | 환경·신원·자원 기록, 기존 회귀와 새 연결시험, 등록 분석·게이트 실행, stdout·stderr·rc 보존 |

### 변경집합 끝

신규·변경 분석 함수의 접두는 `det_siiii_rates_`로 제한한다. 기존 `cmd_analyze`, `cmd_selftest`, `main`의 변경이 필요하면 접두 함수로 이동·개명하고 필요한 호출 연결만 바꾼다. 모듈 상수·CLI·문서문자열·import 변경은 이 일반화에 직접 필요한 범위만 허용한다.

기존 수치 보조 함수와 역산 의존 파일, 생산 소스·바이너리·원자자료·봉인 입력·선행 러너·선행 산출은 변경하지 않는다. 기계 게이트 도구도 변경집합 밖이다.

기존 선 표적 CLI와 30개 시험의 사용 경로를 유지한다. `TARGETS`의 `row`·`lo`는 기존 선 표적 전용으로 유지하고 upstream의 준위 4는 `--target-set upstream`으로만 선택한다. 선택을 생략한 기존 실행의 기본 표적이 유지되는지 기존 30개 회귀의 실제 분석 경로와 산출로 확인한다. 출력 manifest는 실제 선택한 계약·표적 집합·셸 역할을 기록한다.

| 필수 산출 | 내용 |
|---|---|
| `coverage.csv` | 등록 연결선의 캡처·W5 적격성, 인구 출처, 결손·구조적 0, 셸 역할 |
| `populations.csv` | 연결선별 표적·상대 인구 구간·적격성·사유와 상태·셸별 표적 교집합 |
| `rates.csv` | 전이별 실제·가상 계수, 표적·상대 인구, 유입·유출·순율·구간·생산 분기 |
| `budget.csv` | 표적·셸별 전체 예산·잔차·밀도 차이 분해·코드 |
| `chain.csv` | §6.4의 이번 상대별 항·연결 성격·인구 출처와 선행 D0/b2 경로·SHA-256 인용; 선행 B 수치 복사 없음 |
| `closure.json` | 산술 폐합·표적 인구 양립성·세대 7 잔차 |
| `checks.json` | G-LINK·G-RATE 집계와 시험 증거 연결 |
| `selftest.json` | 기존 30개 회귀와 새 연결시험의 구별·결과·도구 신원 |
| `result.json` | 주요·보조 셸별 결론, 질문의 해소·미해소, 주장 상한 |
| `manifest.json` | 계약·선행 근거·입력·도구·명령·환경·작업·산출 신원 |

위 산출은 새 실행 디렉터리에 신설한다. 미정의 수치는 `null` 또는 빈 수치칸과 사유로 남긴다. 판정문과 종료감사는 계산 산출·비용·분장·이탈을 함께 연결한다.

## §8. 게이트·직접시험

| 게이트 | 새로 막는 실패 | 통과 조건 |
|---|---|---|
| G-LINK | 대표 선 필수 가정 잔존, 표적·상대 식별이나 상태·셸·세대 혼합, 인구 불일치·결손 은폐 | §2–§3의 준위 식별·표적 교집합·연결선 상대 인구 규칙과 결손 처리가 실제 분석 출력까지 적용됨 |
| G-RATE | 준위 표적 연결에서 생긴 방향·중복 오류, 상대별 합 누락, 사슬의 중복·과장 | 기존 율·분해·분기 회귀 유지, 새 준위 연결의 독립 정답·항 합산·출력 의미 일치 |

### 8.1 기존 시험의 승계와 회귀

선행 수리판의 **30개 사례를 이름·의미·판정 기준과 함께 유지하여 수정 도구로 다시 실행**한다. 이는 변경으로 인한 회귀 검사이며 선행 과학 판정의 재감사가 아니다.

기존 시험의 엄밀한 정답 구간 포함 기준, 임의 허용폭 금지, 실제 분석 경로 검증을 유지한다. 과거 PASS 로그만 연결하여 이번 도구의 회귀 실행을 대신하지 않는다.

### 8.2 새 연결시험

생산 레이아웃의 작은 가상 입력에서 실제 `det_siiii_rates_analyze`와 산출 경로를 호출한다.

1. **준위 표적**
   - 대표 선 `row`·`lo` 없이 준위 ID로 예산을 계산한다.
   - 표적이 전이의 하위·상위에 각각 놓이는 경우를 포함한다.
   - 기존 규칙으로 얻은 표적 교집합·연결선 상대 인구를 사용하고, 두 방향의 유입·유출·예산·분해 및 상대별 항의 합이 독립 정답 구간을 포함하는지 실제 산출까지 확인한다.

각 사례는 주입 위치·독립 정답 또는 기대 사유·실제 사유·rc를 보존한다. 시험 수 자체를 목표로 추가 배터리를 만들지 않는다.

### 8.3 기계 프리플라이트

기존 도구를 그대로 사용한다.

- `scripts/check_prereg_preflight.py`
- `scripts/check_gate_referents.py`
- `scripts/check_doc_canonical.py`
- `scripts/check_dispatch_preflight.py`

검사한 문서·발주서 신원과 각 stdout·stderr·rc를 보존한다. 발주 단계에서는 dispatch 내부의 prereg 재호출 결과를 그 단계의 prereg 증거로 사용하며 같은 문서 신원에 대해 별도 중복 호출하지 않는다. 기계 PASS를 과학 감사나 수치 검증으로 표시하지 않는다.

## §9. 환경·비용·재현 명령

### 9.1 환경

등록 환경은 직전 단과 같은 **a10 CPU 전용 Slurm, GPU 예약 0**이다. 수동 실행 장소는 lageunha다. 로그인 노드에서는 파일 조회·소스 판독만 하며 직접시험·기계 게이트·수치 분석은 실행하지 않는다.

운전자가 현행 제출 규칙과 등록 자원을 대조해 러너를 집행한다. 실제 적용 환경·module·Python·numpy·pandas·스레드 수·노드·작업 ID를 기록한다. lageunha에서 시간 측정이 필요하면 Bash `time`을 사용하고 `/usr/bin/time`의 존재를 가정하지 않는다.

### 9.2 비용 상한

| 항목 | 상한 |
|---|---:|
| 동시 작업 | 1개 |
| 작업당 CPU | 4개 |
| 작업당 메모리 | 16 GiB |
| 시험·기계 게이트·분석·수정 재판독 누적 경과 | 60분 |
| 누적 CPU 예약량; 수동 실행은 같은 기준으로 산정 | 4 core·h |
| GPU 예약·생산 실행 | 0 GPU·h · 0회 |
| 새 도구·시험·산출·로그 합계 | 256 MiB |

직전 실제 지출 약 9.4분·0.63 core·h·18 MiB는 참고치이며 이번 성능 보증이 아니다. 실패·재판독도 상한에 포함하고 자동 재제출하지 않는다.

상한 안에서 특정된 판독기 결함을 수정할 수 있다. 새 물리 범위·생산 실행·비용 확대는 별도 사용자 결정 사항이다.

### 9.3 계획 CLI

다음은 운전자가 구현할 인터페이스다. upstream 선택이 없는 기존 사용 경로는 유지한다.

```bash
python3 -B "$P/scripts/det_siiii_rates_readout.py" selftest \
  --production-source "$T2" \
  --out "$D/results/$RUN_ID/selftest"

python3 -B "$P/scripts/det_siiii_rates_readout.py" analyze \
  --target-set upstream \
  --e-run "$E" \
  --r-run "$R" \
  --production-source "$T2" \

  --prior-rates "$D0/results/b2/analysis" \
  --selftest "$D/results/$RUN_ID/selftest" \
  --out "$D/results/$RUN_ID/analysis"
```

upstream CLI에서는 `--prior-pop`을 사용하지 않으며 필수 인자로 요구하지 않는다. 기존 선 표적 CLI의 호환 경로는 유지한다. `--prior-rates`는 §6.4의 선행 준위 5 셸 19 근거를 경로·SHA-256으로 인용하는 데 사용한다. 이번 인구는 E·R 캡처에서 복원한다.

upstream의 표적·주요 셸·보조 셸은 본 계약대로 고정하고 결과를 보고 바꾸지 않는다.

기계 게이트 계획 명령(개정 문서만 검사할 때는 prereg 단독 호출, 발주 단계에서는 §8.3에 따라 dispatch 내부 호출로 갈음):

```bash
python3 -B "$P/scripts/check_prereg_preflight.py" \
  "$P/docs/PREREG_DET_SIIII_UPSTREAM_2026-10-07.md" --root "$P"

python3 -B "$P/scripts/check_gate_referents.py" \
  "$P/docs/PREREG_DET_SIIII_UPSTREAM_2026-10-07.md" --root "$P"

python3 -B "$P/scripts/check_doc_canonical.py" \
  "$P/docs/PREREG_DET_SIIII_UPSTREAM_2026-10-07.md"

python3 -B "$P/scripts/check_dispatch_preflight.py" "$ORDER" --root "$P"
```

`ORDER`는 운전자의 실제 발주서다. 실행 시 정확한 argv와 출력 경로를 기록한다. 재현은 원 산출을 덮지 않는 새 실행 식별자로 수행한다.

## §10. 처분과 완료 조건

### 10.1 필수 완료 내용

판정문 첫 문단에는 **스펙트럼 잔차 감소 0**, **이번 가지에 공개 CMFGEN 잣대가 있음(UV2500)**, **준위 5 셸 19의 MIXED 판정을 이어받아 그 상대 준위 4를 진단함**을 명시한다. 공개 인구·율 잣대가 있다는 뜻으로 확대하지 않는다.

- 준위 4의 주요 셸 19와 보조 셸 24·30에 대한 자료 연결·표적 교집합·연결선 상대 인구·율 예산.
- 과정별 율·잔차·대칭 분해·상대별 인구 항.
- 계수 항 지배인지, 상대 인구 항으로 이어지는지, 또는 다른 등록 분기인지에 대한 표적·셸별 답.
- 선행 준위 5 셸 19 근거를 인용한 한 단계 사슬 및 `RETURN_TO_LEVEL_5`·`OUTSIDE_TARGET_SET`의 명시.
- 기존 30개 회귀와 새 연결시험, 기계 게이트 원로그.
- 재현 가능한 신원·명령·비용·보존·분장·이탈 기록.

### 10.2 처분

| 처분 | 기준 |
|---|---|
| `PASS_LIMITED` | 구현·시험·필수 산출이 적격하게 완료됨. 생산 해 폐합의 선행 미달을 반드시 보존하고, 추가 인구·율 결손이 있으면 그로 인해 미해소인 주요 표적·셸과 과학 질문을 첫 문단에 명시 |
| `REWORK` | 계약과 다른 복원·연결·계수·오차·분해·분기·시험 구현, 불일치 은폐, 연결선 누락, 사슬 주장 과장 |
| `BLOCKED` | 핵심 입력·권위·자원 등의 실제 문제로 등록된 최소 산출을 완료하지 못함 |

직전 계약의 생산 해 재현 불가와 완료 기준을 승계하므로 **전체 처분 상한은 `PASS_LIMITED`**다. 이번에 질문을 좁혔다는 이유로 선행 생산 해 폐합의 미달을 삭제하여 `PASS`로 높이지 않는다.

대수적 질문의 달성 범위는 처분과 별도로 명시한다.

- 준위 4의 주요 셸 19에서 완전한 유효 예산과 등록 분기가 나오면, **“등록 주요 셸의 상류 대수 귀속 완료; 생산 해 폐합 미달”**로 쓴다.
- 주요 셸에 코드 6이 남으면, 해당 표적·셸의 상류 귀속은 미달이다.
- 코드 4·5는 완전한 유효 예산에 근거할 때 정당한 혼합·수치 미분리 답이다.
- 결손 목록 작성만으로 상류 귀속 목표 전체를 달성했다고 쓰지 않는다.
- 보조 셸 24·30의 성공으로 주요 셸 19의 결손을 대체하지 않는다.

주요 셸의 결과별 후속 기본 경로는 다음과 같다. 코드 0·1이면 1206 Å J̄·NRB 괄호 기전 시험을 검토하며, 코드 1은 실제 지배 항 이름을 함께 전달한다. 코드 2에서 `RETURN_TO_LEVEL_5`로 이어지면 동결 J̄ 연립 교환 해를 설계할지 사용자가 결정한다. 코드 2에서 `OUTSIDE_TARGET_SET`으로 이어지면 준위 사슬의 반복 확장을 중단하고 연립 진단 해 또는 다른 표적을 사용자에게 제시한다. 연결의 존재만으로 후속 경로를 고르지 않고 상대별 항·부호·구간을 함께 보고하며, 두 성격이 함께 기여하면 단일 경로로 강제하지 않는다. 코드 3–6은 각각 등록된 비평형·혼합·수치 미분리·결손을 보고하고 같은 방식의 준위 추적을 자동 반복하지 않는다. **후속 물리 결정은 사용자에게 있다.** 생산 수리·후속 준위 선택·새 캡처·반복 연장은 이번 처분으로 자동 승인되지 않는다.

## §11. 집행 기록·분장

| 업무 | 담당 | 한계·규칙 |
|---|---|---|
| 기안·개정·종료감사 | astra | 계약 저자가 이행을 감사하는 한계를 명시; 실제 호출 모델 ID 기록 |
| Q-GOAL·Q-LEAN 사전감사 | Fable fresh | 독립 자기답; 호출 실패 시 gpt-6.1-sol, 실패 증거·실제 ID 보존 |
| 구현·직접시험·기계 게이트·실행·판독·판정 | 운전자 | 구현자·판정자 겸임 명시 |
| 코드 감사 | 해제 — 개정25-z 승계 | 별도 정례 코드 감사 추가 없음 |
| 범위·비용 확대·후속 물리 결정 | 사용자 | 현재 승인 밖 결정 |

| 집행 항목 | 현재 상태·증거 |
|---|---|
| 사용자 결정 | §0의 D 사용자 결정문 둘째 항목: 「후보 2로 교체」, 표적 준위 4 |
| 선행 폐합 | `0f0e588`, D0 정본 b2, `PASS_LIMITED` 승계 |
| 새 표적 가용성 실측 | 운전자·lageunha; §1.4 증거 |
| 본문 기안 | read-only; 공유 컨텍스트·계약·판정·실측 기록·판독기·관련 게이트 조회 |
| 표적·연결 확인 | 준위 4 하나, 연결선 21개; 상대 목록 22개에는 표적 자신이 포함됨. §1.4 증거 |
| 개정 1 파일화·기계 프리플라이트 | 미집행 |
| 독립 Q-GOAL·Q-LEAN 사전감사 | 86d130b 대상 Fable fresh(`claude-fable-5-1`): Q-GOAL 부합(조건부)·Q-LEAN 적정·폐합 차단 0·기재로 족함 11. `/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/fable_preaudit.md`; 개정 1의 감사 결과로 재표기하지 않음 |
| 구현·새 도구 신원 고정 | 발주 시 upstream 경로 미구현; 기안 중 판독기 작업 트리에 미커밋 변경을 조회함. 본 개정 준수·직접시험·새 도구 신원은 미검증 |
| 기존 회귀·새 연결시험 | 미집행 |
| 본 분석·재현·비용 확인 | 미집행 |
| 운전자 판정·astra 종료감사 | 미집행 |

기안자는 파일 수정·수치 분석·직접시험·작업 제출을 하지 않았다. 미집행 항목을 예상 PASS로 채우지 않는다. 과거 원로그가 없으면 `LOG_NOT_PRESERVED`로 기록하고, 새 재실행을 과거 원로그로 표시하지 않는다.

Fable 권고 3의 별도 복원선 선택·전체 소비선 정합성 장치는 적용하지 않는다. 준위 4의 모든 연결선에서 캡처·W5의 존재가 실측되어 기존 표적 교집합·연결선 상대 인구 규칙으로 충분하다.

Fable 권고 5의 내부 연결 시험·별도 단언 구현은 추가하지 않는다. 표적이 하나여서 서로 다른 표적 사이의 연결이 성립하지 않는다.

Fable 권고 8의 동결 J̄ 연립 진단 해는 이번 범위에 추가하지 않는다. 생산 해 재현 불가와 진단용 교환 해의 설계 가능성은 별개이며, §10의 후속 사용자 결정 뒤 별도 계약에서 다룬다.

## §12. 기계 선언

아래 선언은 §7의 변경집합·참조 실물과 **표적·셸별 귀속 코드 및 상대별 연결 성격**의 분기만 검사한다. 인구 복원·율 계산·생산 해 폐합·사슬의 과학적 의미를 대신하지 않는다.

`v`는 선언 검사용 값 하나다. 귀속 검사는 기존 코드 0–6을, 연결 성격 검사는 10=`RETURN_TO_LEVEL_5`, 11=`OUTSIDE_TARGET_SET`을 사용한다. 10·11은 이 선언의 연결 성격 부호화이며 과학 귀속 코드에 추가하지 않는다. 그 밖의 값·결손·복수 값은 `X`다. 연결 상대 ID와 이름의 대응은 §6.4를 따르며, 이 선언은 그 물리적 연결을 검증하지 않는다. coverage·폐합·지배 항·세대·사유·주요/보조 역할은 별도로 기록한다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-SIIII-UPSTREAM (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_siiii_rates_readout\\.py|/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/tools/run_siiii_upstream\\.slurm)",
    "symbol": "det_siiii_rates_",
    "roots": [
      "scripts/det_siiii_rates_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/tools/run_siiii_upstream.slurm"
    ],
    "expected_extra": [
      "scripts/det_siiii_rates_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/tools/run_siiii_upstream.slurm"
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
      [6.0, 6.0],
      [9.0, 9.0],
      [10.0, 10.0],
      [11.0, 11.0]
    ],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,2,3,4,5,6,10,11) for x in v)",
      "c0": "sum(1 for x in v if x == 0)",
      "c1": "sum(1 for x in v if x == 1)",
      "c2": "sum(1 for x in v if x == 2)",
      "c3": "sum(1 for x in v if x == 3)",
      "c4": "sum(1 for x in v if x == 4)",
      "c5": "sum(1 for x in v if x == 5)",
      "c6": "sum(1 for x in v if x == 6)",
      "return5": "sum(1 for x in v if x == 10)",
      "outside": "sum(1 for x in v if x == 11)"
    },
    "rules": [
      {"name": "X", "predicate": "not valid"},
      {"name": "NRB_FORM_DOMINANT", "predicate": "valid and c0 == 1"},
      {"name": "OTHER_RATE_DOMINANT", "predicate": "valid and c1 == 1"},
      {"name": "PARTNER_POPULATION_DOMINANT", "predicate": "valid and c2 == 1"},
      {"name": "RESIDUAL_DOMINANT", "predicate": "valid and c3 == 1"},
      {"name": "MIXED", "predicate": "valid and c4 == 1"},
      {"name": "UNRESOLVED", "predicate": "valid and c5 == 1"},
      {"name": "NOT_IDENTIFIABLE", "predicate": "valid and c6 == 1"},
      {"name": "RETURN_TO_LEVEL_5", "predicate": "valid and return5 == 1"},
      {"name": "OUTSIDE_TARGET_SET", "predicate": "valid and outside == 1"}
    ],
    "adversarial_fixtures": [
      {"name": "nrb", "mix": [[0,1]]},
      {"name": "other-rate", "mix": [[1,1]]},
      {"name": "partner-population", "mix": [[2,1]]},
      {"name": "residual", "mix": [[3,1]]},
      {"name": "mixed", "mix": [[4,1]]},
      {"name": "unresolved", "mix": [[5,1]]},
      {"name": "missing-physics", "mix": [[6,1]]},
      {"name": "return-to-level-5", "mix": [[10,1]]},
      {"name": "outside-target-set", "mix": [[11,1]]},
      {"name": "invalid-code", "mix": [[9,1]]},
      {"name": "missing-code", "mix": []},
      {"name": "multiple-codes", "mix": [[0,1],[1,1]]}
    ]
  },
  "references": [
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/CLAUDE.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PREREG_DET_SIIII_RATES_2026-10-07.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_DET_SIIII_RATES_2026-10-07.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/astra_closeout.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/astra_closeout2.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/results/b2/identity.sha256"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/results/b2/analysis/budget.csv"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/results/b2/analysis/rates.csv"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/results/b2/analysis/populations.csv"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/results/b2/analysis/result.json"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/results/b2/analysis/manifest.json"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/evidence/production_paths_grep_2026-10-07.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/evidence/USER_DECISION_2026-10-07.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/evidence/level4_feasibility_2026-10-07.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/fable_preaudit.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_DET_OBS_WRITER_2026-10-02.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/READOUT_CAUSE1_SII_IONIZATION_CHANNEL_2026-09-25.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/READOUT_CAUSE2_FUV_LINE_LAMBDA_STALL_2026-09-25.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_DET_NRB_E_2026-10-02.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/det_siiii_rates_readout.py",
      "flags_existing": [
        "det_siiii_rates_lines",
        "det_siiii_rates_state",
        "det_siiii_rates_w5_record",
        "det_siiii_rates_sl_check",
        "det_siiii_rates_coll",
        "det_siiii_rates_rad",
        "det_siiii_rates_budget",
        "det_siiii_rates_decompose",
        "det_siiii_rates_dominance",
        "det_siiii_rates_analyze"
      ]
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/det_siiii_pop_readout.py",
      "flags_existing": [
        "recover",
        "capture_records"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c",
      "flags_existing": [
        "nlte_assemble_rate_matrix_impl",
        "nlte_bb_rate_graph_contains",
        "nlte_bb_jbar_canonical",
        "bb_nrb_arm",
        "nlte_build_projection",
        "nlte_precompute_within_sl_frac_checked",
        "nlte_update_tau_sobolev_with_authority"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_net_rate.c",
      "flags_existing": [
        "line_net_bracket_rates",
        "line_net_cmfgen_exponx",
        "line_net_sobolev_radiation"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/population_contract.c",
      "flags_existing": [
        "population_within_superlevel_build"
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