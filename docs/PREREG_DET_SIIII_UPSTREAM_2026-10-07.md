# 사전등록 — DET-SIIII-UPSTREAM: Si III 준위 6·7·8·10·11의 율 예산과 상류 인구 차이 귀속

2026-10-07 · 기안 역할: astra · read-only  
운전자 파일화 대상: `docs/PREREG_DET_SIIII_UPSTREAM_2026-10-07.md` — **신설 예정**  
상태: **사전감사 전 기안. 구현·기계 게이트·직접시험·본 분석·판정 미집행.**

## §0. Q-GOAL·Q-LEAN 자기답과 과학 질문

### Q-GOAL — 최종목적에 직접 기여하는가?

DET-SIIII-RATES는 Si III 1892 Å 상위 준위 2의 E′−R′ 밀도 차이가 주요 셸 24·30에서 상대 준위 인구 항에 의해 지배됨을 확정했다. 그 항의 주된 구성은 준위 6·7·8·10·11이다. 이는 **승계 사실**이며 이번 단에서 다시 판정하지 않는다.

이번 목표는 **그 다섯 준위의 세대 7 율 예산과 밀도 차이를 분해하여, 준위 2에서 미식별로 남은 상대 인구 차이가 이번 표적에서 율 계수 항으로 귀속되는지, 다시 다른 상대 준위 인구 항으로 이어지는지 판정하는 것**이다.

여기서 율 계수 항은 NRB 형태·복사 입력·충돌 계수 변화를 구별한다. 잔차 지배·혼합·수치 미분리·자료 결손도 등록된 답이다. 상대 준위가 에너지상 더 높다는 전제를 두지 않는다.

이 결과는 후속 조사를 국소 계수, 다른 준위의 인구, 또는 고정 상태의 비평형·기록 결손 중 어디로 이어갈지 결정하는 데 쓰인다. 이를 생략하면 준위 2의 상대 인구 지배를 설명의 종점으로 오인하거나, 1300 Å·1110 Å 준위의 큰 밀도비 자체를 원인으로 재명명하게 된다. 최종 스펙트럼 차이의 물리적 설명을 위한 진단이며, 이번 작업 자체가 스펙트럼을 개선하지는 않는다.

**기안자 자기답: PASS.**

### Q-LEAN — 불필요한 계측·게이트를 추가하는가?

분석은 다섯 표적의 주요 셸 24·30과 보조 셸 19로 제한한다. 표적에 직접 연결된 생산 선의 합집합은 124개다. 인구 복원을 위해 추가로 읽는 소비선은 필요한 준위를 복원하는 용도에만 사용한다.

- 기존 판독기를 일반화하고 러너만 신설한다.
- 새 판독기·생산 solver·복사장 replay·추가 생산 실행을 만들지 않는다.
- 새 직접시험은 준위 표적, 대체 소비선 복원, 공통 구간 교집합, 불일치 처리에 한정한다.
- 선행 30개 직접시험은 회귀로 유지한다. 기존 생산식·오차 기준을 새로 감사하는 별도 시험 묶음을 만들지 않는다.
- 기존 G-LINK·G-RATE 안에서 새 연결을 검증한다.
- 셸 10·15의 자료가 존재해도 분석 범위를 확장하지 않는다.
- 이번 상대 준위들의 율 예산을 다시 재귀적으로 계산하지 않는다.

**기안자 자기답: APPROPRIATE.**

위 자기답은 사전감사를 대신하지 않는다. **Fable fresh가 Q-GOAL, 이어서 Q-LEAN을 독립으로 답한다.** 호출 실패 시 사용자 지정 대체 모델 gpt-6.1-sol을 사용하며 실패 기록과 실제 모델 ID를 보존한다.

### 승인 근거와 주장 상한

사용자 결정문:

`/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/evidence/USER_DECISION_2026-10-07.txt`

DET-SIIII-RATES 폐합 후 다음 후보 「1」을 선택한 결정이다.

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

1. 준위 2의 셸 24·30 분기는 코드 2, `PARTNER_POPULATION_DOMINANT`다.
2. 상대 인구 항의 주된 구성은 준위 8·6·7·10·11이며, 이들의 복원 밀도는 E′가 R′보다 약 \(3.6\times10^4\)–\(1.8\times10^5\)배 높다.
3. 준위 2의 NRB 형태 항과 복사 입력 항은 반대 부호다.
4. Si NLTE 쌍은 (II, III) 하나이며 Si III는 최상위 이온이다. `TOPSTAGE_IV`는 비활성이다.
5. Si III 여기 준위의 직접 과정은 bb 복사·bb 충돌이다. 활성 충돌식은 `NLTE_COLL_FIX`, \(\Upsilon=\max(\Upsilon_{\rm vR},1)\)이다.
6. 그 밖의 직접 항은 `STRUCTURAL_ZERO`다. 간접 영향의 부재를 뜻하지 않는다.
7. 세대 7 캡처는 후속 물질 갱신의 평형해가 아니다. 직전 계약 §5.3의 생산 해 재현 불가와 판정 상한을 승계한다.
8. 선행 30개 직접시험, 생산 연산 순서에 근거한 구간 처리, 종료감사에서 수정된 허용폭·분해·분기 검증을 승계한다.

기존 설정·신원 검사는 **승계 전제가 같은 입력에 적용되는지 확인하는 연결 검사**로 유지한다. 생산 경로 전수조사나 확정 사실의 재감사를 반복하지 않는다.

### 1.4 새 표적의 자료 가용성 근거

- `/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/evidence/upstream_feasibility_2026-10-07.txt`
- `/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/evidence/upstream_missing_partners_2026-10-07.txt`

이 기록은 소비선·W5 행의 존재와 연결 가능성을 보여 준다. **역산 적격성·공통 인구 양립성·율 예산 완전성의 통과 결과는 아니다.**

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
| `work/nlte_levels_final.csv` | 세대 8 보조 비교만 |
| D0 정본의 분석 산출 | 준위 2의 확정 귀속과 이번 표적을 연결하는 선행 근거 |

생산식의 권위는 T2다. 현재 작업 트리의 생산 소스로 바꾸지 않는다.

대형 캡처의 기존 봉인 신원을 승계하고 새로 선택한 레코드의 출처를 기록한다. 전체 복제·전체 CSV 변환을 하지 않는다. 새 결과 경로가 이미 있으면 거부하고 새 실행 식별자를 사용한다. 기존 결과와 실패 기록은 보존한다. 특정 파일시스템의 스냅숏·reflink·확장 속성 등을 전제로 하지 않는다.

## §2. 표적·셸·생산 범위

### 2.1 준위 표적

모든 표적은 Z=14, ion=2의 **준위**로 지정한다. 대표 방출선 행을 표적의 필수 식별자로 요구하지 않는다.

| 준위 | 명칭 | g | 에너지, eV | 연결선 | 자기 자신을 제외한 상대 준위 | 그중 준위 ≥100 |
|---:|---|---:|---:|---:|---:|---:|
| 6 | 3p² ³P₀ | 1 | 16.081798199 | 11 | 11 | 3 |
| 7 | 3p² ³P₁ | 3 | 16.0983525691 | 26 | 26 | 3 |
| 8 | 3p² ³P₂ | 5 | 16.1304086836 | 20 | 20 | 3 |
| 10 | 3s3d ³D₂ | 5 | 17.723025387 | 33 | 33 | 8 |
| 11 | 3s3d ³D₁ | 3 | 17.7233241889 | 34 | 34 | 7 |

실측 기록의 상대 목록 12·27·21·34·35는 표적 자신을 포함한 끝점 집합이다. 예산의 합은 \(j\ne u\)에 대해서만 취한다.

전이 ID는 CSV 헤더를 제외한 **0-based 데이터 행**이다. 별도 `line_id` 열로 치환하지 않는다. 표적이 전이의 하위 준위인 선도 전부 포함한다.

### 2.2 셸의 역할

| 셸 | 등록 역할 | 결론에서의 사용 |
|---|---|---|
| 24·30 | 주요 셸 | 준위 2의 선행 상대 인구 지배를 이어받는 주 판정 |
| 19 | 보조 셸 | 표적 연결선 소비가 완전한 상태에서 동일 분해를 비교; 주요 셸 대체 불가 |
| 17–30 중 위 셸 외 | 가용성·결손 기록 | W5 결손에 따른 전체 귀속 미식별 보존; 보간하지 않음 |
| 10·15 | 분석 제외 | W5 존재 사실만 기록; 새 인구·율 분석 없음 |

다섯 준위 전부에 같은 역할을 적용한다. 주요 판정 단위는 **표적 준위×셸**이며, 24·30을 평균하거나 다수결로 묶지 않는다. 셸 19가 더 완전하거나 원하는 분기를 보인다는 이유로 주 결론을 옮기지 않는다.

### 2.3 직접 생산 과정

다섯 준위는 모두 cutoff 100 미만의 개별 solve 준위다. 직접 예산은 bb 복사와 bb 충돌만 계산한다.

직접 광이온화·복사 재결합·충돌 이온화 및 삼체 재결합·DR·비열적 이온화·전하교환은 승계 근거에 따라 `STRUCTURAL_ZERO`로 기록한다. 이를 위한 새 적분기나 반응 계산기를 구현하지 않는다.

근거:

`/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/evidence/production_paths_grep_2026-10-07.txt`

준위 ≥100의 상대 준위는 실제 super-level 100의 구성원으로 취급한다. full-level 전이 계수와 구성원 인구를 사용한 예산에 투영 분율을 다시 곱하지 않는다.

### 2.4 표적 집합 내부 연결

기안 시 E·R의 선 목록을 조회한 결과, 다섯 표적 사이의 직접선은 발견되지 않았다. **10↔11 및 7↔8도 없다.** 에너지 인접성이나 항 명칭만으로 연결을 추가하지 않는다.

준위 2는 다섯 표적의 실제 상대 준위다. 따라서 이번 귀속이 다시 준위 2의 인구 항을 가리킬 수 있다. 이는 §6의 되돌아오는 연결로 보고하며, 독립된 새 원인으로 세지 않는다.

일반화된 판독기는 표적끼리 연결된 입력도 처리할 수 있어야 한다. 그 경우 각 표적의 방정식에 해당 선을 한 번씩 올바른 방향으로 넣고 같은 공통 인구표를 사용한다. 현재 생산 입력에 없는 연결을 결과에 만들어 넣지는 않는다.

## §3. 세대 7 공통 인구 복원

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

### 3.2 필요한 인구와 복원선의 분리

필요한 준위 집합은 다음으로 고정한다.

\[
L=U\cup\{j:\text{등록 표적 }u\in U\text{와 생산 선으로 연결됨}\},
\quad U=\{6,7,8,10,11\}.
\]

복원 후보선은 **같은 런·같은 셸·같은 캡처 세대에서 소비·mapped이고, 끝점 중 하나가 \(L\)에 속하는 Si III 선**이다. 표적과 상대 준위를 직접 잇는 선으로 제한하지 않는다.

복원 후보선의 다른 끝점 때문에 필요한 인구 집합이나 율 예산을 재귀 확장하지 않는다. τ·S 역산은 두 끝점 인구를 직접 주므로 그 다른 끝점의 사전 인구를 요구하지 않는다.

율 선과 복원선을 구별한다.

- **율 선:** 표적에 연결된 124개 생산 선.
- **복원선:** 필요한 준위의 인구를 얻는 소비선. 율 선 밖일 수 있다.
- 복원선 자신의 W5가 없더라도 τ·S 역산 권위가 적격하면 인구 복원에 사용할 수 있다.
- 복원선의 J̄·β·τ·S를 다른 율 선으로 옮기지 않는다.

동일 입력을 선택 판독한 결과를 재사용하며, 표적마다 캡처를 다시 스캔하지 않는다.

### 3.3 공통 구간 교집합

런 \(X\), 셸 \(s\), 준위 \(k\)마다 모든 적격 복원선의 구간을 모은다.

\[
\mathcal I^X_{k,s}
=
\bigcap_{r\in\mathcal E^X_{k,s}}
[n^{X,r}_{k,\mathrm{lo}},n^{X,r}_{k,\mathrm{hi}}].
\]

규칙은 다음과 같다.

1. 복원선별 행 ID·끝점 방향·세대·적격성·원 구간을 보존한다.
2. 직접 연결선과 대체선을 모두 같은 교집합에 넣는다. 먼저 성공한 선만 고르지 않는다.
3. 비어 있지 않은 교집합을 해당 상태·셸·준위의 **유일한 공통 인구 구간**으로 사용한다.
4. 모든 표적·유입·유출 항은 이 공통 인구를 재사용한다.
5. 대표값은 교집합 내부의 결정론적 값으로 정한다. 대표값 선정은 오차폭 확대나 물리 인구의 clamp가 아니다.
6. 선 하나만 적격하면 그 구간을 사용하고 단일선 복원임을 기록한다.
7. 부적격 선은 기존 생산·역산 기준에 따라 제외하고 사유를 남긴다. 결과에 맞추어 제외하지 않는다.

적격 구간들의 교집합이 비면 `POPULATION_INCONSISTENT`다. 평균·중앙값·임의 허용폭·원하는 부분집합으로 덮지 않는다. 해당 준위를 필요로 하는 표적 예산은 코드 6으로 두고 충돌한 선과 구간을 기록한다.

적격 복원선이 없고 승인된 투영으로도 얻지 못하면 `MISSING_G7_POPULATION`이다. 불일치가 존재하는 인구를 투영 유도로 우회하지 않는다.

### 3.4 미소비 율 선의 처리

실측된 R′ 미소비선 수는 다음과 같다.

| 셸 | 준위 6 | 준위 7 | 준위 8 | 준위 10 | 준위 11 |
|---:|---:|---:|---:|---:|---:|
| 19 | 0 | 0 | 0 | 0 | 0 |
| 24 | 4 | 1 | 0 | 0 | 0 |
| 30 | 4 | 6 | 5 | 0 | 3 |

E′는 등록 분석 셸의 모든 표적 연결선을 소비했다. 미소비선의 정확한 행 ID와 상대 준위는 §1.4 실측 증거를 연결한다.

미소비 율 선도 다음이 충족되면 전체 예산에 포함한다.

- 양 끝점의 세대 7 공통 인구가 같은 상태·셸에서 적격하게 복원됨.
- 그 율 선 자신의 원자 계수와 W5 복사 입력이 적격함.
- 생산의 실제 NRB 분기와 충돌 분기를 적용할 수 있음.

그 선 자체의 캡처 τ·S는 **없음으로 유지**한다. 다른 선의 τ·S를 복사하거나 가짜 소비 레코드를 만들지 않는다. 소비 여부와 전체 율 계산 가능 여부를 별도 열로 기록한다.

미소비선이 약하다는 이유로 버리거나 0으로 놓지 않는다. 반대로 소비되지 않았다는 이유만으로 생산 율 선에서 삭제하지 않는다. 소비·mapped는 인구 역산의 권위이고, 생산 율 그래프 포함 여부는 별도 권위다.

### 3.5 Super-level과 세대 8

기존 super-level Boltzmann 양립성 검사를 필요한 구성원에 적용한다. 생산 온도·투영 권위가 확인된 기존 방식의 투영 유도는 `G7_PROJECTION_DERIVED`로 구별한다. 별도 super-level 사이를 LTE 비로 채우지 않는다.

세대 8은 새로 복원된 인구와의 보조 비교만 허용한다. 결손·불일치 세대 7 인구를 세대 8로 채우지 않는다. 선행 인구가 이번 공통 구간과 양립하는지 확인할 때도 과거의 요약 mid 값을 정확한 참값으로 취급하지 않는다.

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

`STRUCTURAL_ZERO`, 활성 경로의 계산된 0, `MISSING`을 구별한다. 미소비선의 τ·S 결손과 그 선의 율 계산 불가는 같은 상태가 아니다.

## §5. 폐합과 수치 기준

### 5.1 오프라인 산술 폐합

- 전이별 합과 과정별·표적별 예산이 일치해야 한다.
- 같은 준위는 모든 항에서 공통 인구표를 사용해야 한다.
- 한 표적의 행 안에서 전이를 중복 합산하지 않는다.
- 여러 표적이 공유하는 전이는 각 표적 방정식에 올바르게 나타나야 한다.
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

이번에는 이 조건을 재조사하거나 생산 해·동결 세대 7 해를 새로 풀지 않는다. 준위 표적 확대나 대체선 복원은 이 제한을 해소하지 않는다.

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
| 선행 연결 | 셸별 준위 2의 코드 2와 \(B_{2\leftarrow u}\), \(u\in U\), 선행 정본 출처 |
| 이번 표적 | \(u\), 셸, \(\Delta n_u\), 분해 항·구간, 코드·지배 항·잔차·완전성 |
| 이번 상대 연결 | 각 \(j\)에 대한 \(B_{u\leftarrow j}\), 부호·구간·인구 출처·에너지 관계 |
| 연결의 성격 | 선행 준위 2로 복귀, 표적 집합 내부 연결, 그 밖의 상대 준위 |

모든 상대별 항을 기계 산출에 남긴다. 본문에서 일부 큰 항만 제시하면 나머지의 합과 전체표를 함께 연결한다.

표현 규칙:

- 코드 0·1은 **“이번 표적의 등록된 국소 분해에서는 계수 항 지배”**로 쓴다. 인과 사슬의 최종 원인을 찾았다고 쓰지 않는다.
- 코드 2는 **“상대 인구 항으로 이어짐”**으로 쓴다. 그 상대의 인구 원인은 이번 범위 밖이다.
- 준위 2로 되돌아오면 `RETURN_TO_LEVEL_2`로 표시한다.
- 표적 집합 내부 연결이 있으면 `IN_TARGET_SET`으로 표시하고 이미 계산한 해당 준위 행을 참조한다.
- 그 밖은 `OUTSIDE_TARGET_SET`으로 표시한다. 에너지가 낮은 상대도 있을 수 있다.
- 같은 준위가 여러 표적의 상대여도 별개의 독립 원인처럼 세지 않는다.
- 순환 연결은 순환으로 남긴다. 재귀 대입·무한 사슬 확장·새 연립해 계산을 하지 않는다.

선행 \(B_{2\leftarrow u}\)와 이번 \(B_{u\leftarrow j}\)는 서로 다른 표적의 분모를 가진다. **몫을 곱하여 준위 2의 인과 기여율이나 캐스케이드 확률로 만들지 않는다.** 보조 비율을 표시할 때는 대수적 항의 구성비라는 분모와 부호를 명시하고, 분모가 수치적으로 0과 분리되지 않으면 표시하지 않는다.

셸 19의 사슬은 보조 결과다. 24·30에서 얻은 결론을 다른 셸이나 전체 반복 이력으로 확장하지 않는다.

## §7. 최소 변경집합·산출

### 변경집합 — DET-SIIII-UPSTREAM (표)

**기존 판독기 수정으로 구현 가능하다.** 율·분해·구간·분기 계산을 재사용하고, 선 표적 입력과 선별 인구 복원 부분을 일반화한다. 새 판독기 파일은 만들지 않는다.

| 경로 | 함수·항목 | 허용 변경 |
|---|---|---|
| `scripts/det_siiii_rates_readout.py` | `det_siiii_rates_*` | 준위 표적 설정, 필요한 복원선 선택, 공통 인구표·교집합·불일치 전파, 기존 예산 연결, 상대별 사슬 산출, 해당 직접시험·manifest |
| `/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/tools/run_siiii_upstream.slurm` | 오프라인 러너 — 신설 | 환경·신원·자원 기록, 기존 회귀와 새 연결시험, 등록 분석·게이트 실행, stdout·stderr·rc 보존 |

### 변경집합 끝

신규·변경 분석 함수의 접두는 `det_siiii_rates_`로 제한한다. 기존 `cmd_analyze`, `cmd_selftest`, `main`의 변경이 필요하면 접두 함수로 이동·개명하고 필요한 호출 연결만 바꾼다. 모듈 상수·CLI·문서문자열·import 변경은 이 일반화에 직접 필요한 범위만 허용한다.

기존 수치 보조 함수와 역산 의존 파일, 생산 소스·바이너리·원자자료·봉인 입력·선행 러너·선행 산출은 변경하지 않는다. 기계 게이트 도구도 변경집합 밖이다.

기존 선 표적 CLI와 30개 시험의 사용 경로를 유지한다. upstream 표적 집합은 명시적으로 선택하고 기존 실행의 기본 표적을 조용히 바꾸지 않는다. 출력 manifest는 실제 선택한 계약·표적 집합·셸 역할을 기록한다.

| 필수 산출 | 내용 |
|---|---|
| `coverage.csv` | 율 선 소비 여부, W5 적격성, 인구 출처, 대체 복원 여부, 결손·구조적 0, 셸 역할 |
| `populations.csv` | 복원선별 후보 구간·적격성·사유와 상태·셸·준위별 공통 교집합 |
| `rates.csv` | 전이별 실제·가상 계수, 공통 인구, 유입·유출·순율·구간·생산 분기 |
| `budget.csv` | 표적·셸별 전체 예산·잔차·密度 차이 분해·코드; 문서·출력의 한글 명칭은 “밀도” 사용 |
| `chain.csv` | §6.4의 선행 연결·이번 상대별 항·연결 성격·증거 출처 |
| `closure.json` | 산술 폐합·공통 인구 양립성·세대 7 잔차와 생산 해 재현 불가 승계 |
| `checks.json` | G-LINK·G-RATE 집계와 시험 증거 연결 |
| `selftest.json` | 기존 30개 회귀와 새 연결시험의 구별·결과·도구 신원 |
| `result.json` | 주요·보조 셸별 결론, 질문의 해소·미해소, 주장 상한 |
| `manifest.json` | 계약·선행 근거·입력·도구·명령·환경·작업·산출 신원 |

위 산출은 새 실행 디렉터리에 신설한다. 미정의 수치는 `null` 또는 빈 수치칸과 사유로 남긴다. 판정문과 종료감사는 계산 산출·비용·분장·이탈을 함께 연결한다.

## §8. 게이트·직접시험

| 게이트 | 새로 막는 실패 | 통과 조건 |
|---|---|---|
| G-LINK | 선 표적 가정 잔존, 대체 복원선의 상태·셸·세대 혼합, 불일치 인구 은폐, 미소비선의 누락·가짜 소비 처리 | §2–§3의 준위 식별·복원 후보·공통 교집합·결손 처리가 실제 분석 출력까지 적용됨 |
| G-RATE | 새 공통 인구표 연결에서 생긴 방향·중복 오류, 상대별 합 누락, 사슬의 중복·과장 | 기존 율·분해·분기 회귀 유지, 새 준위 연결의 독립 정답·항 합산·출력 의미 일치 |

### 8.1 기존 시험의 승계와 회귀

선행 수리판의 **30개 사례를 이름·의미·판정 기준과 함께 유지하여 수정 도구로 다시 실행**한다. 이는 변경으로 인한 회귀 검사이며 선행 과학 판정의 재감사가 아니다.

기존 시험의 엄밀한 정답 구간 포함 기준, 임의 허용폭 금지, 실제 분석 경로 검증을 유지한다. 과거 PASS 로그만 연결하여 이번 도구의 회귀 실행을 대신하지 않는다.

### 8.2 새 연결시험

생산 레이아웃의 작은 가상 입력에서 실제 `det_siiii_rates_analyze`와 산출 경로를 호출한다.

1. **준위 표적**
   - 대표 선 `row`·`lo` 없이 준위 ID로 예산을 계산한다.
   - 표적이 전이의 하위·상위에 각각 놓이는 경우를 포함한다.
   - 표적끼리 연결된 가상 선에서 같은 공통 인구를 사용하고 각 행의 방향·중복 처리가 맞는지 확인한다.

2. **대체 소비선 복원**
   - 율 선의 캡처만 제거하고, 같은 상대 준위를 다른 소비선에서 복원하게 한다.
   - 해당 율 선의 W5는 유지한다.
   - 완전 입력과 양립하는 인구·예산을 얻고 미소비·대체 출처가 출력되는지 확인한다.
   - 대체선은 표적 124개에 해당하는 율 선 집합 밖의 선 역할을 포함한다.
   - 적격 대체선까지 제거하면 코드 6으로 전환되는지 확인한다.

3. **교집합**
   - 여러 소비선이 같은 준위의 겹치는 구간을 주게 한다.
   - 출력 공통 구간이 전 후보의 교집합이며 모든 사용처에서 같은지 확인한다.
   - 입력 선 순서를 바꾸어도 후보 선택·교집합·분기가 달라지지 않는지 확인한다.

4. **불일치·부적격**
   - 적격 후보끼리 교집합이 비도록 주입하여 `POPULATION_INCONSISTENT`와 관련 예산 코드 6을 확인한다.
   - 세대·셸·mapped·역산 권위가 부적격한 대체선이 적격 후보로 들어가지 못하게 한다.
   - 유효한 대체선이 없을 때 세대 8이나 가짜 τ·S로 총예산을 만들지 않는지 확인한다.

각 사례는 주입 위치·독립 정답 또는 기대 사유·실제 사유·rc를 보존한다. 시험 수 자체를 목표로 추가 배터리를 만들지 않는다.

### 8.3 기계 프리플라이트

기존 도구를 그대로 사용한다.

- `scripts/check_prereg_preflight.py`
- `scripts/check_gate_referents.py`
- `scripts/check_doc_canonical.py`
- `scripts/check_dispatch_preflight.py`

검사한 문서·발주서 신원과 각 stdout·stderr·rc를 보존한다. 기계 PASS를 과학 감사나 수치 검증으로 표시하지 않는다.

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
  --prior-pop "$B/results/a2/analysis" \
  --prior-rates "$D0/results/b2/analysis" \
  --selftest "$D/results/$RUN_ID/selftest" \
  --out "$D/results/$RUN_ID/analysis"
```

`--prior-pop`은 기존 CLI 호환용이며 세대 7 인구의 대입원이 아니다. `--prior-rates`는 선행 준위 2의 확정 귀속·사슬 연결 근거를 읽는다. 이번 인구는 E·R 캡처에서 복원한다.

upstream의 표적·주요 셸·보조 셸은 본 계약대로 고정하고 결과를 보고 바꾸지 않는다.

기계 게이트 계획 명령:

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

- 다섯 표적의 주요 셸 24·30과 보조 셸 19에 대한 자료 연결·공통 인구·율 예산.
- 미소비 율 선의 대체 인구 출처와 적격성·교집합·불일치 결과.
- 과정별 율·잔차·대칭 분해·상대별 인구 항.
- 계수 항 지배인지, 상대 인구 항으로 이어지는지, 또는 다른 등록 분기인지에 대한 표적·셸별 답.
- 선행 준위 2와 연결한 한 단계 사슬 및 순환·범위 밖 상대의 명시.
- 기존 30개 회귀와 새 연결시험, 기계 게이트 원로그.
- 재현 가능한 신원·명령·비용·보존·분장·이탈 기록.

### 10.2 처분

| 처분 | 기준 |
|---|---|
| `PASS_LIMITED` | 구현·시험·필수 산출이 적격하게 완료됨. 생산 해 폐합의 선행 미달을 반드시 보존하고, 추가 인구·율 결손이 있으면 그로 인해 미해소인 주요 표적·셸과 과학 질문을 첫 문단에 명시 |
| `REWORK` | 계약과 다른 복원·연결·계수·오차·분해·분기·시험 구현, 불일치 은폐, 미소비선 누락, 사슬 주장 과장 |
| `BLOCKED` | 핵심 입력·권위·자원 등의 실제 문제로 등록된 최소 산출을 완료하지 못함 |

직전 계약의 생산 해 재현 불가와 완료 기준을 승계하므로 **전체 처분 상한은 `PASS_LIMITED`**다. 이번에 질문을 좁혔다는 이유로 선행 생산 해 폐합의 미달을 삭제하여 `PASS`로 높이지 않는다.

대수적 질문의 달성 범위는 처분과 별도로 명시한다.

- 다섯 표적의 주요 셸 모두에서 완전한 유효 예산과 등록 분기가 나오면, **“등록 주요 셸의 상류 대수 귀속 완료; 생산 해 폐합 미달”**로 쓴다.
- 주요 셸에 코드 6이 남으면, 해당 표적·셸의 상류 귀속은 미달이다.
- 코드 4·5는 완전한 유효 예산에 근거할 때 정당한 혼합·수치 미분리 답이다.
- 결손 목록 작성만으로 상류 귀속 목표 전체를 달성했다고 쓰지 않는다.
- 보조 셸 19의 성공으로 주요 셸의 결손을 대체하지 않는다.

생산 수리·후속 준위 선택·새 캡처·반복 연장은 이번 처분으로 자동 승인되지 않는다.

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
| 사용자 결정 | §1.4와 별도의 D 사용자 결정문; 후보 「1」 |
| 선행 폐합 | `0f0e588`, D0 정본 b2, `PASS_LIMITED` 승계 |
| 새 표적 가용성 실측 | 운전자·lageunha; §1.4 증거 |
| 본문 기안 | read-only; 공유 컨텍스트·계약·판정·실측 기록·판독기·관련 게이트 조회 |
| 표적 간 선 확인 | E·R 선 목록 조회에서 10↔11 포함 표적 집합 내부 직접선 없음 |
| 정본 파일화·기계 프리플라이트 | 미집행 |
| 독립 Q-GOAL·Q-LEAN 사전감사 | 미집행 |
| 구현·새 도구 신원 고정 | 미집행 |
| 기존 회귀·새 연결시험 | 미집행 |
| 본 분석·재현·비용 확인 | 미집행 |
| 운전자 판정·astra 종료감사 | 미집행 |

기안자는 파일 수정·수치 분석·직접시험·작업 제출을 하지 않았다. 미집행 항목을 예상 PASS로 채우지 않는다. 과거 원로그가 없으면 `LOG_NOT_PRESERVED`로 기록하고, 새 재실행을 과거 원로그로 표시하지 않는다.

## §12. 기계 선언

아래 선언은 변경집합·참조 실물과 **표적·셸별 귀속 코드**의 분기만 검사한다. 인구 복원·율 계산·생산 해 폐합·사슬의 과학적 의미를 대신하지 않는다.

`v`는 단일 코드다. 0–6 이외 값·결손·복수 코드는 `X`다. coverage·폐합·지배 항·세대·사유·주요/보조 역할은 별도로 기록한다.

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
      [9.0, 9.0]
    ],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,2,3,4,5,6) for x in v)",
      "c0": "sum(1 for x in v if x == 0)",
      "c1": "sum(1 for x in v if x == 1)",
      "c2": "sum(1 for x in v if x == 2)",
      "c3": "sum(1 for x in v if x == 3)",
      "c4": "sum(1 for x in v if x == 4)",
      "c5": "sum(1 for x in v if x == 5)",
      "c6": "sum(1 for x in v if x == 6)"
    },
    "rules": [
      {"name": "X", "predicate": "not valid"},
      {"name": "NRB_FORM_DOMINANT", "predicate": "valid and c0 == 1"},
      {"name": "OTHER_RATE_DOMINANT", "predicate": "valid and c1 == 1"},
      {"name": "PARTNER_POPULATION_DOMINANT", "predicate": "valid and c2 == 1"},
      {"name": "RESIDUAL_DOMINANT", "predicate": "valid and c3 == 1"},
      {"name": "MIXED", "predicate": "valid and c4 == 1"},
      {"name": "UNRESOLVED", "predicate": "valid and c5 == 1"},
      {"name": "NOT_IDENTIFIABLE", "predicate": "valid and c6 == 1"}
    ],
    "adversarial_fixtures": [
      {"name": "nrb", "mix": [[0,1]]},
      {"name": "other-rate", "mix": [[1,1]]},
      {"name": "partner-population", "mix": [[2,1]]},
      {"name": "residual", "mix": [[3,1]]},
      {"name": "mixed", "mix": [[4,1]]},
      {"name": "unresolved", "mix": [[5,1]]},
      {"name": "missing-physics", "mix": [[6,1]]},
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
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/evidence/production_paths_grep_2026-10-07.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/evidence/USER_DECISION_2026-10-07.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/evidence/upstream_feasibility_2026-10-07.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_upstream_20261007/evidence/upstream_missing_partners_2026-10-07.txt"
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