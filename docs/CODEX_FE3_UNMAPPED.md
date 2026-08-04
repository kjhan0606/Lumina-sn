## 결론

관측된 5,924행은 Fe III가 아니다. LCMFLP01이 기록한 `ion=3`은 0-based `ion_number`이므로 분광학적 **Fe IV**다. 실제 Fe III는 `(Z, ion_number)=(26,2)`이고 job 188932의 NLTE 행렬에 정상적으로 포함되어 있다.

따라서 질문의 직접 답은 다음과 같다.

> Fe III 선은 NLTE 매핑 밖에 있지 않다. 포렌식 집계에서 `ion=3`을 “III”로 읽은 stage-number 해석 오류가 있었다. 매핑 밖인 것은 Fe IV이며, job 188932에서 `LUMINA_NLTE_STAGE4`와 element-wide layout이 모두 꺼져 기본 II/III 창만 사용되었기 때문이다.

### 1. 실제 매핑 술어

자료구조는 `NLTEConfig`의 다음 배열이다.

- `nlte_Z[]`, `nlte_ion[]`: NLTE 대상 `(Z, 0-based ion_number)` 목록
- `nlte_line_map[line]`: 런타임 선 배열 인덱스 → NLTE 이온 슬롯, 아니면 `-1`
- `global_to_nlte_level[global_level]`: 전체 준위 → NLTE full-level 인덱스, 아니면 `-1`

정의는 [src/lumina.h:534](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:534)에 있다.

선 소유권을 만드는 정확한 키는 `line_id`가 아니라

```text
(atom->line_atomic_number[line], atom->line_ion_number[line])
```

이다. 이 쌍이 `target_Z[i], target_ion[i]`와 같으면 `nlte_line_map[line]=i`가 된다. [src/lumina_plasma.c:14161](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14161)

실제 tau/source 덮어쓰기에는 추가로 다음 조건이 필요하다.

1. `nlte_line_map[line] >= 0` — 아니면 즉시 제외. [src/lumina_plasma.c:16987](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16987)
2. `(Z,ion)`의 ion-population block이 존재.
3. 선의 `level_number_lower/upper`를 같은 이온의 `level_num`에서 찾을 수 있음. [src/lumina_plasma.c:17015](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17015)
4. 두 global level 모두 `global_to_nlte_level >= 0`. [src/lumina_plasma.c:17029](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17029)
5. `LUMINA_NLTE_SKIP_Z` 또는 element-wide 후보 슬롯의 셸별 authority가 덮어쓰기를 막지 않음. [src/lumina_plasma.c:17009](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17009), [src/lumina_plasma.c:17040](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17040)

LCMFLP01 writer도 준위 population을 `(Z,ion,level_number)`로 찾고 `global_to_nlte_level`이 `-1`이면 `n_lower/n_upper=-1`을 기록한다. [src/lumina_cmfgen.c:806](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:806)

### 2. 탈락 원인 판별

| 후보 | 판정 | 근거 |
|---|---|---|
| 준위 매핑 실패 | 원자 준위 부재는 배제; 이온 창 부재의 결과로 `-1` | Fe IV는 `levels.csv`에 200준위, `level_number=0…199`; 4,336개 선의 양 끝도 모두 이 범위 안이다. LCMFLP01에서도 `g`, 에너지가 정상 해석됐다. 다만 Fe IV 자체가 target에 없어서 그 준위들의 `global_to_nlte_level`이 전부 `-1`이다. |
| super-level 축약 탈락 | 배제 | projection은 대상 이온의 모든 full level을 먼저 `global_to_nlte_level`에 넣고, 별도로 `fl_to_super`를 지정한다. [src/lumina_plasma.c:14145](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14145) 풀 준위 population도 SL 해를 Boltzmann fraction으로 재분배해 보존한다. [src/lumina_plasma.c:16801](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16801) |
| `line_id` 재번호 조인 파손 | 이 경로에서는 배제 | atomic loader는 `line_id` 열을 읽지 않고 같은 `line_list.csv`의 Z, ion, level-number, f, λ 열을 배열 순서로 읽는다. [src/lumina_atomic.c:649](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:649) `nlte_line_map`도 `line_id`를 전혀 사용하지 않는다. LCMFLP01의 `line_id` 역시 CSV ID가 아니라 루프 인덱스 `l`이다. [src/lumina_cmfgen.c:842](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:842) 대장에 적힌 cross-directory join 결함은 다른 소비자에는 유효할 수 있지만 이 매핑의 원인은 아니다. |
| 이온 window 밖 | **확정** | 기본 target은 Fe `(26,1),(26,2)` 즉 Fe II/III까지만 포함한다. Fe IV `(26,3)`는 없다. [src/lumina_plasma.c:7677](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:7677) `LUMINA_NLTE_STAGE4=1`일 때만 `(26,3)`가 추가된다. [src/lumina_plasma.c:7693](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:7693) job 188932 출력도 `Z=26 ion=1:2698`, `ion=2:1500`만 열거하며 `ion=3` 슬롯이 없다. |
| 파장·f·tau 별도 컷 | 매핑 원인으로 배제 | projection에는 Z/ion 이외의 컷이 없다. f와 λ는 매핑 후 tau 계산에만 쓰인다. LCMFLP01의 `600–3000 Å`, `tau>1e-12`는 캡처 행 선택 조건일 뿐이다. [src/lumina_cmfgen.c:731](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:731) |

코드 자체도 `ion_number 3 = spectroscopic IV`라고 명시한다. [src/lumina_plasma.c:14231](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14231) 반대로 Fe III 진단은 명시적으로 `(Z=26, ion=2)`를 찾는다. [src/lumina_cuda.cu:2007](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:2007)

### 3. 규모와 분포

실행 원자자료와 LCMFLP01을 읽기 전용으로 집계한 결과다.

#### 실제 Fe III `(26,2)`

- 전체 원자 선: **136,263개**
- NLTE 이온 매핑: **136,263/136,263 = 100%**
- 미매핑: **0개, 0%**
- 600–3000 Å: **70,594개**
- LCMFLP01 선택 선: **70,594개**, 모두 lower/upper NLTE 준위가 정의됨
- 선택 행: **141,188행**, 전부 population-defined
- 따라서 “미매핑 Fe III 파장 분포”는 공집합이다.

#### 문제의 실제 이온: Fe IV `(26,3)`

- 전체 원자 선: **4,336개**
- NLTE 매핑: **0개**
- 미매핑: **4,336개, 100%**
- 600–3000 Å 원자 선: **2,981개**
- tau 문턱까지 통과한 LCMFLP01 고유 선: **2,977개**
- 행 수: **5,924행**

5,924행의 파장 분포:

| 대역 | 행 | 비율 |
|---|---:|---:|
| 600–1000 Å | 1,980 | 33.4% |
| 1000–1500 Å | 1,058 | 17.9% |
| 1500–2000 Å | 1,913 | 32.3% |
| 2000–2500 Å | 529 | 8.9% |
| 2500–3000 Å | 444 | 7.5% |

범위는 600.007–2993.846 Å, 중앙값은 약 1480.9 Å다. 즉 EUV/FUV 쪽에 강하게 몰려 있다.

셸별로는:

- s0: 2,977행
- s8: 2,947행
- s16, s20, s45: 0행

매핑 자체는 셸 비의존적이다. 이 셸 분포는 bulk tau가 캡처 문턱 `1e-12`를 넘은 곳만 나타낸 결과다. 기록된 전체 BALL `chi_line` 중 Fe IV 몫은 s0 약 **0.664%**, s8 약 **0.160%**였다. “46.7%”는 전체 BALL이 아니라 모든 미정의 행이 가진 기록-A `chi_line` 내부의 비율이다.

### 4. 희석/볼츠만 bulk 근사의 물리적 타당성

`compute_tau_sobolev`는 `ion_number_density`와 partition function을 받아, 비준안정 준위에는 `W`, metastable에는 1을 곱한 `T_rad` Boltzmann population으로 tau를 만든다. [src/lumina_plasma.c:2636](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:2636)

판정은 두 층으로 나뉜다.

- **T2 population-native 실험의 관점:** 명백한 결함이다. SE로 푼 `n_l,n_u,S_l`가 없으므로 population-native opacity/source를 구성할 수 없다. bulk tau 대입은 사전등록된 단일요인 실험을 깨뜨린다.
- **일반 생산 스펙트럼의 근사로서:** **UNRESOLVED**. Fe IV가 진정한 trace stage이고 충돌적으로 LTE에 가까우면 fallback으로 방어 가능하다. 그러나 이 캡처에서는 광구 쪽 s0/s8에서만 살아 있고, 미정의 opacity의 46.7%를 차지한다. 코드에도 Fe IV를 올리는 `LUMINA_NLTE_STAGE4` 경로가 이미 존재한다. 따라서 “무시 가능한 선이라 bulk면 충분하다”는 근거는 현재 자료에 없다.

생산상 영향의 최종 판단에는 동일 상태에서 stage-IV ON/OFF로 Fe IV `n_l,n_u,S_l,tau`, Fe III/IV 이온분율, BALL flux를 비교해야 한다. 이번 규율상 그 런은 수행하지 않았다.

### 5. Fe III level-17 metastable trap과의 관계

직접 원인은 **별개**임이 코드에서 확인된다.

- 현재 커버리지 결손: Fe IV `(26,3)` 이온 전체가 target window 밖.
- level-17 trap: 정상 매핑된 Fe III `(26,2)` 내부의 준위 17 문제.

코드는 level 17을 Fe III NLTE 배열에서 직접 찾아 population을 읽는다. [src/lumina_cuda.cu:2013](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:2013) 또한 trap의 직접 기전을 “metastable이며 downward radiative line이 0이라 line-driven collision assembly에도 drain이 없고 cascade로 쌓임”이라고 특정한다. [src/lumina_plasma.c:15277](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15277) 실제 Zhang Fe III collision data 경로는 level 17→ground 충돌 drain을 별도로 추가한다. [src/lumina_plasma.c:15423](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15423)

다만 Fe III↔Fe IV 이온화 ladder를 추가하면 level-17 population이 간접적으로 바뀔 가능성까지는 코드 판독만으로 배제할 수 없다. 이 간접 관계는 **UNRESOLVED**다. 판별하려면 stage-IV만 바꾼 A/B에서 다음을 동시에 계측해야 한다.

- Fe III level-17 `b_k/gnd`
- level-17 유입·하향 충돌/radiative rate budget
- Fe III/Fe IV 이온 총량
- Fe IV 선의 population-native tau/source coverage

정리하면, “Fe III가 중심 이온인데 창 밖”이라는 모순은 없다. **중심 이온 Fe III는 완전히 매핑되어 있고, 5,924행은 0-based ion 번호를 로마 숫자로 잘못 읽은 Fe IV 커버리지 결손이다.**