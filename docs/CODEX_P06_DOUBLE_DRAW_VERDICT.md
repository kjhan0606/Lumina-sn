판정은 **CONFIRMED**입니다. `MA_LINE_DESTRUCT`는 ARTIS lane에서 이미 pre-roll에 포함된 동일 전이의 \(C_{\downarrow}\)를 터미널에서 다시 사용해 collisional thermalization 확률을 중복 반영합니다.

## 합산영역

`compute_transition_probabilities`는 활성 레벨의 전체 transition block을 순회합니다([lumina_plasma.c:3811](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:3811>), [lumina_plasma.c:3821](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:3821>)).

각 `ttype==-1` radiative-deactivation 전이에 대해:

1. 해당 물리 라인의 \(C_{\downarrow}\)를 계산합니다([lumina_plasma.c:4177](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4177>), [lumina_plasma.c:4195](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4195>)).
2. 그 동일한 지역변수 `C_down`으로
   \[
   \epsilon_{tid,s}=\frac{C_{\downarrow}}
   {C_{\downarrow}+A_{ul}\beta}
   \]
   를 `ma_line_eps[tid,s]`에 씁니다([lumina_plasma.c:4203](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4203>), [lumina_plasma.c:4214](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4214>)).
3. 이어 같은 `C_down`을 parity 경로의
   `kp_deact += C_down*dE`에 더합니다([lumina_plasma.c:4220](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4220>), [lumina_plasma.c:4240](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4240>)).

따라서

\[
kp_{\rm deact}=\sum_{j\in\text{현재 상위레벨의 }ttype=-1}
C_{\downarrow,j}\Delta E_j
\]

이며, 나중에 터미널로 선택될 라인 \(t\)의 \(C_{\downarrow,t}\Delta E_t\)도 이미 이 합에 포함됩니다. 선택 전 합산이므로 “나중에 선택된 터미널 라인만 제외”하는 장치가 없습니다.

그 합으로

\[
p_k=\frac{kp_{\rm deact}}{sum_{\rm rates}+kp_{\rm deact}}
\]

를 만들고([lumina_plasma.c:4516](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4516>), [lumina_plasma.c:4527](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4527>)), GPU에서 transition 선택보다 먼저 추첨합니다([lumina_cuda.cu:4190](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4190>), [lumina_cuda.cu:4208](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4208>)).

pre-roll을 통과해 `ttype==-1` 전이가 선택되면 동일 `tid`의 `ma_line_eps[tid,s]`를 읽어 다시 추첨합니다([lumina_cuda.cu:4335](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4335>), [lumina_cuda.cu:4346](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4346>), [lumina_cuda.cu:4351](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4351>)).

결론: **터미널 전이의 동일 \(C_{\downarrow}\)가 두 확률에 모두 들어갑니다. 배제 설계가 아닙니다.** 두 branch가 control-flow상 동시에 발생하지 않는다는 주석([lumina_cuda.cu:2101](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:2101>))은 순차 조건부 추첨의 표준 구조일 뿐, 확률 measure 중복을 제거하지 않습니다.

## ARTIS 대조

ARTIS는 각 downward transition에 대해 한 번만 다음 energy-flow 성분을 구성합니다.

- `RADDEEXC`: \(R\Delta E\)
- `COLDEEXC`: \(C\Delta E\)
- `INTERNALDOWNSAME`: \((R+C)E_{\rm lower}\)

근거는 [macroatom.cc:85](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:85>)–[macroatom.cc:110](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:110>)입니다.

그 뒤 모든 macro-atom action을 하나의 누적 배열에 놓고 하나의 uniform draw로 action을 선택합니다([macroatom.cc:392](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:392>), [macroatom.cc:396](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:396>)).

- `RADDEEXC`가 뽑히면 line만 선택한 뒤 즉시 r-packet을 방출하고 종료합니다([macroatom.cc:193](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:193>), [macroatom.cc:227](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:227>), [macroatom.cc:406](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:406>)).
- `COLDEEXC`가 뽑히면 k-packet으로 변환하고 종료합니다([macroatom.cc:421](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:421>)).

ARTIS의 radiative line 선택 뒤에는 \(C/(C+A\beta)\) 추가 파괴 추첨이 없습니다.

## 토이 산술

다른 branch가 없는 2-준위 계에서 \(x=C/(A\beta)\)라 두면:

\[
p_{\rm ARTIS}=\frac{x}{1+x}
\]

Lumina는 pre-roll \(p=p_{\rm ARTIS}\) 뒤 살아남은 radiative branch에서 다시 같은 \(p\)를 적용하므로

\[
p_{\rm Lumina}=p+(1-p)p=2p-p^2
\]

\[
\Delta p=p_{\rm Lumina}-p_{\rm ARTIS}
=p(1-p)=\frac{x}{(1+x)^2}.
\]

| \(C/(A\beta)\) | ARTIS 열화 | Lumina 합성 열화 | 초과 열화 |
|---:|---:|---:|---:|
| 0.001 | 0.099900% | 0.199700% | 0.099800%p, ARTIS 대비 +99.90% |
| 0.01 | 0.990099% | 1.970395% | 0.980296%p, +99.01% |
| 0.1 | 9.090909% | 17.355372% | 8.264463%p, +90.91% |
| 1 | 50.000000% | 75.000000% | 25.000000%p, +50.00% |

특히 \(C\ll A\beta\)에서는 열화가 거의 정확히 두 배가 됩니다.

## 실측

게이트는 실제로 설정됐습니다: `LUMINA_KPACKET=1`([stdout.log:49](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity54/stdout.log:49>)) 및 `LUMINA_MA_LINE_DESTRUCT=1`([stdout.log:71](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity54/stdout.log:71>)). Device flag도 1이고 eps table이 업로드됐습니다([stdout.log:3777](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity54/stdout.log:3777>)).

표본 \(\epsilon\) 산술도 코드 식과 맞습니다([stdout.log:446](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity54/stdout.log:446>)).

- Co III: \(A\beta=2.2699\times10^5\), \(C=49.45\) → \(\epsilon=2.178\times10^{-4}\), 출력 `0.0002`.
- Fe III: \(A\beta=1.9165\times10^4\), \(C=14.13\) → \(\epsilon=7.367\times10^{-4}\), 출력 `0.0007`.
- Ni III: \(A\beta=3.2586\times10^6\), \(C=78.95\) → \(\epsilon=2.423\times10^{-5}\), 출력 반올림 `0.0000`.

12 iteration 합계는:

- 터미널: 6,524,793,663
- 추가 파괴: 34,569,259
- 실현 파괴율: **0.00529814 = 0.529814%**
- iteration별 최대: it6의 **1.0177%**([stdout.log:24509](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity54/stdout.log:24509>))

`[KPACKET]`은 shell0 level-평균 \(p_k=1.491\times10^{-4}\)–\(3.044\times10^{-4}\), shell49는 \(8.033\times10^{-7}\)–\(9.073\times10^{-6}\)를 기록합니다([stdout.log:445](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity54/stdout.log:445>), [stdout.log:14254](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity54/stdout.log:14254>)).

다만 지정 `stdout.log`에는 `[KPD-FE]`가 전혀 없습니다. 소스상 이 진단은 `stderr`에 출력됩니다([lumina_plasma.c:4553](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4553>)). 또한 이 co-evolve 로그의 `[KPACKET]`은 방문가중 실현 pre-roll 횟수가 아니라 level-평균입니다. 따라서 동일 방문 표본에서 \(p_k\)와 terminal destruction을 직접 합산하는 로그 검증은 불가능합니다. 이는 증거 제한이지만, 동일 `C_down` 재사용이라는 소스 판정을 약화하지는 않습니다.

## provenance

`ARTIS_PARITY_GAP_AUDIT.md`는 ARTIS macro-atom의 \((R+C)\epsilon\) action과 k-packet gap을 설명하지만([ARTIS_PARITY_GAP_AUDIT.md:38](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/ARTIS_PARITY_GAP_AUDIT.md:38>)), `MA_LINE_DESTRUCT`나 radiative 선택 후 추가 ARTIS 추첨을 제시하지 않습니다.

도입 주석은 오히려 이를:

- “physical two-level photon destruction”
- “Fork B (T0 finding)”
- ARTIS와 같은 \(A\beta\) lottery가 terminal photon을 항상 재방출한다는 문제의 closure

로 설명합니다([lumina_cuda.cu:2090](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:2090>)). 즉 **ARTIS analog의 이식이 아니라 별도 two-level closure**입니다. parity54 배너가 기존 4-channel draw를 이미 “ARTIS-exact”라고 선언한 것([stdout.log:191](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity54/stdout.log:191>))과도 구분됩니다.

## 판정·처분

**CONFIRMED — P0 ARTIS parity 오염.**

처분 권고:

- ARTIS 비교 lane에서는 `LUMINA_MA_LINE_DESTRUCT=0`.
- 기존 ARTIS 판정런은 이 gate가 켜진 결과를 ARTIS-equivalent로 취급하지 말고 OFF 재판정.
- CMFGEN/ALI lane에서 이 closure를 연구할 수는 있지만, 현재 ARTIS `COLDEEXC` pre-roll과 동시 사용한 형태는 독립적인 물리 검증이 필요합니다.
- 기존 parity54에 대한 직접 사건율 영향 상한은 관측상 iteration 최대 **1.0177% of terminal opportunities**, 전체 가중 평균 **0.5298%**입니다.
- 이것은 최종 스펙트럼 오차의 엄밀한 상한은 아닙니다. 한 번 파괴된 packet은 k-packet 경로에서 다른 파장으로 재분배되고, 한 packet이 여러 terminal opportunity를 거칠 수 있으므로 spectral-bin 변화는 이 비율로 제한되지 않습니다. 따라서 사후 수치 보정 대신 gate-OFF 재실행이 필요합니다.

코드·문서 수정, 시뮬레이션 실행 및 git 조작은 하지 않았습니다.