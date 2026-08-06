## 평가 전제

[실측] Lumina 논문은 두 팔이 같은 불투명도·방출률과 같은 통계평형을 공유한다고 서술한다(`paper_main.tex:245-276`).  
[실측] 실제 발췌 코드는 결정론 전용 경로(`lumina/lumina_cuda.cu:7889-7914`), 결정론 수렴 뒤 플라즈마를 동결한 MC 경로(`lumina/lumina_cuda.cu:10073-10110`, `10684-10746`), 기본적으로 피드백 없는 MC shadow 경로(`lumina/lumina_cuda.cu:8445-8453`, `9135-9154`), 선택적 MC-rate 피드백 경로(`lumina/lumina_cuda.cu:9206-9237`)를 구별한다.  
[추정] 따라서 아래에서는 논문상의 two-arm 개념과 코드에서 실제로 정의되는 고정점을 분리한다.  
[실측] 물리적 외부 판정자는 CMFGEN이고 ARTIS는 MC 방법론 비교자라는 역할 분담이 논문에 명시돼 있다(`paper_main.tex:788-800`).  
[실측] 다만 사용된 CMFGEN 스냅숏도 고정온도·미완전 population correction 상태여서 조건부 기준이다(`paper_main.tex:776-786`).

## 1. 수렴의 성격과 고정점

| 항목 | ARTIS coevolution | Lumina two-arm |
|---|---|---|
| 반복 사상 | [실측] timestep \(k\)의 packet estimator가 reduce된 뒤 \(k+1\)의 물질 상태에 쓰인다(`artis/sn3d.cc:647-686`, `artis/radfield.cc:43-50`).<br>[추정] 고정 epoch로 쓰면 사상은 대략 \(x_{k+1}=S[\widehat R_N(T(x_k))]\)이다. | [실측] 결정론 전용 루프에서는 CMF field와 NLTE 상태가 반복되고 MC가 우회된다(`lumina/lumina_cuda.cu:7889-7914`).<br>[추정] 이 경로의 사상은 \(x_{k+1}=S[R_{\rm det}(T_{\rm det}(x_k))]\)이다. |
| 실제로 수렴하는 것 | [실측] 기본 설정은 timestep당 반복이 하나이며 물리 시간이 전진한다(`artis/sn3d.cc:938-963`).<br>[추정] 따라서 일반 ARTIS 계산은 한 epoch의 정적 고정점보다 잡음이 실린 시간의존 궤적 \(x(t_k)\)에 수렴한다. | [실측] `THEN_MC`에서는 결정론 상태가 먼저 수렴하고 이후 MC 동안 \(T_e\), ionization, NLTE population을 다시 풀지 않는다(`lumina/lumina_cuda.cu:10073-10085`, `10684-10746`).<br>[추정] 이때 state 고정점은 결정론 팔 하나의 고정점이며 MC 팔은 그 상태에 조건부인 확률분포만 가진다. |
| Shadow two-arm | [실측] ARTIS 원 실행에는 독립적인 noiseless field가 없다(`paper_main.tex:145-151`). | [실측] coevolve shadow는 현재 population으로 MC를 돌리되 결정론 \(J_\nu\)를 덮어쓰지 않고 두 장을 별도로 저장한다(`lumina/lumina_cuda.cu:8445-8453`, `9135-9154`, `9392-9445`).<br>[추정] 그러므로 shadow MC는 수렴 사상의 일부가 아니라 결정론 고정점에 대한 진단이다. |
| MC-feedback two-arm | [실측] MC photoionization field와 line-\(\bar J\)를 다음 반복에 지연 공급하는 선택 경로가 있다(`lumina/lumina_cuda.cu:9206-9237`). | [추정] 이 경로를 켜면 Lumina도 ARTIS형 stochastic coevolution에 가까워지고, 고정점은 MC 혼합계수·점유 gate·지연 규칙을 포함한 별도의 hybrid 고정점이 된다. |

### 고정점이 같은 조건

| 경우 | 판정 |
|---|---|
| 연속 물리 극한 | [추정] 동일한 원자자료·경계조건·시간항·불투명도·방출률을 쓰고 \(N_{\rm pkt}\to\infty\), 공간·주파수·각도 격자를 연속극한으로 보낸다면 두 방법은 같은 NLTE 복사수송–SE 해를 목표로 한다. |
| 유한한 현재 코드 | [실측] ARTIS는 상세 estimator가 없을 때 dilute-Planck 또는 다중-bin 표현을 사용한다(`artis/radfield.cc:716-731`, `artis/ratecoeff.cc:689-709`).<br>[실측] Lumina 결정론 생산 경로는 1000 bins, \(N_{\rm shell}+8\)개의 rays, bin별 expansion opacity를 사용한다(`lumina/lumina.h:533-535`, `lumina/lumina_cmfgen.c:1458-1524`, `1787-1807`).<br>[추정] 따라서 현재 두 코드가 푸는 이산 방정식은 같지 않고 유한 해상도 고정점도 일반적으로 다르다. |
| 논문 도식의 “두 rate가 같은 SE에 입력” | [실측] 논문은 두 입력을 서술하지만 중복되는 동일 물리 rate를 선택·혼합·제약하는 규칙까지 정의하지 않는다(`paper_main.tex:245-276`).<br>[추정] 두 추정치를 단순 합하면 rate를 이중계수하므로, shared SE라는 사실만으로는 하나의 수학적 고정점이 정해지지 않는다. |
| consistent split | [실측] 논문도 consistent split은 같은 고정점을 가지지만 서로 다른 값을 운반하는 inconsistent split은 자체의 비물리적 고정점을 가진다고 명시한다(`paper_main.tex:545-557`). |

[추정] 결론적으로 “같은 문제를 향한다”와 “현재 같은 고정점을 가진다”는 서로 다른 명제다.  
[추정] 전자는 공통 연속극한에서 성립할 수 있지만 후자는 현재의 field 표현, source closure, MC 소유율과 시간처리 때문에 성립하지 않는다.

## 2. 오차: variance와 bias

### MC estimator에서 population까지

[실측] 두 코드의 path estimator는 packet energy와 cell 내부 이동거리의 합으로 \(J_\nu\) 또는 photoionization rate를 만든다(`artis/radfield.cc:675-702`, `paper_main.tex:383-406`).  
[추정] 조건부로 올바르게 표본된 선형 estimator의 분산은 대략 \(1/N_{\rm eff}\)로 감소하지만, 동일 packet의 여러 segment와 macro-atom cascade가 상관돼 있으므로 \(N_{\rm eff}\)는 단순 packet 수보다 작을 수 있다.  
[추정] 정규화 조건을 포함한 SE를 \(A(R)n=b\)라 쓰면 작은 rate 잡음의 일차 전파는
\[
\delta n\simeq-A_c^{-1}\,\delta A\,n
\]
이며 \(A_c^{-1}\)는 보존행을 포함해 제한된 SE 역연산자를 뜻한다.  
[추정] 따라서 작은 MC rate 잡음도 trace ion stage, 거의 특이한 rate matrix, 서로 큰 rate의 차로 정해지는 population에서는 크게 증폭될 수 있다.  
[실측] ARTIS의 bound-bound·bound-free·비열적 rate는 동일한 element-wide dense matrix에 들어간다(`artis/nltepop.cc:478-619`, `621-654`, `1238-1260`).  
[추정] \(n=A^{-1}b\)가 rate에 비선형이므로 estimator가 조건부로 불편향이어도 일반적으로 \(E[n(\widehat R)]\neq n(E[\widehat R])\)인 유한표본 bias가 남는다.

### 오차 분해

| 오차원 | ARTIS coevolution | Lumina two-arm |
|---|---|---|
| MC 표본 variance | [실측] \(J,\nu J\), 상세 bf rate와 선택 line-\(J_{\rm blue}\)가 packet path에서 축적된다(`artis/radfield.cc:656-714`, `828-868`).<br>[추정] 모든 radiative rate와 그에 민감한 population이 realization-dependent다. | [실측] MC arm은 binned \(J_\nu\), line-\(\bar J\), packet spectrum과 event stream을 표본한다(`lumina/lumina.h:579-588`, `lumina/lumina_cuda.cu:3922-3938`, `6230-6257`).<br>[추정] 동결·shadow 모드에서는 이 variance가 spectrum·history에는 남지만 population에는 들어가지 않는다. |
| MC-feedback variance | [추정] coevolution의 본질적 state variance다. | [실측] 선택된 photoionization blend 또는 line-\(\bar J\) consumer를 켜면 MC field가 한 반복 지연으로 state에 들어간다(`lumina/lumina_cuda.cu:9206-9237`).<br>[추정] MC 비중을 \(\alpha\)라 하면 독립 부분의 rate variance는 대략 \(\alpha^2\operatorname{Var}\widehat R_{\rm MC}\)로 남는다. |
| MC 표현 bias | [실측] 상세 continuum estimator가 없으면 binned/dilute-Planck field 적분 또는 LUT로 대체된다(`artis/ratecoeff.cc:689-709`, `artis/radfield.cc:716-731`).<br>[추정] 실제 UV 구조가 이 family에 들어가지 않으면 \(N_{\rm pkt}\to\infty\)에서도 bias가 사라지지 않는다. | [실측] Lumina MC의 직접 photoionization estimator는 bin 내부의 \(\sigma_\nu J_\nu/h\nu\)를 flight 중 표본하도록 설계돼 있다(`paper_main.tex:383-405`).<br>[추정] 이 정보가 결정론-owned SE에 쓰이지 않으면 spectrum에는 존재하지만 state의 bin bias를 교정하지 못한다. |
| 결정론 variance | [추정] ARTIS 자체에는 해당 팔이 없다. | [추정] 고정 격자·고정 상태에서 MC sampling variance는 없지만 roundoff·미수렴 오차는 별개다. |
| 결정론 주파수 bias | [실측] Lumina 생산 조립은 Sobolev line들을 1000개 expansion-opacity bin에 합치며 formal solve는 각 bin을 순회한다(`lumina/lumina_cmfgen.c:1787-1807`, `2578-2589`).<br>[추정] 좁은 UV edge, line-resolved pumping, bin 내부 재분배와 연속 CMF 주파수 결합의 오차가 이 팔에 남는다. |
| 결정론 각도·공간 bias | [실측] rays 수는 \(N_{\rm shell}+8\)이고 각 ray가 교차 shell을 short-characteristic으로 돈다(`lumina/lumina_cmfgen.c:1514-1524`, `2404-2525`).<br>[추정] ray quadrature와 shell reconstruction bias는 packet 수를 늘려도 줄지 않는다. |
| 공통 source bias | [실측] 현재 결정론 line source는 solved-population 경로가 기본 비활성이며 \(B_\nu(T_e)\) fallback을 사용한다(`lumina/lumina_cmfgen.c:1755-1765`, `1804-1807`, `paper_main.tex:983-1003`).<br>[실측] 결정론과 MC field가 600–3000 Å에서 3% 이내로 맞았는데도 둘 모두 CMFGEN보다 과도한 UV field를 가졌고, 논문은 이를 공통 source의 coherent recirculation에 귀속했다(`paper_main.tex:864-895`).<br>[추정] 그러므로 두 장의 일치는 공통모드 bias를 발견하지 못한다. |
| 실패·fallback bias | [실측] ARTIS는 낮은 온도나 회복 불가능한 NLTE matrix 실패 시 LTE population으로 돌아간다(`artis/nltepop.cc:1183-1191`, `1319-1325`).<br>[추정] 이 선택은 잡음이 실패 경계를 넘는 경우 state-dependent bias를 만든다. | [실측] Lumina pair/GPU 경로에도 초기 grey/LTE·실패 fallback 조건이 존재한다(`lumina/lumina_cuda.cu:1390-1429`).<br>[추정] fail-closed 경로와 실제 fallback 경로를 구분하지 않으면 결정론이라는 이유만으로 population bias가 없어지지는 않는다. |

### coevolution 잡음은 누적되는가

| 질문 | 판정 |
|---|---|
| estimator 값을 계속 합산하는가 | [실측] 아니다; ARTIS는 state update가 이전 estimator를 소비한 뒤 \(J,\nu J\), bf와 line estimator를 모두 0으로 만든다(`artis/sn3d.cc:647-673`, `artis/radfield.cc:656-673`). |
| state에는 기억되는가 | [실측] 이전 estimator로 갱신된 population·온도·불투명도가 다음 packet 수송의 입력이 된다(`artis/update_grid.cc:429-490`, `artis/sn3d.cc:676-686`).<br>[추정] 따라서 raw sum은 누적되지 않지만 잡음의 물질상태 피드백은 누적될 수 있다. |
| 안정한 경우 | [추정] 선형화된 state map의 수축률이 \(|\rho|<1\)이면 오래된 교란은 대략 \(\rho^m\)으로 씻기고 현재 estimator가 유지하는 stationary noise floor만 남는다. |
| 광학적으로 두껍거나 trace-stage가 stiff한 경우 | [실측] line-scattering lambda iteration의 \(\rho\)가 두꺼운 한계에서 1에 접근한다는 분석이 논문에 있다(`paper_main.tex:559-590`).<br>[추정] 이 경우 잡음의 상관시간이 길어져 random walk 같은 drift로 보일 수 있지만, 이는 estimator를 산술적으로 계속 더한 결과는 아니다. |
| 실제 시간의존 계산 | [실측] ARTIS packet은 cell boundary뿐 아니라 timestep 끝까지 전파되고 packet 상태가 다음 시간으로 넘어간다(`artis/rpkt.cc:507-640`).<br>[추정] 이때 앞선 잡음은 실제 물리 궤적을 바꾸므로 나중 epoch에서 단순 평균으로 제거할 수 없는 경로 오차가 된다. |

## 3. 도달할 수 있는 물리와 현재 빠진 물리

| 물리 | ARTIS coevolution | Lumina two-arm |
|---|---|---|
| Macro-atom 형광 | [실측] line absorption이 macro-atom을 활성화하고 radiative·collisional·recombination·내부 상하향 전이를 무작위로 걷는다(`artis/rpkt.cc:572-590`, `artis/macroatom.cc:333-529`).<br>[실측] radiative pumping과 비열적 excitation도 내부-up 확률에 포함된다(`artis/macroatom.cc:112-135`). | [실측] MC arm은 line/BF activation과 macro-atom cascade, k-packet, emission event를 실제로 추적한다(`lumina/lumina_cuda.cu:6230-6257`, `6301-6410`).<br>[추정] 따라서 packet spectrum의 형광 재분배는 가능하지만, 동결 모드에서는 그 실제 realization이 population을 되먹임하지 않는다. |
| 결정론 형광 | [추정] ARTIS에는 별도 결정론 emissivity 팔이 없다. | [실측] 논문상으로는 solved upper population의 emissivity가 형광을 담지만(`paper_main.tex:79-94`), 현재 해당 line source는 기본 비활성이고 global/ALI completion은 후속 작업이다(`paper_main.tex:983-1003`, `1053-1055`).<br>[추정] 현재 결정론 고정점만으로 완성된 multilevel fluorescence closure를 주장하기는 이르다. |
| 선중첩·line forest | [실측] ARTIS는 내림차순 line list의 Sobolev resonance를 차례로 통과시키고 각 line optical depth를 경쟁시킨다(`artis/input.cc:1327-1353`, `artis/rpkt.cc:82-193`). | [실측] Lumina MC도 packet이 만나는 Sobolev line들의 optical depth를 순차 합산한다(`lumina/lumina_cuda.cu:3877-3953`).<br>[실측] 별도 overlap correction은 ±10개 이웃, 3 thermal widths 안의 line으로 \(\tau_i^2/(\tau_i+\tau_{\rm overlap})\)를 적용한다(`lumina/lumina_plasma.c:15758-15823`). |
| 선중첩의 한계 | [추정] 두 MC 모두 팽창류의 순차 Sobolev resonance와 line blanketing은 다루지만, 유한 profile이 같은 위치에서 중첩되는 정적 line-transfer 문제를 직접 푸는 것은 아니다. | [추정] Lumina의 overlap correction은 이를 근사하지만, 이웃 수·폭·\(\tau\) 변환 자체가 남는 모델 bias다. |
| 비열적 여기·전리 | [실측] ARTIS SE에는 비열적 bound-bound excitation과 다중 전리 channel이 모두 들어간다(`artis/nltepop.cc:525-558`, `621-654`). | [실측] Lumina 자료구조와 matrix에는 gamma heating 및 `nonthermal_ioniz_rate`가 있지만 비열적 level excitation 항은 보이지 않는다(`lumina/lumina.h:699-707`, `lumina/lumina_plasma.c:14306-14308`).<br>[추정] 발췌 범위에서는 비열적 전리는 가능하지만 비열적 여기 spectrum은 ARTIS와 동등하지 않다. |
| 3차원 | [실측] ARTIS packet은 3차원 위치·방향과 cell boundary를 따라가며 논문도 실제 3D/time transport를 명시한다(`artis/rpkt.cc:507-640`, `paper_main.tex:134-141`). | [실측] 현재 `Geometry`와 packet state는 spherical radial shells의 \(r,\mu\)이고 결정론도 1D rays다(`lumina/lumina.h:190-207`, `lumina/lumina_cmfgen.c:1514-1524`).<br>[실측] 논문은 3D moment/MC-Eddington 확장을 미래 작업으로 둔다(`paper_main.tex:1023-1048`). |
| 시간의존 | [실측] ARTIS는 packet의 timestep-end census와 시간 전진을 실제 수송 사상에 포함한다(`artis/rpkt.cc:528-640`, `artis/sn3d.cc:938-963`). | [실측] Lumina는 epoch를 바꾸면 반경과 밀도를 homologous scaling하지만 한 실행의 iteration은 그 epoch의 정적 반복이다(`lumina/lumina_plasma.c:15826-15838`).<br>[추정] 독립 epoch spectrum은 만들 수 있어도 시간항·packet census·freeze-out을 포함한 공진화는 현재 two-arm이 갖지 않는다. |
| 미분가능 수송 | [실측] Lumina formal solve는 diagonal 및 tridiagonal \(\Lambda\) response를 저장한다(`lumina/lumina_cmfgen.c:2528-2667`).<br>[실측] 완전한 population–temperature–field global Newton/Jacobian-vector product는 개발 중이다(`paper_main.tex:608-632`). | [추정] ARTIS의 개별 packet 경로는 line 선택·boundary crossing·macro-atom 분기 때문에 pathwise 전역 미분가능하지 않다.<br>[추정] ensemble 기대값을 통계적으로 미분하는 것은 원리상 가능하지만, 현재 ARTIS가 noiseless global transport Jacobian을 제공한다는 근거는 없다. |

[추정] 미분가능성은 새로운 원자물리를 추가하는 조건이라기보다, optically thick multilevel coupling과 trace-stage ionization을 한 전역 implicit 해로 안정적으로 닫는 데 필요한 수치적 성질이다.  
[추정] 그러므로 “미분가능하지 않으면 그 물리를 표현할 수 없다”보다 “같은 물리를 강결합 고정점까지 수렴시키기 어렵다”가 정확하다.

## 4. 비용 스케일링

[추정] \(C\)를 cell 또는 shell 수, \(L\)을 line 수, \(q_e\)를 원소 \(e\)의 SE unknown 수, \(N_\nu\)를 frequency bins, \(N_{\rm ray}\)를 rays, \(N_{\rm pkt}\)를 packets, \(K\)를 packet당 실제 line 후보·boundary·interaction 수라 둔다.

| 부분 | 코드에서 읽히는 스케일링 | 실용 한계 |
|---|---|---|
| ARTIS SE | [실측] 원소별로 7개의 \(q_e\times q_e\) rate matrix를 만들고 dense LU를 수행한다(`artis/nltepop.cc:54-106`, `913-953`).<br>[추정] 메모리는 \(O(q_e^2)\), 풀이는 cell·원소당 \(O(q_e^3)\)이며 내부 \(T_e,n_e,\) population 반복 수가 곱해진다. | [추정] 많은 상세 준위를 한 원소에 동시에 넣을 때 dense matrix가 먼저 병목이 될 수 있다. |
| ARTIS packet transport | [실측] 각 packet step은 현재 주파수 아래의 line resonance를 순차 탐색하며 boundary·timestep 끝과 경쟁한다(`artis/rpkt.cc:82-193`, `507-640`).<br>[추정] 비용은 \(O(N_{\rm pkt}K)\)이고 단순 \(O(N_{\rm pkt})\)가 아니다. | [추정] 조밀한 line forest, 높은 재흡수·trapping, 희귀 UV cell/rate의 요구 표본수가 \(K\)와 \(N_{\rm pkt}\)를 함께 키운다. |
| ARTIS line·estimator 메모리 | [실측] line list 자체는 \(O(L)\) 배열이고 상세 line/bf estimator는 선택된 line·continuum과 cell에 따라 늘어난다(`artis/input.cc:1438-1476`, `artis/radfield.cc:56-85`). | [추정] 모든 line에 cell별 상세 estimator를 요구하면 \(O(CL)\) 방향으로 접근한다. |
| Lumina 결정론 line assembly | [실측] 각 shell에서 모든 line을 순회해 expansion opacity와 emissivity를 bin에 합친다(`lumina/lumina_cmfgen.c:1787-1807`).<br>[추정] 이 단계는 \(O(CL)\)이다. | [추정] iron-group line list와 많은 shell의 곱이 직접 비용이 된다. |
| Lumina formal solve | [실측] \(N_{\rm ray}=C+8\)이고 각 frequency·ALI pass에서 각 ray가 최대 \(C\)개의 shell을 통과한다(`lumina/lumina_cmfgen.c:1514-1524`, `2426-2515`, `2578-2667`).<br>[추정] 현재 1D 구현은 최악에 \(O(N_\nu N_{\rm ALI}C^2)\)이고 field 메모리는 \(O(CN_\nu)\)이다. | [추정] frequency resolution, optically thick scattering에서 필요한 ALI 횟수, radial resolution의 곱이 한계다. |
| Lumina SE | [실측] pair 경로는 \(N^2\) matrix를 할당하고 직접 Gaussian elimination을 한다(`lumina/lumina_plasma.c:12514-12569`, `14654-14713`).<br>[실측] super-level 모드는 full levels를 더 작은 solve unknown으로 접는다(`lumina/lumina.h:630-640`).<br>[추정] dense solve 비용은 \(O(q_e^3)\), 메모리는 \(O(q_e^2)\)이며 super-level 수가 실질 변수가 된다. | [추정] element-wide window를 넓힐수록 dense solve가 급격히 비싸지고, 과도한 super-level 압축은 반대로 물리 bias를 남긴다. |
| Lumina MC | [실측] packet별 main loop와 line-list 순차 trace 구조가 있다(`lumina/lumina_cuda.cu:6088-6154`, `3877-3953`).<br>[추정] ARTIS와 마찬가지로 \(O(N_{\rm pkt}K)\)이며 line density와 trapping을 무시한 packet-only scaling은 코드 구조와 맞지 않는다. | [추정] two-arm은 이 MC 비용을 결정론 비용에 더해 지불하며, event history 저장량도 실제 event 수에 비례한다. |
| Lumina arm 공유 메모리 | [실측] \(\tau_{\rm Sobolev}\), transition probabilities, line-\(\bar J\)는 각각 \(CL\), \(C N_{\rm trans}\), \(CL\) 배열을 가질 수 있다(`lumina/lumina.h:211-278`). | [추정] line-resolved 두-field 진단을 모두 켜면 packet 수보다 \(CL\)형 상태 배열이 메모리 한계를 먼저 만들 수 있다. |
| 기하 차원 | [실측] Lumina 발췌에는 3D 결정론 경로가 없고 성능 수치도 아직 TODO다(`paper_main.tex:902-908`, `1023-1048`).<br>[추정] 따라서 Lumina의 3D 비용이나 양 방법의 wall-clock 우열은 이 코드에서 근거 있게 산출할 수 없다. | [추정] ARTIS의 3D 증가는 angular grid가 아니라 cell 수와 packet crossing 수를 통해 들어가며, 미래 Lumina 3D 결정론 scaling은 현재 코드로 판정할 수 없다. |

## 5. ★ two-arm은 대체인가, 포함인가, 다른 것인가

| 질문 | 물리적 판정 |
|---|---|
| coevolution을 대체하는가 | [추정] 결정론-owned frozen/shadow two-arm은 MC realization이 state에 주는 되먹임을 제거하므로 coevolution의 대체물이 아니라 다른 폐루프 분해다. |
| coevolution을 포함하는가 | [실측] Lumina에는 MC field를 다음 반복의 photoionization과 line pumping에 쓰는 선택 경로가 있다(`lumina/lumina_cuda.cu:9206-9237`).<br>[추정] 이 gate에서 MC 소유율을 높이고 결정론 팔을 shadow로 두면 ARTIS형 coevolution이 two-arm 설계 공간의 한 극한으로 들어간다. |
| 현재 기본 실행이 그 포함관계인가 | [실측] 아니다; 기본 coevolve stage는 state를 건드리지 않는 shadow이고 `THEN_MC`는 수렴한 결정론 state를 동결한다(`lumina/lumina_cuda.cu:7338-7355`, `8445-8453`, `10073-10085`). |
| “두 장의 일치”가 주는 것 | [실측] Lumina는 같은 shell/bin에서 \(J_{\rm det}\)와 \(J_{\rm MC}\)의 UV/optical 색과 진폭을 직접 비교한다(`lumina/lumina_cuda.cu:9392-9445`).<br>[추정] 이는 transport estimator의 sampling 문제, ordering 문제, 또는 두 transport kernel의 차이를 분리하는 내부 진단이다. |
| 두 장의 일치가 증명하지 못하는 것 | [실측] 두 장이 3% 안에서 일치했는데도 CMFGEN 대비 UV와 trace photoionization이 1.2–1.8 dex 틀린 사례가 논문에 있다(`paper_main.tex:864-895`).<br>[추정] 같은 opacity·source를 공유하는 두 팔의 일치는 공통 source bias나 빠진 microphysics의 검증이 아니다. |
| coevolution에서는 원리적으로 불가능한가 | [추정] 아니다; 순수 ARTIS 실행 내부에는 noiseless 두 번째 장이 없지만, 동일한 동결 ARTIS state에 독립 결정론 formal solution을 평가하면 비교는 원리적으로 가능하다. |
| 그 순간에도 coevolution인가 | [추정] 그 비교기를 붙이는 순간 방법론적으로 두 번째 팔을 추가한 것이므로, 진단은 coevolution 고유물이 아니라 two-arm 확장물이 된다. |
| MC 복제 두 개로 대신할 수 있는가 | [추정] 독립 MC seed 두 개는 variance와 sampling convergence를 측정할 수 있지만 두 실행에 공통인 field 표현·source bias는 검출하지 못한다. |
| two-arm이 잃는 것 | [실측] ARTIS는 packet과 state를 실제 timestep을 넘어 함께 전진시키지만 Lumina 동결 MC는 state solve를 생략한다(`artis/rpkt.cc:631-640`, `lumina/lumina_cuda.cu:10684-10746`).<br>[추정] 따라서 현재 two-arm은 rare MC channel의 realization-dependent state feedback, packet census가 남기는 시간 기억, 실제 3D–시간 공진화를 잃는다. |
| coevolution이 잃는 것 | [실측] ARTIS 표준 흐름에는 독립 noiseless field가 없고 population이 한 단계 지연된다(`paper_main.tex:145-151`).<br>[추정] 따라서 sampling variance와 공통 모델 bias를 내부에서 분리하기 어렵고 전역 differentiable transport response도 직접 얻지 못한다. |
| event ancestry | [실측] Lumina event record에는 packet ID와 absorption/emission channel이 들어간다(`lumina/lumina_cuda.cu:6230-6257`, `6404-6410`).<br>[실측] ARTIS의 제시된 macro-atom 로그는 cell·ion·입출력 transition을 기록하지만 packet ID 기반 전체 ancestry는 보이지 않는다(`artis/macroatom.cc:410-414`).<br>[추정] 그러므로 현 구현에서는 Lumina가 형광 인과 진단을 더 직접 보존하지만, 그 인과는 동결 모드에서 결정론 state에 조건부다. |

[추정] 가장 정확한 분류는 “two-arm은 coevolution의 상위호환”이 아니라 “MC와 결정론의 state 소유권을 선택할 수 있는 별도 아키텍처”다.  
[추정] state를 결정론 팔이 소유하면 variance를 population에서 제거하는 대신 MC-state 인과 피드백을 잃고, MC 팔이 소유하면 coevolution을 회복하는 대신 population variance도 되돌아온다.

## 6. 부트스트랩과 반복 0

| 단계 | ARTIS | Lumina |
|---|---|---|
| 외부 seed | [실측] 첫 timestep의 온도는 trapped-energy 계산 또는 gridsave에서 이미 주어진다(`artis/update_grid.cc:397-400`). | [실측] TARDIS reference deck를 읽고 \(n_e\)와 \(T_e\)를 deck opacity 자료로 초기화한다(`lumina/lumina_main.c:113-157`). |
| 첫 population | [실측] 첫 timestep은 partition function과 ionization balance를 계산하며 LTE timestep이 최소 하나여야 한다(`artis/update_grid.cc:422-427`, `573-586`). | [실측] deck \(T_e\)를 generation 1 seed로 한 번 발행하고 그 온도에서 partition, electron density, ion population과 Sobolev \(\tau\)를 다시 계산한다(`lumina/lumina_main.c:253-265`, `lumina/lumina_plasma.c:6535-6618`, `6680-6722`). |
| 첫 수송 | [실측] timestep 0 state를 먼저 갱신한 뒤 estimator를 비우고 packet을 수송하며 그 estimator는 다음 timestep에 쓰인다(`artis/sn3d.cc:643-686`). | [실측] CPU 경로의 iteration 0은 estimator를 비우고 seed state로 MC를 돌리며 plasma/NLTE update는 `iter>0`에서 시작한다(`lumina/lumina_main.c:409-429`, `603-674`). |
| LTE 시대 | [실측] `num_lte_timesteps` 동안 이전 MC \(J\)에서 \(T_J\)를 얻되 \(T_e=T_R=T_J,\ W=1\)로 두고 LTE ion balance를 푼다(`artis/input.cc:1737-1740`, `artis/update_grid.cc:447-468`). | [실측] 초기의 optically thick shell만 지정된 `GREY_ITERS` 동안 LTE@\(T_e\)로 보내는 선택 criterion이 있다(`lumina/lumina_cuda.cu:1401-1424`, `8077-8092`). |
| 두 방식의 대응 | [추정] Lumina의 deck-seed LTE가 ARTIS timestep 0의 직접 대응물이다. | [추정] Lumina의 초기 thick-shell grey/LTE와 첫 deterministic \(J_\nu\) 반복을 합친 것이 ARTIS의 여러 LTE timestep이 담당하는 안정화·field 생성 역할에 가장 가깝다. |
| 중요한 차이 | [추정] ARTIS의 LTE 시대는 물리 시간이 실제로 전진하고 두 번째 timestep부터 MC field를 소비하지만, Lumina의 bootstrap은 한 epoch에서 deck seed를 벗어나기 위한 반복이다. | [추정] 그러므로 `num_lte_timesteps`와 `GREY_ITERS`는 역할은 유사해도 같은 물리 시간연산자는 아니다. |

## 조건별 판정 요약

| 조건 | 유리한 물리적 성질 |
|---|---|
| 3D·시간의존·packet census가 핵심 | [추정] 현재 실물 범위에서는 ARTIS coevolution만 해당 물리를 직접 전진시킨다. |
| 한 epoch의 trace-stage population에서 sampling variance를 제거해야 함 | [추정] 결정론-owned Lumina two-arm이 목적에 맞지만, 주파수·source discretization bias를 CMFGEN과 별도로 판정해야 한다. |
| 형광 인과와 packet별 UV→optical 경로가 핵심 | [실측] 두 코드 모두 macro-atom redistribution을 갖지만 Lumina는 packet-ID event stream을 제공한다(`lumina/lumina_cuda.cu:6230-6257`). |
| 비열적 excitation이 핵심 | [실측] 제시된 구현에서는 ARTIS에 명시적 rate가 있고 Lumina에는 nonthermal ionization만 확인된다(`artis/nltepop.cc:525-558`, `lumina/lumina.h:699-707`). |
| 전역 implicit 선형화가 핵심 | [추정] 결정론 response를 가진 two-arm이 적합한 방향이지만 현재 Lumina의 완전한 global linearization은 아직 개발 중이다. |
| 최종 물리 정답 판정 | [실측] Lumina 논문의 규칙대로 CMFGEN population·field·spectrum이 최종 외부 기준이고 ARTIS 일치는 정답 인증이 아니다(`paper_main.tex:788-800`). |

## 물리적으로 결판나는 시험 3개

| 시험 | 설계와 관측량 | 무엇이 결판나는가 |
|---|---|---|
| 1. UV 흡수–광학 형광 응답 시험 | [추정] 동일한 1D plasma와 atomic data에서 좁은 UV band의 입력에너지를 변화시키고, 4000–8000 Å로 재방출된 에너지 분율, 개별 Fe/Co line equivalent width, UV 흡수와 최종 optical emission의 packet ancestry를 측정한다.<br>[추정] line 간격을 0–3 thermal widths로 바꾼 계열을 함께 두어 blend centroid와 escape fraction도 측정한다. | [추정] ARTIS macro-atom, Lumina MC macro-atom, Lumina 결정론 emissivity, optional overlap correction이 같은 형광·선중첩 물리를 내는지 구분된다.<br>[추정] 총 spectrum은 CMFGEN을 기준으로 하고 event ancestry는 energy-channel 진단으로 사용한다. |
| 2. UV trace-ionization 분산–편향 분리 시험 | [추정] 동일 epoch의 SN Ia model에서 ARTIS와 Lumina MC는 여러 독립 seed와 \(N_{\rm pkt}\) ladder를, Lumina 결정론은 \(N_\nu\), ray, shell 해상도 ladder를 사용한다.<br>[추정] shell별 Fe II/III/IV 분율, \(n_e\), 1500–3000 Å flux, Fe II/Fe III optical feature ratio와 photoionization rate를 측정한다. | [추정] seed 간 산포가 줄면서 남는 ARTIS offset은 estimator 표현·비선형 bias이고, deterministic grid refinement에 따라 움직이는 Lumina offset은 discretization bias다.<br>[추정] 두 Lumina field가 서로 맞지만 CMFGEN과 함께 어긋나면 공통 source bias가 직접 확인된다. |
| 3. 3D–시간 기억 시험 | [추정] off-centre \(^{56}\)Ni clump가 있는 ejecta를 여러 epoch에 걸쳐 진화시키고 viewing-angle별 bolometric light curve, UV–optical 색, line centroid·width, polarization과 late-time ion freeze-out을 측정한다.<br>[추정] 같은 model의 spherical limit와 angular average는 epoch별 CMFGEN reference에 연결한다. | [실측] ARTIS는 3D packet과 timestep census를 실제로 전진시키지만 현재 Lumina는 radial 1D stationary-epoch 구조다(`artis/rpkt.cc:507-640`, `lumina/lumina.h:199-207`, `paper_main.tex:1023-1048`).<br>[추정] 따라서 현재 실물의 도달 가능 물리 차이는 viewing-angle 및 시간기억 관측량에서 즉시 드러나며, spectrum 한 epoch의 일치로 숨길 수 없다. |

[추정] 최종 판정은 조건부다.  
[추정] 정적 1D population의 variance 억제와 내부 field 교차진단이 목표면 결정론-owned two-arm이 다른 정보를 주고, 3D·시간의존·realization-dependent state feedback이 목표면 현재 ARTIS coevolution이 다른 물리를 보존한다.  
[추정] 어느 경우에도 MC–결정론 두 장의 일치만으로 물리적 정답이 인증되지는 않으며, 공통 opacity·source·atomic closure의 최종 판정은 CMFGEN 비교가 맡아야 한다.