도구를 호출하거나 저장소를 읽지 말고, 아래 제공된 확정 사실만으로 1,400단어 이내의
실행 세부계획을 작성하라. 배경설명보다 단계별 산출물과 PASS/FAIL 갈림길이 중요하다.

Lumina-sn A2-10은 exact deterministic CMF의 line net `4pi*(eta-chi*Jbar)` 부호를
검증한다. 물리값에는 floor/cap/clamp/jitter/repair가 절대 금지이며 음수·부호 불확정은
원인 규명 전 fail closed한다. 허용 후보는 실제 solve residual을 줄이는 것과, 그
residual에서 유도되는 검증된 error envelope의 증명 정밀도 개선뿐이다.

현재 A100x2 iteration 0: 45회, residual 9.6662782724980344e-9 < tol 1e-8,
exact PASS, R6 Q_g=1,391,131/Q_E=2,180,286 전부 valid. R7은 endpoint별 첫
UNRESOLVED_CANCELLATION에서 중단했다. population negative 및 모든 repair counter는 0.

lower witness: line15 shell10, eta=3.4046285166846181e-162,
chi=2.6297471840772978e-8, Jbar=2.9609771047620172e-51,
Jbar bound=2.0938704855191517e-50, signed rate=-9.7849550420208522e-58,
rate uncertainty=6.9194822653875239e-57.

upper witness: line1279130 shell18, eta=1.919589616555497e-37,
chi=9.7379405430504441e-33, Jbar=1.9711586301764101e-5,
Jbar bound=1.3423475903514383e-9, net/sr=8.7062397632962857e-42,
signed rate=1.0940582993553446e-40, uncertainty=1.6426383122528282e-40,
cancellation condition=44095.870032182931.

구현 사실: Jbar bound는 cellwise exact error upper의 Gaussian profile 가중 투영이다.
rate uncertainty는 chi*Jbar_bound를 4pi 및 deck scale로 변환한다. envelope refinement는
`e_(k+1)=residual_upper+K(e_k)`, 현재 8회 고정. 첫 failure 즉시 종료라 나머지 unresolved
분포는 모른다. 다음 계산 후보는 opt-in 전 셀 census, refinement/tolerance 분리 실험,
A100x2 1-iteration 진단, 통과 뒤 4-iteration flight다. 최종 검증은 양팔 일치가 아니라
CMFGEN/ARTIS의 동일 물리량·단위·셀/선에 대한 finite 값 비교여야 한다.

다음에 답하라:
1) 두 witness의 sign 판정에 필요한 Jbar bound와 현 bound/필요 bound 비율을 검산.
2) census, refinement-only, tolerance-only의 올바른 순서와 각 gate.
3) 1-iteration→4-iteration flight gate 및 CMFGEN finite 비교 시작 시점.
4) 놓친 실패 모드와 금지할 조치.

형식: 첫 줄 `VERDICT: ACCEPT/REVISE/REJECT`; 이어서 6~10 번호 단계 각각 목적,
구현 대상, 산출물, PASS/FAIL, 실패 시 갈림길; 마지막 `DO NOT DO`와 값비싼 계산 전
필수 증거 3개. Codex 의견을 반복하지 말고 공격적으로 검수하라.
