# Fable 중요 계획 위임 — A2-10 cancellation 폐합 (2026-08-15)

당신은 Lumina-sn의 독립 검수자다. 구현은 하지 말고, 아래 현장과 저장소 코드를
직접 대조하여 **반증 가능한 세부 실행계획**을 작성하라. 동의나 요약보다 계획의
순서, 단계별 산출물, 통과/중단 조건, 값비싼 계산을 피하는 갈림길이 중요하다.

## 절대 제약

1. 물리량에 numerical floor/cap/clamp/jitter/음수 삭제/사후 repair를 쓰지 않는다.
2. 음수 또는 부호 불확정은 원인을 증명하기 전 패치하지 않고 fail closed한다.
3. `J`, population, emissivity, opacity 등 생산 물리값은 판정 편의를 위해 변경하지 않는다.
4. 허용 후보는 실제 exact solve의 더 작은 residual, 그리고 그 residual로부터 유도되는
   검증된 error-envelope의 증명 정밀도 개선이다. 이것도 효과와 비용을 측정한 뒤 채택한다.
5. 두 팔의 자기일치는 독립 물리 검증이 아니다. 최종적으로 CMFGEN 또는 ARTIS의
   **finite 값**을 재현·비교해야 한다.

## 현재 확정 현장

- binary SHA: `653f94b7f9916fdb879fab7e04f53f04a2be337f248ff2d1094f424380e78637`
- run root:
  `/gpfs/kjhan/lumina/det_convergence_single_total_mgpu/det1234_20260815T084331Z_653f94b7f991`
- A100 2장 exact CMF iteration 0은 PASS: 45 iterations,
  residual `9.6662782724980344e-09 < 1e-8`, caller time 1443.35 s.
- R6 generation 1 PASS: `Q_g=1,391,131`, `Q_E=2,180,286`, invalid/partial/unsampled 0.
- R7/A2-10은 두 endpoint trial에서 각각 첫 `UNRESOLVED_CANCELLATION`에 fail closed했다.
- population negative는 없고 floor/clamp/jitter counter는 모두 0이다.

### lower endpoint 첫 witness

- line 15, shell 10, Z=14, ion=4
- `tau=5.2162577545682245e-08`
- `eta=3.4046285166846181e-162`
- `Jbar=2.9609771047620172e-51`
- `Jbar_error_upper=2.0938704855191517e-50`
- `chi=2.6297471840772978e-08`
- absorption `7.7866212033652645e-59`
- signed rate `-9.7849550420208522e-58`
- uncertainty `6.9194822653875239e-57`

### upper endpoint 첫 witness

- line 1,279,130, shell 18, Z=16, ion=1, levels 21→60
- `tau=5.5631205972899587e-31`
- `eta=1.919589616555497e-37`
- `Jbar=1.9711586301764101e-05`
- `Jbar_error_upper=1.3423475903514383e-09`
- `chi=9.7379405430504441e-33`
- absorption `1.919502554157864e-37`
- net per sr `8.7062397632962857e-42`
- signed rate `1.0940582993553446e-40`
- uncertainty `1.6426383122528282e-40`
- cancellation condition `44095.870032182931`

`line_jbar_gaussian_discrete_shells()`는 cellwise `fine_error_upper`를 Gaussian profile로
가중 투영해 `Jbar_error_upper`를 만든다. `line_net_rate_evaluate()`는
`fma(-chi,Jbar,eta)`와 `chi*Jbar_error_upper`에서 부호를 검증한다. physical value에는
rounding 보정이 없고 error bound에만 outward rounding이 있다. envelope refinement는
`candidate_next = residual_upper + K(candidate)` 형태다. 현재 refinement 수는 8로 고정이다.

## 반드시 읽을 자료

- `docs/CURRENT_PLAN.md` 마지막 A2-10 구간
- `docs/HANDOVER_2026-08-08.md` 최신 구간
- `docs/CMFGEN_LINE_NET_DATA_CONTRACT_2026-08-09.md`
- `src/line_jbar.c`
- `src/line_net_rate.c`
- `src/cmf_error_envelope.c`
- `src/cmf_exact_sliding.c`
- `src/cmf_exact_multigpu.cu`
- `src/lumina_cmfgen.c`의 exact solve 및 Jbar 투영 구간
- `src/lumina_plasma.c`의 A2-10 판정 구간
- `scripts/submit_det_convergence_2026-08-08.sh`
- `scripts/run_det_convergence_2026-08-08.slurm`

## 답해야 할 질문

1. 위 두 witness에서 현재 bound가 sign 판정에 필요한 bound보다 각각 얼마나 큰지,
   식과 수치로 검산하라.
2. 더 많은 envelope refinement와 더 작은 solve tolerance 중 무엇을 먼저 측정해야 하며,
   둘을 어떻게 분리 실험해야 하는가?
3. 첫 실패에서 즉시 종료하는 현재 방식보다, 생산값을 바꾸지 않는 opt-in 전 셀
   unresolved census가 먼저 필요한가? 필요하다면 최소 계측 필드를 정하라.
4. 1-iteration A100x2 진단과 4-iteration flight 사이의 정확한 gate를 정하라.
5. CMFGEN/ARTIS finite 값 비교를 어느 시점에, 어떤 동일 물리량·단위·셀/선 매핑으로
   시작해야 하는가?
6. Codex가 놓친 실패 모드나 계획의 잘못된 전제를 공격하라.

## 출력 형식

- 먼저 `VERDICT: ACCEPT / REVISE / REJECT` 한 줄.
- 이어서 6~10개 번호 단계. 각 단계마다 목적, 수정/열람 파일, 산출물,
  PASS/FAIL 기준, 실패 시 다음 갈림길을 적는다.
- 마지막에 `DO NOT DO` 목록과, 가장 비싼 계산 전에 반드시 확보할 증거 3개를 적는다.
- 장황한 배경설명은 생략하고 구현자가 그대로 실행할 수 있을 만큼 구체적으로 쓴다.
