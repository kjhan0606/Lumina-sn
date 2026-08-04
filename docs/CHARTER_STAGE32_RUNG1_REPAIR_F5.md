# Stage 3.2 Rung 1 — F5 수리 발주 (v2 패치 REJECT)

`patches/stage32_rung1_readonly_lambda_v2.patch`
(sha256 `60bc65c172f95da5c10a07f37743b1f51b007ab9645b503aa0a7ff1d028655be`)를
운전석 감사에서 **REJECT**한다. F1·F3·F4 수리는 유지하고, 아래 F5 하나만 고쳐
`patches/stage32_rung1_readonly_lambda_v3.patch`로 낸다.

## 0. 먼저 못박는 것

- **사전등록은 고정이다.** `patches/stage32_rung1_expected_changes.txt`의 문안과
  `rho_local` 예측 구간을 바꾸지 마라.
- **1단 범위는 그대로다.** 읽기 전용 계측. 선원함수·불투명도·방출률·율·
  population·수송 상태를 바꾸지 마라. 2단 이후를 구현하지 마라.
- v2에서 통과한 F1(branch-site disposition), F3(가드 제거), F4(세대 규율)를
  깨뜨리지 마라. **수리 범위를 F5 밖으로 넓히지 마라.**
- v2를 덮어쓰지 마라. 패치를 트리에 적용하지 마라. commit 하지 마라.
- 무거운 연산·GPU·모델 런 금지.

## 1. F5 — 행 에너지가 production 조립값이 아니다

v2는 assembly 루프에 다음을 심는다(patch v2 :532, :539).

```c
cs->stage32_line_eta[(size_t)slot*NS+s] += w*Sl;
cs->stage32_boundary_eta[idx]           += w*Sl;
```

삽입 위치가 `cs->chi_line[idx] += w;` 바로 앞, 즉 **ε를 곱하기 전**이다. 그러나
production은 세 줄 뒤에서

```c
if (!emiss_b && eps_phys) {
    double el = radeq_line_eps_phys(l, ne_s, Te, tau);
    if (el < 0.0) el = 1.0;
    if (el < eps_floor) el = eps_floor;
    if (el > eps_cap)   el = eps_cap;
    eta_l = w * el * Sl;
} else {
    eta_l = w * Sl;
}
```

를 `eta_line`에 더한다(`src/lumina_cmfgen.c:1376`, writer 대응부 `:792-801`).
`emiss_b`는 A/B 진단 컨텍스트 포인터이므로 평시 NULL이고 ε 분기가 실제로 탄다.

동시에 authoritative 대조군은 ε가 **들어간** 값이다.

```c
cs->stage32_eta_pre_epay[idx] = eta_ln;
```

따라서 `closure_residual = authoritative − selected − boundary`는 **ε ≡ 1일 때만**
0이다. 실 payload에서 `LUMINA_CMFGEN_LINE_EPS_PHYS=1`이고 ε ≠ 1임은 실측으로
확정돼 있다(`/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/uv_mapsplit/`:
header `eps_phys=1`, s0 BALL에서 ε 있는 에너지 `3.86e-4` vs 없는 값 `1.30`).

**요구**: 행에 기록하는 에너지를 production이 `eta_line`에 실제로 더하는 바로 그
값으로 하라. 식을 다시 쓰지 말고 **production이 계산한 `eta_l`을 그대로 재사용**하라.
`eps_phys` 분기를 계측 쪽에서 복제하면 또 어긋난다. boundary 누적도 같다.

## 2. F5 — fixture가 결함을 가리는 구조를 없애라

v2 fixture의 스텁은 다음이다(patch v2 :681-683).

```c
double radeq_line_eps_phys(int l,double n,double T,double t) {
    (void)l;(void)n;(void)T;(void)t; return -1.0;
}
```

production은 이 음수를 `if (el < 0.0) el = 1.0;`으로 받는다. 즉 **fixture에서는 ε가
항상 정확히 1**이라 위 불일치가 발현될 수 없다. 보고된 `closure_residual = 0.0`은
그래서 나온 값이다. 이는 v1 리뷰가 F2에서 이미 규탄한 "fixture가 비-production
식으로 불일치를 가리는 구조"와 같은 부류가 자리만 옮긴 것이다.

**요구**: 스텁이 **선마다 다른, 1이 아닌 유한 ε**를 내도록 하라. `eps_floor`/`eps_cap`
경계에 걸리는 값과 그 사이의 값을 모두 포함하라.

**요구**: `eps_phys=0` 경로도 fixture로 덮어라. 그 경로에서는 `eta_l = w*Sl`이므로
계측과 production이 일치해야 한다.

## 3. 음성 대조 (필수)

이 저장소의 게이트는 **주입 결함으로 FAIL을 시연해야 PASS 자격**이 있다. v2의
`closure_residual = 0.0`은 결함이 발현 불가능한 fixture에서 나온 값이므로 게이트로
인정하지 않는다.

**요구**: ε가 1이 아닌 fixture 위에서 다음 두 결함을 각각 심고 checker가 **FAIL하는
것을 실제 출력으로 시연**하라.

1. 누적을 `w*Sl`로 되돌리는 결함(= v2의 현 상태). closure가 FAIL해야 한다.
2. authoritative 쪽만 ε를 빼는 결함. 역시 FAIL해야 한다.

두 번째가 필요한 이유는, 양쪽에서 동시에 ε를 빼면 closure는 닫히지만 측정량이
production이 아니게 되기 때문이다. 이 경우도 잡히는지 밝히고, 못 잡으면 못 잡는다고
보고하라 — 잡히도록 억지 가드를 넣지 마라.

## 4. 파급 확인

행 에너지는 사전등록 판독(`tau>=100` 에너지의 90% 이상, 에너지 가중 `rho_local`
중앙값)의 **가중치**다. ε는 선마다 다르므로 이 결함은 균일 배율이 아니라 census를
지배하는 선의 구성을 바꾼다. 수리 전후로 fixture의 가중 판독이 어떻게 달라지는지
수치로 제시하라.

## 5. 산출물과 규율

- `patches/stage32_rung1_readonly_lambda_v3.patch`. v2 sha256과 v3 sha256을 모두 보고.
- clamp/floor/cap/fallback/대체값을 **새로** 넣지 마라. production이 이미 적용한
  `eps_floor`/`eps_cap`은 production의 것이지 계측의 것이 아니다 — 계측은 production이
  낸 값을 읽을 뿐 다시 자르지 마라. 값이 정의되지 않으면 정의되지 않는다고 기록하고
  중단하라.
- `make selftest_stage32_rung1`과 CPU `make -B lumina`를 격리 복사본에서 수행하고
  결과를 보고하라. **실제 작업 트리에서 빌드하지 마라** (v2 작업에서 작업공간의
  `lumina` 바이너리를 rebuild한 이탈이 있었다).
- F5 각 항목에 대해 무엇을 어떻게 고쳤고 **어느 시험이 그것을 잡는가**를 파일:줄로
  제시하라. "고쳤다"만으로는 접수하지 않는다.
- 전체 보고는 `docs/CODEX_STAGE32_RUNG1_REPAIR_F5.md`.
