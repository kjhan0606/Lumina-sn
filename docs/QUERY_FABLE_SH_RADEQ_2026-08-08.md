# Fable 질의 — SH-RADEQ 기준 방정식과 선 방출 표현

날짜: 2026-08-08

응답 수신: `docs/FABLE_VERDICT_SH_RADEQ_2026-08-08.md` — `VERDICT=REVISE`.

Codex가 정적으로 확정한 사실은 다음과 같습니다.

1. 기준 CMFGEN 덱은 `FIX_T=T`, `USE_EHB`/`COMP_EHB` 미지정(기본 false)입니다.
   그러므로 `STEQ_T = integral(chi_noscat*J-eta_noscat) dnu`는 계산하지만 그 근으로
   온도를 풀지 않았습니다.
2. A2-10 사전등록 장부는 photoionization을 `(1-nu0/nu)` 초과 에너지로 정의하고
   collisional/nonthermal/adiabatic을 따로 둡니다. 의미상 CMFGEN EHB에 더 가깝습니다.
3. CMFGEN line eta는 `n_upper*A_ul`에서 직접 만듭니다. Lumina는 expansion-opacity
   `chi_eff = nu*(1-exp(-tau))/(c*t*dnu)`에 `S_line`을 곱합니다. 정상 cell에서는 이는
   `n_upper*A_ul*h*nu*beta_esc/(4*pi*dnu)`와 동등해야 하지만, source 미발행 또는
   exact cancellation에서는 division 표현이 깨집니다.
4. Lumina 단열항은 현재 전자 병진항 `3*n_e*k*T/t`뿐입니다. CMFGEN은 원자+전자,
   전자분율 구배, 여기·전리 내부에너지 구배를 포함합니다.
5. 현재 committed CPU opacity는 메모리 절약을 위해 line slab을 `n_lines=0`으로 두고,
   A2-09가 가변 `OpacityState`의 tau/source 배열을 다시 읽습니다. 선택한 직접 방출식에
   필요한 최소 입력만 immutable generation-bound view로 바꿀 예정입니다. 따라서 Q2에서
   `beta_esc`가 필요하다면 어떤 tau 정의와 generation을 소유해야 하는지도 명시해 주십시오.

## Q1. A2-10의 canonical 방정식

아래 중 어느 것을 canonical solve로 고정해야 합니까?

- **A. EHB_THERMAL**: 현 사전등록의 초과에너지 photoheating/collisional ledger를 유지.
  CMFGEN은 `COMP_EHB=T, USE_EHB=F` 진단 실행으로 먼저 항을 대조하고, 최종에는
  `USE_EHB=T` free-T truth를 별도 생산.
- **B. RE_INTEGRAL**: 기준 CMFGEN의 `STEQ_T`와 동일한 `chi*J-eta` 전체 에너지식을
  canonical solve로 변경.
- **C. DUAL**: 한 식만 온도 producer로 명시하고 다른 식은 독립 closure diagnostic으로
  유지. 어느 식이 producer인지도 지정해 주십시오.

## Q2. Sobolev/expansion line emission

Lumina의 A2-08 line absorption이 이미 `(1-exp(-tau))` effective operator를 쓰는 조건에서,
A2-09 line eta의 canonical bin 적분은 어느 것입니까?

- **A.** `n_upper*A_ul*h*nu*beta_esc(tau)/(4*pi*dnu)`
- **B.** `n_upper*A_ul*h*nu/(4*pi*dnu)`
- **C.** 다른 식(프로파일/escape ownership을 명시해 주십시오)

또한 `tau=0`이 population cancellation인데 `n_upper>0`인 cell을 exact-zero emission으로
둘 수 있는지, 아니면 직접식으로 유한 방출을 보존해야 하는지 판정 부탁드립니다.

## Q3. 단열항의 현재 단계 최소 범위

CMFGEN 재현을 위해 SH-RADEQ에서 즉시 다음 중 어디까지 구현해야 합니까?

- **A.** CMFGEN `EVAL_ADIABATIC_V3`의 원자+전자+내부에너지/구배 전량
- **B.** homologous SN에서 식을 엄밀히 축약한 항(축약식 제시 필요)
- **C.** 현재 전자 병진항을 임시 유지하되 gate를 `BLOCKED_INCOMPLETE_ADIABATIC`으로 고정

## 요청 답변 형식

```text
Q1 = A|B|C — 근거
Q2 = A|B|C — tau=0 cancellation 처리 포함
Q3 = A|B|C — 허용되는 임시 상태 포함
VERDICT = PROCEED | REVISE | BLOCK
```
