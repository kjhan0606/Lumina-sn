# Fable E 발주 — Codex D 물리 평가의 **인증**

## 입력

- `OUT_D_coevolve_vs_twoarm.md` — coevolution(ARTIS) vs two-arm(Lumina) 물리 평가
- 참조: `OUT_A`(Lumina 배선도) · `OUT_B`(ARTIS 대조) · `OUT_C`(비물리 판정) · `paper_main.tex`
- 원 소스: `./lumina/` · `./artis/`

## 과제 — 인증 또는 반박

Codex D 의 주장 하나하나에 대해 다음 셋 중 하나:

- `CERTIFIED` — 물리적으로 옳다 (근거 제시)
- `REFUTED` — 틀렸다 (무엇이 실제인지, `파일:행`)
- `OVERSTATED` — 방향은 맞으나 근거가 주장을 지탱하지 못한다 (어디까지가 지탱되는지)

**분량보다 정확도.** 확신 없으면 `UNDECIDED`.

## ★반드시 판정할 것

### (1) "두 장의 일치" 의 증명력

Codex D 는 이렇게 적었다:
> 두 장이 3% 안에서 일치했는데도 CMFGEN 대비 UV 와 trace photoionization 이
> **1.2–1.8 dex 틀린 사례가 논문에 있다.**

운전석은 1층 재작성 문서(`docs/LAYER1_REPLAN_2026-08-07.md` §2 3층)에서
**"두 팔의 일치는 고리 안에서 얻어지는 독립 잣대"** 라고 썼고, 그 위에
"3층 항목 상당수가 어긋남 지도로 직접 판정 가능해진다" 는 계획을 세웠다.

판정하라: **그 계획 전제가 유효한가?**
- 두 장의 일치가 무엇을 증명하고 **무엇을 증명하지 못하는가**를 물리로 구분하라.
- 일치가 참일 때 남는 **공통 편향(shared bias)** 의 원천을 열거하라
  (같은 원자자료·같은 opacity/source 조립·같은 이산화·같은 계약).
- 그렇다면 3층 재작성 계획을 어떻게 고쳐야 하는가.

### (2) 아키텍처 분류

Codex D 의 결론: *"two-arm 은 coevolution 의 상위호환이 아니라, MC 와 결정론의
state 소유권을 선택할 수 있는 별도 아키텍처"*. 이것을 인증/반박하라.
특히 **결정론이 state 를 소유하면 MC-state 인과 되먹임을 잃는다**는 주장.

### (3) 빠진 물리

Codex D: *"비열적 excitation 은 ARTIS 에 명시적 rate 가 있고 Lumina 에는
nonthermal ionization 만 확인된다"*. 사실인가? 사실이면 그 결손의 물리적 크기는?

### (4) 부트스트랩 대응

*"ARTIS 의 LTE 시대는 물리 시간이 실제 전진하지만 Lumina 의 bootstrap 은
한 epoch 안의 반복"* — 이 차이가 물리적으로 무엇을 뜻하는가.
`OUT_C` 에서 인정된 사슬(seed T_e → LTE 물질 상태 → solver tau → 결정론 formal
solve → 무잡음 J → SE)이 이 차이를 감당하는가.

## 규율

- 한국어. 표 중심. `[실측]`/`[추정]` 구분. `파일:행`.
- **최종 심판은 CMFGEN.** ARTIS 일치는 정답 인증이 아니다.
- 수리 코드 금지.
- 마지막에 **"1층 재작성 문서에서 고쳐야 할 것"** 을 항목으로 적어라
  (운전석이 그 문서를 고칠 것이다).
