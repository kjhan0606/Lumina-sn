# Fable 중요 판정 질의 — 완전 CMFGEN 단열항과 all-shell trial 경계

이 질의는 flight 허가가 아니라, 잘못된 scalar 구현을 막기 위한 중요 물리/구조 판정이다.
문서 주장만 따르지 말고 아래 공식 CMFGEN 파일과 Lumina 실물을 직접 확인해 달라.

## 읽을 실물

- `/tmp/cmfgen_ref_20260808.OGALJ7/extract/cur_cmf/new_main/subs/eval_adiabatic_v3.f`
- `/tmp/cmfgen_ref_20260808.OGALJ7/extract/cur_cmf/new_main/cmfgen_sub.f:2915-2965`
- `docs/CMFGEN_ADIABATIC_V3_MAPPING_2026-08-08.md`
- `src/radeq_publication.h`
- `src/radeq_publication.c`
- `src/lumina_plasma.c:12695-12880`
- `docs/SPEC_A2_09_10_V1.md:390-470`

공식 배포 URL은
`https://sites.pitt.edu/~hillier/cmfgen_files/cur_cmf_18jun25.tar.gz`다.

## Codex가 실측한 구조

1. `STEQ_T`용 main call은 `MEAN_EN=INT_EN`, `SUM_EN=TOT_ENERGY`다.
2. cgs signed cooling은 다음 네 항으로 대응된다.

```text
(3/2)(n_atom+n_e) k v dT/dr
+ (n_atom+n_e) k T div(v)
+ (3/2)n_atom k T v d(n_e/n_atom)/dr
+ n_atom v d(u_int/atom)/dr
```

3. homologous이면 `div(v)=3/t`다.
4. `u_int/atom`은 neutral ground 기준의 누적 전리+여기 에너지다.
5. 현재 callback은 한 shell의 trial T만 받고 committed neighbor/population/eta를 읽는다.
   완전항은 인접 shell의 trial T, electron fraction, internal energy가 필요하다.
6. gradient 때문에 signed total은 음수가 될 수 있으나 현 ledger에는 cooling slot만 있다.

## 제안 구현

- 먼저 모든 shell의 `r,v,T,n_atom,n_e,u_int/atom`을 받아 네 signed component를 내는
  pure vector producer를 구현하고 known-answer/negative control로 봉인한다.
- production A2-10은 그대로 `RADEQ_INCOMPLETE_ADIABATIC`으로 막아 둔다.
- 다음 단계에서 scalar callback을 all-shell private atomic candidate callback으로 교체한다.
  trial마다 population/partition/n_e → opacity/emissivity → vector adiabatic → 전 shell
  residual을 같은 token으로 평가하고, 전 shell 성공 뒤 한 번만 게시한다.
- signed total은 `max(q,0)` cooling과 `max(-q,0)` heating으로 나누고 raw 네 component를
  보존한다. 이를 위해 A2-10 term schema를 version-up한다.
- Lumina cell center는 `0.5*(inner+outer)`로 두고, CMFGEN 배열 방향을 뒤집은 one-sided
  stencil(`s=0 -> 1`, `s>0 -> s-1`)을 쓴다.

## 판정 요청

다음 형식으로 짧고 명확하게 판정해 달라.

```text
PHYSICS_MAPPING = ACCEPT | REVISE
ALL_SHELL_TRANSACTION = REQUIRED | NOT_REQUIRED
SIGNED_LEDGER_SPLIT = REQUIRED | NOT_REQUIRED
STENCIL_AND_CENTER = ACCEPT | REVISE
IMPLEMENT_NOW = VECTOR_PRODUCER_ONLY | OTHER
```

`REVISE`나 `OTHER`면 반드시 공식 소스 또는 Lumina 계약의 구체적 줄과 함께 대체식을
제시해 달라. 특히 공식 파일의 diagnostic `COL_EN` 혼재를 적분형 `STEQ_T` 주경로와
구별해 달라.
