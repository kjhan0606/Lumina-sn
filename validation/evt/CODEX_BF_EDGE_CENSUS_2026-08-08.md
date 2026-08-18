# MC-EVT BF edge census — 2026-08-08

판정 대상은 MC-EVT 실측에 사용한 동일 덱
`data/tardis_reference_toy06_19p48d_sivcaiv_active`다.

실행:

```text
python3 scripts/census_bf_edges_below_grid.py \
  --ref-dir data/tardis_reference_toy06_19p48d_sivcaiv_active
```

결과:

```text
positive=24542 above=23835 below_or_at_all=707
default_active=707 neutral_option=0 cmfgen=707 kramers_fallback=0
[MC-EVT][BF-EDGE-CENSUS][BLOCKED]
active_below_or_at_nu_min=707 action=REOPEN_SH_GRID rc=3
```

가장 낮은 문턱 witness는 `global=1535, Z=20, ion=1, level=61`,
`nu_edge=5.84852771e13 Hz` (`lambda=51259.4747 Å`)다. 이 준위는 CMFGEN
단면적을 가지며 기본 생산 경로에서 활성이다.

따라서 Fable이 허용한 조건부 정책인 “활성 BF edge가 하나도 없을 때만 격자밖
BF 사건 측도를 `EXACT_ZERO`로 선언”의 전제가 성립하지 않는다. MC 접근자의
`OUT_OF_GRID`를 0으로 바꾸지 않는다. `NLTE_NU_MIN=1.5e14 Hz` 아래를 포함하도록
SH-GRID를 재개방해야 한다.

자가검사 `python3 scripts/census_bf_edges_below_grid.py --selftest`는 양의 대조
1개, 중성 옵션 대조 1개, 격자 안 대조 1개를 사용하며 rc 0으로 통과했다.
