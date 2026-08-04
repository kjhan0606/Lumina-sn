감사 결과는 **반려**입니다. 구현·파일 수정은 하지 않았습니다.

다만 시스템의 read-only sandbox가 패치를 거부했습니다.

```text
patch rejected: writing is blocked by read-only sandbox
artifact=ABSENT
```

따라서 `docs/CODEX_AUDIT_ORDER_CD.md`는 생성하지 못했습니다.

핵심 반박 사항:

- 권위 사슬 단절: `mk_sn_hydro.py:2-3`은 지정된 1시간 StaNdaRT 파일이 아니라 별도 19.48일/807-zone 파일을 읽고, `_jnu4/SN_HYDRO_DATA`가 아닌 sibling 디렉터리에 씁니다.
- 전체 921개 `abundances.csv` 중 533개가 `ΣX=1`을 벗어납니다. DDC15 생성기는 untracked species를 inert mass로 남겨 합이 1 미만이어야 한다고 명시하므로 D1-6 치명 처리는 부당합니다.
- 정본은 abundance 8원소, atom masses 15원소입니다. 누락 Z는 `12,13,21,22,23,24,25`입니다. 따라서 D1-5를 적용하면 G9 정상 덱도 FAIL합니다.
- G1–G6은 현재 결함을 잡지만 G7은 비회귀 게이트라 결함 입력에서도 PASS합니다. G8은 실행 조건 6개와 주석 수정 1개를 잘못 “7개 fail-closed”로 묶었습니다.
- `slurm_*.sh` 직접 writer는 “30여 곳”이 아니라 138개입니다.
- C2 동일 수리 대상은 다음 네 파일입니다.

  - `deck_regen_fullcov_driver.py`
  - `deck_regen_r1_vintage_driver.py`
  - `deck_regen_r4_ftos_driver.py`
  - `deck_regen_r4_offcontrol_driver.py`

- 30개 필드를 50회 읽는 현재 루프는 실제로 셸 30–49에 모두 `0.0`을 기록했습니다.

```text
shells_30_49=[0.0 × 20]
pointer_advances_28_33=[4,4,0,0,0,0]
all_outer_zero=True
```

발주서가 놓친 무증상 경로도 있습니다: `NaN/Inf/garbage/음수/후행문자`, 중복 Z, 잘못된 shell header 순서, 8192바이트 초과 행, G7 사전 해시 미봉인, 그리고 E의 잘못된 재격자화·붕괴 결과가 모든 동일성 게이트를 통과하는 경로입니다.

**반려**

쓰기 권한이 제공되어야 완성된 감사문을 지정 경로에 산출할 수 있습니다.