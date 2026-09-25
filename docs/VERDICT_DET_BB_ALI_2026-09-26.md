# 판정 — DET-BB-ALI (2026-09-26)

계약 = `docs/PREREG_DET_BB_ALI_2026-09-25.md`(개정 1, 721ffba). 판정자 = 운전자(개정25). 런 = job 404407(syn09, h100×2, 44,093 s; 첫 제출 404249 는 syn102 DRAIN 으로 무출력 취소). 산출 루트 `/gpfs/kjhan/lumina/cmfgen_map/bb_ali_20260925/run1`, 판독기 `scripts/bb_ali_judge.py`(동결 sha = `bb_ali_20260925/JUDGE_FROZEN.txt`, 실행 전 `sha256sum -c` OK).

## §1. 분기 = **X (형식)** · 판독 = NRB-FLAT 상당 · user 결정: 재실행 없이 폐합

| 게이트 | 결과 | 근거 |
|---|---|---|
| G-ID | PASS | 델타 = {+`LUMINA_BB_NRB=1`, `LUMINA_A210_SPRODUCER_CAPTURE` 0→1}; 소스 차이 = 정확히 4경로; 승인=빌드=manifest=트리; 바이너리 4중 일치 `19391e34…`; referents sha `f27a9a95…` (`bb_ali_result.json`) |
| G-FIRE | PASS | `[BB-NRB]` SEED 1행(`operator=1 armed=0`) + k=0..5 `operator=2 armed=1 captured=1` 각 1행; `SPRODUCER-CAPTURE status=CAPTURED` 6행; `[BB-NRB][FATAL]` 0행 |
| G-RUN | **FAIL** | footer `MODEL_FAILED`, model_rc=1. 원인 = 6반복 완주 뒤 **계약 밖 부가 계측** `[XSHELL][BLOCKED] reason=MEASUREMENT_GATE_FAILURE`(gates=111100; 398939 계승 `LUMINA_XSHELL_RESPONSE=1`, `lumina_cmfgen.c`:9202). 그 밖 G-RUN 항목(ion k=0..5·spectral/shell k=3..5·금지 태그 5종 0행·Γ 72행 완전·gen max=Σ)은 전부 충족(`readout_run1.log` 의 G-RUN problems = footer 1건뿐) |

⟹ §5 규칙상 분기 = **X(미시연)**. §5 X 의 「다음」(같은 델타 1회 재실행)은 **user 결정(2026-09-26 「닫읍시다」)으로 집행하지 않는다**: 재실행은 같은 XSHELL 게이트에 다시 걸리거나(델타 불변), 게이트를 끄려면 exports 델타가 계약 밖으로 바뀐다. 아래 판독은 완비된 물리 산출에서 동결 판독기 함수로 계산한 **판독이며 판정이 아니다**.

## §2. 판독 (코어 J_core 12셸, k=4,5 기하평균, 제어 = 398939 같은 반복 지수; `bb_ali_20260925/readout_run1.log`)

| 통계 | 값 | §5 문턱 |
|---|---|---|
| med δΓ (S II gamma_phot) | **−0.059 dex** | ≤ −0.3 (field_moved) |
| med δx (S III/S II) | **+0.006 dex** | ≤ −0.5 (x_dropped) |
| n_both | 2 / 12 | ≥ 6 |
| up | 없음 | ≥ +0.3 / +0.5 |
| (지지) med δS 100–2000 Å | **−0.183 dex** | ≤ −0.15 (source_moved) |

`classify` 적용 시 NRB-FLAT. 셸별: sh19–21 δx −0.76~−1.22·δΓ −0.29~−0.36(유일한 동시 하강); sh12–15 δx **+0.32~+0.40**(상승)·δΓ −0.03~−0.09; sh10–11·22–24 |δ| ≲ 0.25. 모든 코어 셸이 개입 후에도 공개값 위 0.5–2.3 dex(예 sh24 x 166 대 x_pub 0.744; sh20 1.2 대 0.392).

**읽을 수 있는 것**: 이 솔버·6반복 창에서 bb 율을 CMFGEN 비겹침 Sobolev 순율 형태로 바꾸면 FUV 대역 소스는 문턱 이상 내려가지만(−0.18 dex), 코어 S II 광이온화율은 중앙값 −0.06 dex 만 움직이고 단계비는 중앙값 불변이다. 두꺼운 FUV 선의 Λ-정체는 과이온화 격차(1–2.3 dex)를 설명하는 크기가 아니다 — 사전 기대(Q-LEAN 경량 감사: NULL 쪽)와 일치.
**말하지 않는 것**(계약 §1·§6): 원천 부재의 기각(판독은 FLAT 상당일 뿐 형식 분기는 X) · J_cont(겹침·연속체) 원천 판정 · 공개 재현 · sh12–15 상승의 기전.

## §3. 폐합·다음

- DET-BB-ALI 폐합(X, 판독 FLAT 상당). user 지시(09-25)에 따라 **Λ-정체 방향 파기는 여기서 끝낸다**.
- 주선 = CMFGEN 잣대 복구(`/gpfs/kjhan/cmfgen_runs/toy06_19p48d_ntedep{,_cold}`, 기한 2026-10-02).
- 지지 판독 `bb_ali_thick_lines.py` 는 형식 분기가 X 라 실행하지 않았다(판독 결론에 불필요).

## §4. 분장 장부 (장부 정본 = 사전등록 집행 기록; 여기는 이 판정 단계 칸)

| 단계 | 규약상 | 실제 | 위반 |
|---|---|---|---|
| 기안 | Fable (25-b) | grok 대행 초안 → Fable 기안·개정 1 | 없음(대행 초안은 폐기급 결함으로 Fable 이 대체) |
| 사전감사 | astra 1회 + Q-GOAL/Q-LEAN | astra 1회(차단 4) + grok Q-GOAL/Q-LEAN(차단 0) | 경량 감사 1회를 Opus 5.5 에 오배정(참고로만 보존) |
| 구현·selftest | 운전자 (25-o) | 운전자 | 없음 |
| 코드 감사 | Fable 1회 | Fable 1회(차단 0) | 없음 |
| 판정 | 운전자 | 운전자 | 없음 |
| 종료감사 | astra 한 겹 (25-q) | (이 문서 커밋 후 발주) | — |
