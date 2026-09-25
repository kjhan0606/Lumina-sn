# 판정 — DET-BB-ALI (2026-09-26)

계약 = `docs/PREREG_DET_BB_ALI_2026-09-25.md`(개정 1, 721ffba). 판정자 = 운전자(개정25). 런 = job 404407(syn09, h100×2, 44,093 s; 첫 제출 404249 는 syn102 DRAIN 으로 무출력 취소). 산출 루트 `/gpfs/kjhan/lumina/cmfgen_map/bb_ali_20260925/run1`, 판독기 `scripts/bb_ali_judge.py`(동결 sha = `bb_ali_20260925/JUDGE_FROZEN.txt`, 실행 전 `sha256sum -c` OK).

## §1. 분기 = **X (형식)** · 판독 = NRB-FLAT 상당 · user 결정: 재실행 없이 폐합

| 게이트 | 결과 | 근거 |
|---|---|---|
| G-ID | PASS | 델타 = {+`LUMINA_BB_NRB=1`, `LUMINA_A210_SPRODUCER_CAPTURE` 0→1}; 소스 차이 = 정확히 4경로; 승인=빌드=manifest=트리; 바이너리 4중 일치 `19391e34…`; referents sha `f27a9a95…` (`bb_ali_result.json`) |
| G-FIRE | PASS | `[BB-NRB]` SEED 1행(`operator=1 armed=0`) + k=0..5 `operator=2 armed=1 captured=1` 각 1행; `SPRODUCER-CAPTURE status=CAPTURED` 6행; `[BB-NRB][FATAL]` 0행 |
| G-RUN | **FAIL** | footer `MODEL_FAILED`, model_rc=1. 원인 = 6반복 완주 뒤 `[XSHELL][BLOCKED] reason=MEASUREMENT_GATE_FAILURE`(gates=111100 = G-RATE·G-DIR 실패; `lumina_cmfgen.c`:9202). ⚠**계약 밖이 아니라 계약 자체의 결함**: 계약 §2(i) 가 `LUMINA_XSHELL_RESPONSE=1` 유지를 「캡처만, 398939 동형」으로 오기하며 요구했고 G-ID 가 그 존재를 강제했는데, XSHELL 의 G-RATE 는 398939 sh24 Γ 상수(`gamma_census6/7` = 2.824526205e-2·3.874831249e-2, `lumina_cmfgen.c`:8676–8679, 상대 1e-6)를 핀으로 박고 G-DIR 은 dx24 < 0 을 요구한다(:8680–8683). 이 런은 sh24 Γ +8 %(0.03049)·dx24 = +14.9(`run1/work/lumina_xshell_response.json`)라 두 게이트가 결정론적으로 떨어진다 — NRB 가 sh24 를 조금이라도 움직이면 G-RUN 의 「SUCCESS·model_rc 0」은 계약의 exports 아래에서 **구성상 도달 불가**였다. 기안(Fable)·사전감사(astra)·기계 게이트가 모두 놓친 계약 결함이며 트리거 계수(계약 결함 중단)에 1건 기재한다. 그 밖 G-RUN 항목(ion k=0..5·spectral/shell k=3..5·금지 태그 5종 0행·Γ 72행 완전·gen max=Σ)은 전부 충족(`readout_run1.log` 의 G-RUN problems = footer 1건뿐) |

⟹ §5 규칙상 분기 = **X(미시연)**. §5 X 의 「다음」(같은 델타 1회 재실행)은 **user 결정(2026-09-26 「닫읍시다」)으로 집행하지 않는다**: 재실행은 같은 XSHELL 핀 상수에 결정론적으로 다시 걸리고(델타 불변), 게이트를 끄거나 핀을 풀려면 exports 델타(§2(i)·G-ID) 또는 소스 4경로(G-ID)가 바뀌어 §5 X 의 「계측·계약 위반만 수리」가 아니라 **계약 개정**이 된다 — user 는 개정·재실행 대신 폐합을 택했다. 아래 판독은 완비된 물리 산출에서 동결 판독기 함수로 계산한 **판독이며 판정이 아니다**.

## §2. 판독 (코어 J_core 12셸, k=4,5 기하평균, 제어 = 398939 같은 반복 지수; `bb_ali_20260925/readout_run1.log`)

| 통계 | 값 | §5 문턱 |
|---|---|---|
| med δΓ (S II gamma_phot) | **−0.059 dex** | ≤ −0.3 (field_moved) |
| med δx (S III/S II) | **+0.006 dex** | ≤ −0.5 (x_dropped) |
| n_both | 2 / 12 | ≥ 6 |
| up | 없음 | ≥ +0.3 / +0.5 |
| (지지) med δS 100–2000 Å | **−0.183 dex** | ≤ −0.15 (source_moved) |

`classify` 적용 시 NRB-FLAT. 셸별: sh19–21 δx −0.76~−1.22·δΓ −0.29~−0.36(유일한 동시 하강); sh12–15 δx **+0.32~+0.40**(상승)·δΓ −0.03~−0.09; sh10–11·22–24 |δ| ≲ 0.25. 모든 코어 셸이 개입 후에도 공개값 위 0.5–2.3 dex(예 sh24 x 166 대 x_pub 0.744; sh20 1.2 대 0.392).

**읽을 수 있는 것**: 이 솔버·6반복 창에서 bb 율을 CMFGEN 비겹침 Sobolev 순율 형태로 바꾸면 FUV 대역 소스는 문턱 이상 내려가지만(−0.18 dex), 코어 S II 광이온화율은 중앙값 −0.06 dex 만 움직이고 단계비는 중앙값 불변이다. 응답은 셸에 따라 갈린다 — sh19–21 은 x 가 −0.76~−1.22 dex 내려 그 셸 gap(1.67–1.92 dex)의 45–70 % 에 해당하고 Γ 도 −0.29~−0.36 이나(sh19 는 k=5 에 −0.34 로 되돌아옴; 궤적 표 = `readout_run1.log`), sh12–15 는 x 가 +0.32~+0.40 오르며, 나머지는 문턱 안이다. 코어 중앙값이 FLAT 상당인 것은 이 갈림의 합이지 「Λ-정체가 격차를 설명하는 크기가 아니다」는 뜻이 아니다 — 형식 분기가 X 이고, FLAT 이라도 계약 §5·§6 상 원천 부재의 기각이 아니며(6반복 창·상쇄 되먹임·남은 J_cont 자기 기여·Si·SRCE_CHK 셀 제외), 두꺼운 선 귀속 지지 열(`bb_ali_thick_lines.py`)은 실행하지 않았다. 「NULL 쪽」 사전 기대는 운전자 메모(09-25 계획)의 것이고, 계약 등록 예측은 「H 참이면 DOWN」이며 grok Q-GOAL/Q-LEAN 산출에는 NULL 기대가 없다.
**말하지 않는 것**(계약 §1·§6): 원천 부재의 기각(판독은 FLAT 상당일 뿐 형식 분기는 X) · J_cont(겹침·연속체) 원천 판정 · 공개 재현 · sh12–15 상승의 기전.

## §3. 폐합·다음

- DET-BB-ALI 폐합(X, 판독 FLAT 상당). user 지시(09-25)에 따라 **Λ-정체 방향 파기는 여기서 끝낸다**.
- 주선 = CMFGEN 잣대 복구(`/gpfs/kjhan/cmfgen_runs/toy06_19p48d_ntedep{,_cold}`, 기한 2026-10-02).
- 지지 판독 `bb_ali_thick_lines.py` 는 형식 분기가 X 라 실행하지 않았다(판독 결론에 불필요).

## §4. 분장 장부 (장부 정본 = 사전등록 집행 기록 — ⚠집행 6–10(구현·코드 감사·빌드·제출·판독·판정·종료감사)은 사전등록에 **미기입**(집행 1–5·개정 1 까지만, PREREG:122–127); 기입 전까지 이 표가 유일한 장부다)

| 단계 | 규약상 | 실제 | 위반 |
|---|---|---|---|
| 기안 | Fable (25-b) | grok 대행 초안 → Fable 기안·개정 1 | 없음(대행 초안은 폐기급 결함으로 Fable 이 대체) |
| 사전감사 | astra 1회 + Q-GOAL/Q-LEAN | astra 1회(차단 4) + grok Q-GOAL/Q-LEAN(차단 0) | 경량 감사 1회를 Opus 5.5 에 오배정(참고로만 보존) |
| 구현·selftest | 운전자 (25-o) | 운전자 | 없음 |
| 코드 감사 | Fable 1회 | Fable 1회(차단 0) | 없음 |
| 판정 | 운전자 | 운전자 | 없음 |
| 종료감사 | astra 한 겹 (25-q) | **Fable fresh 대행 1회**(astra Codex 인증 만료 — `bb_ali_20260925/closeout_astra_out.txt` 0바이트 = 미시연; user 09-26 결정, 재감사 없음) | **계약 기안자 = 종료감사자**(자기 계약 감사; 개정24-b 가 없앤 형태) — user 결정으로 수용, 기재 |
| 판독 | 운전자 | 운전자(`readout_run1.sh` — 동결 판독기 함수 호출; 판독기 `full` 은 X 에서 δ 표를 쓰지 않아 result.json 에 δ 표 없음) | 없음 |
