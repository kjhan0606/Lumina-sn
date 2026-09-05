# 코드 검수 1회차 — DET-DUAL-AUTH-2 변경집합 (Fable fresh, read-only, 2026-09-05)

검수자 = Fable fresh 서브에이전트(운전석과 다른 컨텍스트). 대상 = 발주 4 산출(`scripts/stage_superlevel_b.sh` RESCALE 훅 520행 diff, `scripts/dproxy_readout/spectrum_contrast.py` 274행). 탐침 산출 `/gpfs/kjhan/lumina/dproxy_audits/readout_331050_fable/order4/fable_review_probe/`.

| 질문 | 답 |
|---|---|
| Q0 봉인 입력에 안전한가 | **YES** — 봉인 루트 읽기만, 목적지 미존재 강제, reflink 복사 후 hardlink 0 검증. RESCALE 추가분은 쓰기 연산 0 |
| Q1 발주가 범위를 좁혔/넓혔는가 | **NO** |
| Q2 base 모드 + env 1개 뿐인가 | **NO** — F-1·F-2·F-3 |
| Q3 `spectrum_contrast.py` = E-3 | **YES** — 헤더·격자 bit-동일·5대역·ln 비·median/max/min·제외 계수·JSON·표준 라이브러리, hc=12398.419843320026 eV·Å. 실물(331050 대 자기) PASS, 대역 계수 399/131/76/131/497 |
| Q4 숨은 clamp/floor | **NO** |
| Q5 음성대조 실효 | 부분 YES — 「base 가 이미 키를 가진 경우」 픽스처 부재가 F-1·F-2 를 가림 |

**폐합 차단**
- **F-1**: 봉인 R-II·331050 `input/resolved_lumina.exports:71` 에 이미 `declare -x LUMINA_NLTE_PER_ION_RESCALE="0"`. `rewrite_exports`(356–427) 는 이 키 분기가 없어 "0" 행을 통과시키고 `="1"` 을 **추가** ⟹ 중복 키 ⟹ `verify_export_delta` `EXPORT_DUPLICATE_AFTER_REWRITE` rc=70 (탐침 P1/P2). 543 행 `"…RESCALE" in base` 도 거부. **모드가 실 base 에 대해 실행 불능.**
- **F-2**: 신설 검사 684–686 `STAGED_DPROXY_BASE_HAS_RESCALE_ENV` 가 base 에 키가 없기를 요구 ⟹ 봉인 331050 루트 자체가 rc=70 (탐침 P3). **기존 `DPROXY-PROV-K42(-H200)` 스테이징 회귀.**
- **F-3**(장부 귀속): arm 런루트의 suffix·job_name·`superlevel_b_mode.txt`·`EXPECTED_EXPORT_DELTA mode=`·`OWNER.txt`·`STAGE_SEAL mode=`·post-model seal 이 전부 **참조(base) 모드**를 자칭. arm 과 base 는 exports 를 열어야만 구별.

**기재로 족함**
- 실 base 에 스테이징 시 복사 후 죽으면 READY 없는 부분 런루트가 남음(기존 동작).
- `Δlog n_upper`(338짝)는 대조기 범위 밖(사전등록도 그렇게 지정).
- **사전등록 §4·G-0·변경집합 표의 「1행 더함」 전제가 틀렸다** — 실제는 0→1 **값 변경**(sorted diff 2행). 저자(운전석)가 `env_universe.h` 등록만 보고 base exports 는 확인하지 않았다 ⟹ F-1 의 상류 원인.
- Codex 보고의 selftest 통과는 재현되나 픽스처가 실 base 와 다르다 ⟹ 「통과」가 아니라 「미시연」.

**SAFE_TO_STAGE: NO.** 정정 1회차(전건 반영) 대상.
