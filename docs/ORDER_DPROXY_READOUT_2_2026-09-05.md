# 발주 2 — 반상태 R-II 대조 (runtime SL 사상 이식) — 2026-09-05

작성 = 운전석(Fable). 구현 = gpt-5.6-sol. 발주 1 의 `readout_dproxy_sl_331050.py` 를 **확장**한다(새 스크립트 금지). 판정 아님.

## 0. 근거
runtime SL 사상(`fl_to_super`)은 `LUMINA_SUPER_CUTOFF=100` 으로 원자 로드 시 결정되며 `LUMINA_SUPER_LEVELS` 값과 무관하다(331050·R-II 둘 다 `13015 FL -> 1173 SL`). 따라서 331050 provenance 행의 `upper_global → (runtime_sl_id, super_anchor_global)` 표는 같은 덱을 쓴 R-II 에도 유효하다. R-II 는 `super_mode=0`(`within_sl_frac ≡ 1`) 이므로 `B = n_upper` 다.

## 1. 대상 (읽기 전용)
- 대조 루트 `RII=/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/rii_20260826T134337Z_ion1` — **봉인 루트**. 분석 전후 `RII_RUN_SEAL.sha` 전건 검증(`sha256sum -c`), 하나라도 어긋나면 중단.
- 사상 원천 = 발주 1 의 331050 판독(`pairs.csv` 또는 새 `sl_map.csv`: upper_global, runtime_sl_id, super_anchor_global, within_sl_frac_hex).

## 2. 요구
1. `--sl-map sl_map.csv` 옵션: ROW 에 provenance 필드가 없을 때 upper_global 로 runtime_sl_id·anchor 를 이식. 이식 불가(사상에 없는 upper_global)는 제외+사유 계수. `within_sl_frac` 은 **이식하지 않는다** — `--semi-state` 옵션에서 frac≡1 로 두고 그 사실을 JSON 에 기록.
2. 331050 자체에 `--sl-map` 을 적용했을 때 provenance 필드와 **전건 일치**해야 한다(자기 검증, 불일치 0 아니면 실패).
3. R-II 992 짝 전체와, 331050 과 공통인 231 짝 부분집합 두 가지로 §3 통계(발주 1 과 동일 정의)를 낸다. 산출 `readout_rii_all.json`, `readout_rii_common231.json`, 각 `pairs_*.csv`, `sl_groups_*.csv`.
4. 공통 231 짝에 대해 331050(승격) 대 R-II(반상태)의 짝 단위 `dlogJ`·`dlogn` 을 나란히 담은 `contrast_common231.csv` (line, upper_global, runtime_sl_id, dlogJ_prom, dlogJ_semi, dlogn_prom, dlogn_semi).
5. selftest 에 음성 2 추가: 사상 누락 upper_global, 사상 표와 provenance 불일치.

## 3. 금지·보고
발주 1 §4 와 같다. 보고는 R-II 전체·공통 231 각각의 SL 수·spread 최대·rho_SL·블록 수, 그리고 seal 검증 결과 두 번(전·후).
