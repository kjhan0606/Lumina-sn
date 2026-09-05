# 발주 — d-proxy SL 수준 공변동 판독기 (read-only, 331050) — 2026-09-05

작성 = 운전석(Fable). 구현 = gpt-5.6-sol. **판정기가 아니라 판독기**다. 분기 라벨·p-value 를 내지 않는다.

## 0. 대상 (읽기 전용 — 이 경로들에 어떤 파일도 만들거나 고치지 마라)
- 런루트 `R=/gpfs/kjhan/lumina/superlevel_b_h200x2_k36/slb_20260903T080407Z_dproxy_provenance_k42_ion1_h200`
  (Slurm 331050, H200, model.rc=0, post-model seal 은 FAIL 이라 **미봉인 루트**다 — 판독 JSON 에 `sealed=false` 와 `stderr.log` sha256 을 기록)
- `R/stderr.log` (2.3 MB) — 한 번만 스트리밍
- `R/input/global_atomic/**/levels.csv` (열: atomic_number,ion_number,level_number,energy_eV,g,metastable,super_level,configuration) — `find` 로 실경로 특정
- `line_list.csv`(363 MB) 는 **읽지 마라** — 필요한 `A_ul`·`nu` 는 ROW 행에 이미 있다.

## 1. 산출 (이 디렉토리 `$PWD` 안에만)
- `readout_dproxy_sl_331050.py` — `python3 -B`, 표준 라이브러리만(numpy 허용). 인자: `--run-root R --out-dir DIR`.
- 실행 산출: `readout.json`, `pairs.csv`(짝 단위), `sl_groups.csv`(runtime SL 단위), `exclusions.csv`
- `selftest_readout.py` — 합성 stderr 로 양성 1 + 음성 최소 6(중복 짝·publication 경계 부재·비유한값·runtime SL 반복 간 표류·within_sl_frac hex/decimal 불일치·nu 대 준위 에너지 불일치). 각 음성은 기대 사유 문자열로 fail-closed 시연.

## 2. 파싱 규약
- publication 경계: `[R7][PHASE] lane=DET iter=N phase=view` 행. 그 뒤 ROW 는 publication N. 경계가 정확히 2개(N=0,1)가 아니면 `SCHEMA` 실패.
- 행: `[A2-10][LINE-SATURATION-ROW]` 의 `key=value` 토큰. 필수: phase shell line Z ion lower_global upper_global lower_level upper_level nu A_ul n_upper n_upper_hex Jbar within_sl_frac within_sl_frac_hex runtime_sl_id super_anchor_global population_source used_nlte super_mode outer_iter te_generation nlte_population_generation population_provenance_status ledger_join_key
- 수치는 **hex 가 있으면 hex** 로 읽는다(`float.fromhex`). decimal 과 hex 가 1 ULP 넘게 다르면 실패.
- `outer_iter` 가 publication N 과 다르면 실패.

## 3. 계산 (전부 기술통계 — 판정 아님)
1. publication 별 행 수·(shell,Z,ion) 분포·`population_source`·`population_provenance_status`·`super_mode`·generation 값 집합.
2. 짝: `(shell,line)` 이 양쪽에 **정확히 한 번** 있는 것만. 중복은 실패(제외 아님). 한쪽에만 있는 것은 `exclusions.csv` 에 사유와 함께.
3. 짝 불변식(전건 만족 아니면 실패로 기록하되 계속): runtime_sl_id·super_anchor_global·upper_global·lower_global·nu·A_ul 이 두 publication 에서 동일. `population_source=NLTE_COMMITTED`·`used_nlte=1`·status=OK.
4. 짝 단위: `dlogJ = ln(Jbar1/Jbar0)`, `dlogn = ln(n1/n0)`, `B_k = n_k/within_sl_frac_k` (k=0,1), `dlogB = ln(B1/B0)`. `Jbar>0, n_upper>0, 0<frac<=1` 아니면 제외+사유.
5. **D3 온도**: 각 publication 의 `[A2-10][LINE-ION-OWNER] … shell=0 T_e_K=…` 에서 T_e 집합. 두 값 문자열 동일 여부.
6. **runtime SL 군** `(shell,Z,ion,runtime_sl_id)`: 멤버 짝 수, 상이 `upper_global` 수, 상이 line 수, `within_sl_frac==1` 인 멤버 수, `dlogB` 의 max−min(상이 upper_global 간)·`dlogn` 의 max−min, 반복 간 `within_sl_frac` 변화 max|ln(frac1/frac0)|.
   요약: SL 수, 멤버≥2 인 SL 수, 상이 upper_global≥2 인 SL 수, `dlogB` spread > 1e-10 인 SL 수와 최대값(어느 SL), `dlogn` 동일 통계.
7. **덱 가중 X_s**: `w_l=(g_u/g_l)·A_ul/nu^3`, g 는 levels.csv `(Z, ion, level_number)` 조인(ROW 의 `ion` 과 levels `ion_number` 동일 규약인지 **nu 대 (E_u−E_l)/h 로 전 짝 검증**, 상대오차 > 1e-6 이면 실패). `X_s=Σw·dlogJ/Σw`(같은 line 중복 없음), `Y_s=median(dlogB)`.
8. 블록 `(shell,Z,ion)` 별: 유효 SL 수, X·Y 상이값 수, tie 비율, **midrank stratified Spearman `rho_SL`**(블록 평균 뺀 midrank 의 pooled Pearson; 상이값<2 인 블록은 제외). 참고로 짝 수준 midrank Spearman(dlogJ, dlogn) 도 병기.
9. `readout.json` 에 위 전부 + 입력 sha256 + 실행 host/time. **p-value·permutation·분기 라벨은 계산하지 않는다.**

## 4. 금지
- 런루트·덱에 쓰기 0. `/tmp` 사용 금지(노드-로컬). `.pyc` 남기지 않기(`-B`).
- floor/clamp/jitter/대체값 0. 제외는 전부 계수·사유 기록.
- 이 발주 밖 파일(저장소 `src/`, `scripts/`) 수정 금지.

## 5. 완료 보고 (stdout 마지막에)
파일 목록 + `selftest_readout.py` 결과(양성/음성 건수) + 실물 1회 실행의 요약 12행 이내(짝 수·SL 수·spread 최대·T_e 동일 여부·rho_SL·블록 수). 실물 실행은 `ssh grammar "ssh grammar-debug 'cd $PWD && python3 -B …'"` 로 돌려라. 안 되면 돌리지 말고 그렇게 보고하라.
