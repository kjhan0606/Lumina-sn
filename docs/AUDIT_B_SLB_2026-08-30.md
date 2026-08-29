핵심 수치와 명령은 모두 독립 재현됐으며, 폐합 차단 사유는 없습니다.

| 항목 | 결과 | 독립 재현값·근거 | 계급 |
|---|---|---|---|
| R1 `g_fail=1`·`SLB-F` | 재현됨 | B-2 stderr 1,252행. `[R7][FATAL] iter=0 rc=4` 1건, iter 1은 0건, `summary complete=1` 0건, `complete=0` 1건. `model.rc=1`. 계약 [G-4](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/RUNG_SUPERLEVEL_B_2026-08-28.md:359) 조건 미충족은 이름 있는 G-4 FAIL이고, [§11](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/RUNG_SUPERLEVEL_B_2026-08-28.md:406)의 “어느 G-* 이름 있는 FAIL”에 따라 `g_fail=1 → SLB-F` 적용이 맞음 | 지적 없음 |
| R2 사망 행 수치 | 재현됨 | [stderr:1246](/gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260829T101641Z_probe_ion1/stderr.log:1246): `line=1370680 shell=33 Z=20 ion=1(II) lower=0 upper=3`, `η=1.8649388358084121e-12`, `χJ̄=1.8649390292707385e-12`. 재계산 `S/J̄−1 = −1.03736542245e−7`, `σJ̄/J̄ = 1.45761196807e−7`, `uncertainty/|rate| = 1.40510945991`. `srce_chk=clamp=floor=0` | 지적 없음 |
| R2 정체·중단·소스 | 재현됨 | 덱 행은 Ca II H 계열, `λ=3969.591342 Å`. `LINE-NET-CELL-BLOCKED`는 전체 로그에서 정확 1행이며 1246행 뒤 1251행에서 R7 FATAL, 1252행 종료. 요청된 [line_net_rate.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/line_net_rate.c:79), [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15021), [population_contract.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/population_contract.c:199) 인용 범위 전건 실존·내용 일치 | 지적 없음 |
| R2 Ca II 62준위 무럼프 | 재현됨 | 봉인 덱에서 Ca II는 정확 62준위, level 0–61, `level≥100` 0개. 소스의 `super=min(level_num,K)` 규칙([lumina_atomic.c:1458](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:1458))과 `K=100`을 적용하면 전부 singleton이므로 무럼프 | 지적 없음 |
| R3 대조 계수 | 재현됨 | `RADEQ_SIGN_MISMATCH / UNRESOLVED_CANCELLATION / R7 FATAL`: R-II `0/0/0`(449,757행), R-III `0/0/0`(377,845행), B-2 `1/2/1`(1,252행) | 지적 없음 |
| R4 B-1 상호배타 | 재현됨 | 실측 capture 두 노브: B-1 `0/0`, B-2 `1/1`, ZO `0/0`, R-II `1/1`. B-1은 실제 `A210_Z2A_PROBE`, target ion 1. 계약 §8은 R-II input 복사, §7-1은 B-1 capture=0, [런처 223–231](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_det_convergence_2026-08-08.slurm:223)은 probe에 capture=1을 강제하므로 문면상 상호배타 결론이 그대로 나옴 | 지적 없음 |
| R5 살아남은 계수 | 재현됨 | `[R7][PHASE]`에서 iter 0 commit=189666행, iter 1 시작=189680행을 탐지. 1회 스트리밍 파싱 결과: 짝 2,420, ΔJ̄ 상이값 2,420, Δn_upper 상이값 73, `upper_global` 1,599, 축약비 `1599/73 = 21.904109589:1`. ΔJ̄ 범위 `−0.3100413356…+0.2977772321` | 지적 없음 |
| R5 표준 Spearman | 재현됨 | 동률 평균순위로 선 수준 `ρ=0.45880727880093547`, 73군 수준 `ρ=0.62859434777243`. 철회 고지의 `+0.4588/+0.6286`과 일치. 독립 코드는 [r5_stream.py](/gpfs/kjhan/lumina/slb/auditB2/codex_fresh.chIfbE/r5_stream.py) | 지적 없음 |
| R6 봉인 현재 무결 | 부분 재현 | R-II 전후 `135/135 OK, rc=0`, ZO 전후 `136/136 OK, rc=0`. R-III은 deck 55/55, topion 3/3, flight 3/3 및 binary·exports·sigma·netrate 지문이 전후 모두 일치 | 지적 없음 |
| R6 R-III 런 전체 무결 | 재현 불능(런 전체 manifest 부재) | R-III에는 `*RUN_SEAL.sha/meta`가 0개라 stderr·stdout 등 런 전체의 역사적 무결은 인증할 수 없음. 다만 위 입력 지문은 모두 정상. SLB E-8 자체는 checksum 가능 대상으로 R-II와 census만 규정하므로 판정식 반증은 아님 | «기재로 족함» |
| R6 E-8 모순 해소 여부 | 불일치 | 개정본 [34행](/gpfs/kjhan/lumina/slb/VERDICT_SLB_DPX.md:34)은 여전히 “사후 경로 미실행·미착지”, [92행](/gpfs/kjhan/lumina/slb/VERDICT_SLB_DPX.md:92)은 “사전·사후 rc=0 동시 기재”라고 함. 개정본은 모순을 해소하지 않았음. 현재 무결 자체는 위와 같이 정상 | «기재로 족함» |
| R7 판본·diff | 재현됨 / 문면 1건 불일치 | `.rev1`은 저장소 보존본과 `cmp rc=0`, SHA-256 `82b68e…ffdf`; 개정본도 저장소본과 `cmp rc=0`, `207b30…9404`. diff `+26/−16`, 9헝크, 순수 삭제 헝크 0. 다만 개정본 9행의 “현재=base → `.rev1`=이번 개정” 표기는 실제 판본 관계와 반대 | «기재로 족함» |
| R7 인용·환경·접촉 | 재현됨 | 판정문의 실제 파일:행 인용 `run_det_convergence…:223-231` 및 R2 소스 인용 전건 실존. `sacct -M syntax`로 B-1 `FAILED 70:0, 00:00:05`, B-2 `FAILED 70:0, 02:39:08` 재확인. 봉인 3곳·B-1/B-2 런루트·게이트 산출물은 read-only 접촉만 수행 | 지적 없음 |
| R7 저장소·`.pyc` | 재현됨 | 시작부터 존재한 tracked `M` 2개가 종료 시 동일하며 감리로 인한 tracked 변경 0. `.pyc`는 전후 599개, 경로 해시 동일 `44f048…a571`, 신규 0. 모든 Python은 `python3 -B`; 스크래치에는 파서 1파일만 기록 | 지적 없음 |

감리-B = 통과