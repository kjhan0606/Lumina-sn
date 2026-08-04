수리 완료했습니다. C 리뷰 FAIL 2건을 정면 해소했고 `NEUTRAL·ETA_SPINGATE` 구현은 변경하지 않았습니다.

- Stim recombination: multi-target CSR로 `Σ p_target·corrfactor_target` 구현, clumped `n_e`, 독립 target-map 로드, ARTIS H/KB 정합. v1 map은 단일-route `p=1`로 호환합니다. [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6730), [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:1096)
- Multi-edge OFF 계약: 명시적 `LUMINA_FIX_BF_MULTI_EDGE=0`이 alias보다 항상 우선합니다. [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6442)
- 두 GPU 방출점: 선택 level을 모두 전달하며 동일한 `σ_bf·ν²` Milne sampler를 공유합니다. [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4523)
- target-map 생성기는 v2 CSR round-trip을 지원합니다. [build_ma_radrecomb_target.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/build_ma_radrecomb_target.py:26)
- 보수 기록도 갱신했습니다. [Wave 1.1 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE1_BF_REPAIR_REPORT_2026-07-31.md:8)

최종 CPU oracle 결과:

| 셀 | OFF SHA-256 |
|---|---|
| s0 | `8cbccb2cac2fb7b860eac45edd8479f36f5f5b010e0dd3708d463eff389332b6` |
| s8 | `dad29ce6b39a00609f6b63aa06cb85c8fb323212921081d434d8ca5510115767` |
| s43 | `432952ec471323a7d164a31792c21d117cbc3221af3ac63d753696f57f182112` |

unset ↔ 명시적 0 ↔ `MULTI_EDGE=0 + alias=1` 모두 byte-identical이며 충돌 arm의 FB-MULTI 배너도 없었습니다. Stim v1/v2 출력도 byte-identical, probability KA 선형 잔차는 정확히 `0`이었습니다.

`bench_frozen_oracle` 강제 빌드와 Python 구문검사를 통과했습니다. git 및 GPU 빌드·실행은 하지 않았습니다.