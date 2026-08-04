Wave 3 Codex A 구현을 완료했습니다. 단, 현재 frozen data에서는 target/boundary 계약이 실패하므로 의도대로 solve·commit하지 않고 `EW_FAIL_SHADOW`로 중단됩니다. Acceptance PASS는 선언하지 않습니다.

전체 구현·검증 보고서: [CODEX_WAVE3_A_IMPLEMENTATION_2026-07-31.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE3_A_IMPLEMENTATION_2026-07-31.md)

주요 변경:

- [lumina_element_wide.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:41): strict gate, II–IV indexer, 7개 channel, target CSR, 단일 보존행, equilibration/LU/refinement, condition·pivot·scale·잔차 dump, fail-closed commit.
- [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14523): 기존 rate producer 재사용. EW 캡처 중 TOPSTAGE_IV·CE·legacy clamp/anchor/pin/closure 발화 차단.
- [bench_frozen_oracle.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:686): s0 Fe 및 s20 S shadow 준비. OFF 3셀 fixture는 불변.
- CUDA binary에서 EW ON은 CPU double reference solver로 라우팅. GPU 실행은 하지 않았습니다.

검증 결과:

| 검증 | 결과 |
|---|---|
| `make cuda` | exit 0 |
| OFF 미설정 vs 명시 0 | s0/s8/s43 `cmp=0`, stderr 0, EW artifact 0 |
| s8 shadow 2회 | exit 0, EW dump 16개 전부 byte-identical |
| S II–IV | N=303, raw rank=302, target 574/581 |
| Fe II–IV | N=303, raw rank=302, target 4076/4198 |
| clamp/fallback 발화 | 두 원소 모두 0 |
| TOPSTAGE_IV/pair owner/save-restore | EW candidate에서 0 |

OFF SHA-256:

- s0: `4789f13c89a3bb613e89cb23e836242285aae31bee6065b2631d61324eee1952`
- s8: `a4f1a146a313501a3eaf56232d2d7d3cd4f798425ebd8f426067292edb1538e2`
- s43: `c48d2619f160191d4a91e37334cf165d2fc312d2263635a281112523e70b72aa`

미해결 사항:

- S target CSR 7개, Fe 122개가 빠져 solve가 금지됐습니다.
- 제외 I/V fraction도 `1e-8`을 초과해 boundary gate가 실패합니다.
- s0 Fe shadow는 정상 실행됐습니다. s20 S는 현 parity50 run에 s20 frozen J/C1/C2 archive가 없어 입력 로드 단계에서 fail-closed했습니다.
- 따라서 condition·pivot·refinement는 `not_attempted`, `kappa_2/residual=inf`로 명시했습니다. B가 독립 산정할 raw/normalized 행렬은 보존돼 있습니다.

B는 target gap과 boundary inventory를 먼저 닫은 뒤 §4 산정을 수행해야 합니다. C는 CSR probability 단일 적용, III bb 중복 배제, EW commit 대상의 pair/save-restore/TOPSTAGE 우회를 source에서 검증하면 됩니다.

`git` 명령과 GPU 런은 사용하지 않았습니다.