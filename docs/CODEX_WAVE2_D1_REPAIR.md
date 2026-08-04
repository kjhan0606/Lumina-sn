Wave 2 수리를 완료했습니다. 상세 항별표와 검증 자료는 [종합 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE2_A_REPAIR_REPORT_2026-07-31.md)에 정리했습니다.

핵심 결과:

- D-1: argmax `activation_level` 대신 전체 `(element, ion, level, target)` opacity CDF를 추첨하고, 별도 난수로 `nu_edge/nu` MA-vs-k 분기를 구현했습니다. CPU와 [GPU CDF sampler](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3487) 모두 반영했습니다.
- ARTIS `rpkt.cc:405–445`의 continuum 선택 → target 보존 → `nu_edge/nu` 분기 순서를 대응시켰습니다.
- GPU 두 free-bound 방출점은 BF 초기화 후 실제 `sigma_bf` handle을 재결합해 동일 Milne sampler를 사용합니다. [rebind 위치](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7227)
- 비-bf B형 14건 중 C28·C71을 각각 기본-OFF 게이트로 수리했습니다. C21은 signed maser transfer가 필요해 잔여로 유지했고, 나머지 11건도 [provenance 문서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_CLAMP_PROVENANCE.md)에 물리 근거와 함께 분류했습니다.
- D6 배너는 KPACKET 존재 시에만 `ENABLED`, 없으면 `PARTIAL`, event gate OFF면 `residual`을 출력합니다.

게이트는 모두 기본 OFF입니다.

- `LUMINA_FIX_BF_CONTINUUM_EVENT`
- `LUMINA_FIX_BF_MULTI_EDGE`
- `LUMINA_FIX_MA_J_UNCLAMP`
- `LUMINA_FIX_MA_NO_LINE_THERM`

Oracle 결과에서 unset OFF, 명시적 `=0`, D-1 ON이 동일했습니다. `available` 전체가 기존 제출과 byte-identical이므로 그 부분집합인 strict-compared 행도 동일합니다.

| 셀 | eligible SHA-256 |
|---|---|
| s0 | `beaac19b21bd5b9c0d8c7c81903a1c8c13c8f139ba05cf2e01c414f193678cfa` |
| s8 | `54f9fafad8da44602a419562a2ef37c9f0c726fdad6780c72e99df436e87d05f` |
| s43 | `b971a0381d4d6c8246979c3bb8d013290d65deac6985898795bee94894380804` |

D-1 ON의 packet fate는 frozen oracle 범위 밖이므로 효과를 주장하지 않았습니다. C71과 multi-edge 역시 CUDA 사건 전용이라 oracle 부적격으로 명시했습니다.

검증:

- `make cuda`: 성공, 기존 `g_fgemm_nulo` 미사용 경고만 존재
- GPU 실행: 하지 않음
- git 명령: 사용하지 않음
- transport kernel REG: sm_80/86 `98→102`, sm_90 `92→92`, spill 0
- D-1 ON 추가 메모리: 생산 크기 기준 약 224.26 MiB
- 사건당 최악 26,592-route 선형 scan이며, ON에서는 opacity BF-GEMM도 exact CPU loop로 우회합니다.

결함 계보는 대표값 축약, 에너지 장부 누락, 초기화 순서, 진단 prior의 생산 침투, backend 경로 분기의 다섯 범주로 정리했습니다. Codex B의 known-answer/향후 GPU 계측 절차와 Codex C의 독립 index·double-count·D6 검토 지침도 보고서 끝에 포함했습니다.