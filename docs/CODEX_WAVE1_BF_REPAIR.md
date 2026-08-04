Wave 1.1 bf 보수를 완료했습니다.

- D-3: ARTIS `rpkt.cc:733-765`의 multi-target CSR probability 합산, clumped
  n_e, 독립 target-map 로드와 ARTIS 상수까지 항별 정합했습니다. 순흡수
  `chi_bf`와 spontaneous `eta_bf`를 분리했습니다.
- Neutral bf: O I 등 stage 0 광이온화를 CPU 및 bf-GEMM 양쪽에서 복원했습니다.
- B18-ⓐ: `eta_bf`에 `REC_SPINGATE` 공유 술어를 일관 적용했습니다.
- C59: 명시적 `LUMINA_FIX_BF_MULTI_EDGE=0`이 legacy alias보다 항상 우선하며,
  alias는 새 변수가 unset일 때만 동작합니다. 두 GPU 방출점은 동일한
  level-resolved Milne sampler를 사용합니다.
- 모든 신규 게이트는 기본 OFF입니다.
- upstream 수리와 직접 연결된 항목이 없어 clamp는 제거하지 않았습니다.

검증 결과:

- CPU frozen oracle 강제 빌드 성공
- GPU 실행 및 Git 명령 미사용
- unset/명시적 0/alias 충돌 frozen-cell oracle s0/s8/s43 CSV byte-identical
- 정규화 stdout도 byte-identical
- D-3 LTE KA: 해석식과 절대차 `0`
- 비-bf B형 잔여 14건 목록화 완료

구현은 [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6420), [lumina_bf_gemm.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_bf_gemm.cu:60), [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:2118)에 있습니다.

전체 공식 출처, 게이트표, OFF 불변 논거, Codex B oracle/KA 검증 지침과 잔여 목록은 [Wave 1 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE1_BF_REPAIR_REPORT_2026-07-31.md:8)에 정리했습니다.
