# All-shell atomic trial bundle 상태 — 2026-08-08

상태: **private producer, vector root, replay, 단일 commit까지 구조 폐합 완료**.

하나의 all-shell trial temperature generation은 다음을 모두 private로 생산한다.

1. partition/within-superlevel과 공유 A2-07 CE population core
2. ion writeback, signed Sobolev tau/source, 명시적 EW authority
3. CPU BF(+FF) grid와 detached A2-08 opacity publication
4. 직접 Sobolev line 식을 쓰는 detached A2-09 emissivity/CDF publication
5. neutral-ground 기준 excitation+ionization 내부에너지
6. CMFGEN `EVAL_ADIABATIC_V3` 대응 signed 네 component
7. 같은 후보 publication으로 만든 A210 heating/cooling ledger

전 shell residual은 같은 Te vector에서 평가한다. 수렴한 vector는 bundle을 한 번 더 재생성하며,
root publication의 ledger와 byte-identical일 때만 commit 자격을 얻는다.

최종 preflight가 Te manifest, 모든 세대·provenance, 배열 shape, CDF, BF, tau,
A208/A209 closure와 ownership 분리를 검사한다. 성공 뒤에는 Te, ne, ion/level population,
partition, within-SL, raw tau/source, BF, A208, A209, A210과 후보-local counter를 한 번에
이전한다. preflight 실패에서는 공개 state가 byte-preserved다.

MC와 순수 CMFGEN 외부 루프가 이 commit 직후 `compute_plasma_state` 또는
`nlte_solve_all`을 재호출하던 경로도 제거했다. 선택형 CMFGEN `J_inc`/overlap 보정은 공개
material을 즉시 다시 풀지 않고 다음 outer iteration 후보의 lagged 입력이 된다.

제한:

- 후보 BF는 login-node 안전을 위해 CPU arithmetic으로 고정한다.
- vector root는 coherent all-shell evaluation이지만 componentwise bisection이다.
- physical flight와 외부 oracle 일치는 아직 주장하지 않는다.
- 남은 one-time 환경 parse cache와 오류 `stderr`는 material/publication mutation이 아니지만
  최종 side-effect census에서 계속 추적한다.

검증:

- candidate view/tau/adiabatic/internal-energy/single-commit selftest PASS
- A2-10 N1–N8 PASS
- CPU/OpenMP/CUDA multi-arch compile+link PASS
- GPU binary와 물리 모델은 실행하지 않음

Fable 호출은 동결했다. 로컬 계약으로 해결할 수 없는 새로운 핵심 물리 모순에만 사용자에게
먼저 사유를 보고한 뒤 사용한다.
