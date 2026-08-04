수리 완료했습니다. 귀속·인자분해 로직은 변경하지 않았습니다.

- [scheme_fracture_s0.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/scheme_fracture_s0.py:259)
  - 비트동일 비교를 `64 * eps_float64 * n_terms * max(|a|, |b|)` 한계로 교체했습니다.
  - `k=64`는 참조 closure 게이트에서 계승했으며, 서로 다른 두 누적 순서를 포괄합니다.
  - `n_terms`는 합산 항 수, 분수는 분자 항+분모 항+나눗셈 1회입니다.
  - 실제 로그의 절대차는 `3.3881317890172014e-21`, 상대차는 `1.331634349432582e-16`입니다.
  - 상대 허용 한계는 `1.4210854715202004e-14 * n_terms`입니다.
  - 산출물에 `actual_relative_difference`, `absolute_difference`, `relative_roundoff_limit`, `absolute_roundoff_limit`, `n_terms`, `roundoff_k`, `float64_epsilon`을 수치로 기록합니다.

- [pile_ion_attribution.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/pile_ion_attribution.py:130)
  - 사이드카 존재 시 반드시 검증하며, 불일치·손상·읽기 실패는 FAIL입니다.
  - 사이드카가 없으면 직접 SHA-256을 계산합니다.
  - `provenance.events|lines|matrix.sha256_source`에 `sidecar_verified` 또는 `computed_now`를 기록합니다.
  - `sha256_evidence`와 `sha256_sidecar_path`도 남겨 두 증거의 의미를 명확히 구분합니다.

자기검사 결과:

- 두 `--self-test`: `status=PASS`, 종료 코드 `0`
- 1 ULP 양성 대조: 상대차 `1.3082505504260825e-16`, 허용 `5.684341886080802e-14`
- mapsplit 결함 주입:

  `FAIL (expected): ... rel_diff=0.8221567880099646 rel_limit=5.6843418860808015e-14 n_terms=4`

- 귀속 결함 주입:

  `FAIL (expected): INJECTED-DEFECT emitted-ion attribution failed fixture oracle`

- SHA 불일치 주입:

  `FAIL (expected): SHA-256 mismatch ... expected 0000... got 916f6a...`

실데이터 전체 통과와 8GB 이벤트 로그는 실행하지 않았습니다. 남은 미해결은 계산 노드에서의 실데이터 재실행뿐입니다. 다른 소스 파일 수정이나 커밋은 하지 않았으며, 두 스크립트는 현재도 Git `untracked` 상태입니다.