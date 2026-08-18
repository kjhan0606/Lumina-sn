# Fable 구현 감사 요청: A2-10 per-ion minimal-prefix union

읽기 전용 감사만 수행하시오. 파일·코드·프로세스·Slurm 상태를 변경하지 마시오.

선행 판정과 정정 계약:

- `docs/FABLE_AUDIT_A210_PER_ION_UNION_BINARY_SHA_2026-08-17.md`
- `docs/A210_PER_ION_UNION_PREREGISTRATION_CORRECTION_2026-08-17.md`

감사 대상:

- `src/lumina_plasma.c`
  - `A210LineSaturationDiagnostic.mode`
  - `a210_line_saturation_target_index`
  - `a210_line_saturation_log_per_ion_union`
  - `a210_line_saturation_log_complete`의 mode-2 분기
- `scripts/summarize_a210_line_saturation.py`
- `scripts/compare_a210_line_saturation_intersection.py`
- `scripts/compare_a210_cmfgen_line_saturation.py`
- `scripts/stage_a210_line_saturation_diagnostic.sh`
- `scripts/stage_a210_line_saturation_per_ion_coverage_v4.sh`
- `scripts/monitor_a210_line_saturation_per_ion_coverage_v4.sh`
- 관련 `tests/a2_10_line_saturation_*` 및
  `tests/a2_10_cmfgen_line_saturation_comparison_selftest.py`

실측 검증:

- 새 fat binary SHA:
  `14b199f12d29246e1b7c7173d0ef9e8a9254ba4e8614cc0a838e85c200ad8825`
  (`/tmp/lumina_cuda_a210_per_ion_union_20260817`)
- nvcc 전체 link PASS. 출력 경고는 기존 unrelated unused/static declaration 경고뿐이며
  새 함수 경고는 0.
- `make selftest-a2-10-line-saturation` PASS.
- union summarizer의 row 삭제와 scaled-emission 섭동 음성 대조가 각각 rc=4 FAIL.
- intersection comparator의 공통 row 섭동과 row/meta 삭제 음성 대조가 각각 rc=4 FAIL.
- 수정된 summarizer로 봉인 guard3 `=1` stderr를 다시 처리한 JSON SHA는
  기존 V2 JSON과 동일한
  `4de6f38cf721aff5dc267b4e81b6482e874c99002b43a7534820226b820f6e20`,
  `cmp` rc=0.

구현 의도:

1. 기존 env 값 `1`은 combined 90% prefix 코드와 row 출력 문자열을 그대로 사용한다.
2. 새 값 `2`만 Fe/Co/Ni IV 각각의 total을 계산하고, global emission 내림차순 배열을
   순회하면서 각 ion total의 0.9에 처음 도달할 때까지 선택한다. global 정렬의 ion별
   부분수열이 곧 ion별 emission 내림차순/line-id 동률 순서이므로 최소 prefix다.
3. 최소성 검사는 `selected-last` 같은 큰 수 차감을 하지 않고, 마지막 행을 더하기 직전
   누적값 `ion_before_last < target <= ion_selected`를 직접 보존한다.
4. 공통 `LINE-SATURATION-ROW`는 전체 candidate global rank와 전체 candidate cumulative을
   사용한다. 따라서 기존 global-prefix와 새 union의 교집합 행은 동일 state에서 row
   문자열 전체가 바이트 동일할 수 있다. union 선택 provenance는 별도
   `LINE-SATURATION-UNION-META`와 ion summary에 기록한다.
5. per-ion totals와 selected sums는 양의 long-double 덧셈만 사용한다. 물리값을
   floor/cap/clamp/jitter/repair/abs-fix/삭제/scale하지 않으며 selector 출력은 어떤
   physical producer/publication도 읽지 않는다.
6. global total은 기존 scan-order `diag->total_scaled_emission`을 그대로 보고한다.
   per-ion totals는 global-sort 후 ion별 부분수열 순서로 합산한다. 서로 다른 덧셈
   결합 순서 때문에 global selected_fraction이 직렬화상 0.9보다 극미량 작을 가능성을
   물리 tolerance나 clamp로 숨기지 않고, mode 2의 합격 근거는 세 ion 각각의 exact
   minimal-prefix metadata/V4 owner coverage로 한정했다.

다음 항목을 명시적으로 판정하시오.

1. 위 selector가 사전등록한 per-ion minimal 90% prefix union을 정확히 구현하는가.
2. 큰 수 차감/roundoff 증폭, 잘못된 최소성 증명, 수치 floor/cap/repair가 숨어 있는가.
3. candidate global rank/cumulative을 공통 row에 유지하고 union metadata를 분리한 설계가
   교집합 byte-identity 증거로 적절한가.
4. summarizer/comparator/monitor가 행 삭제·섭동·undercoverage·SHA drift를 fail-closed하는가.
5. 변경이 diagnostic-only이고 물리 producer/publication/co-evolution barrier에 영향 0인가.
6. 현재 구현으로 A100x2 전체 판정런을 발주해도 되는지 `APPROVE` 또는 `BLOCK`으로
   결론을 내리고, BLOCK이면 필요한 국소 수정만 정확히 지정하시오.

승인 시 판정런 뒤 확인할 다음 5단계도 적으시오. 물리 원인 claim은 아직 0이다.
