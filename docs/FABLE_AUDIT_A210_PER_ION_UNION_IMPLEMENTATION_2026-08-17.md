# Fable 구현 감사: A2-10 per-ion minimal-prefix union

질의: `docs/QUERY_FABLE_A210_PER_ION_UNION_IMPLEMENTATION_AUDIT_2026-08-17.md`

Claude Code CLI Fable 모델을 read-only 도구와 plan permission으로 실행했다. 지정된
소스·스크립트·테스트·바이너리·봉인 산출물을 직접 읽은 최종 판정은 다음과 같다.

## 단일 판정

**APPROVE — 현재 구현으로 A100×2 전체 판정런 발주 승인.**

## 항목별 감사 결과

1. `a210_line_saturation_log_per_ion_union`은 global emission 내림차순/line-id 동률
   정렬의 ion별 부분수열을 순회하여 각 ion이 0.9에 처음 도달할 때 멈춘다. 따라서
   Fe/Co/Ni IV 각각의 deterministic minimal prefix union을 정확히 구현한다.
2. C selector 산술은 양의 long-double 덧셈과 `0.9L` 곱뿐이다. 최소성은 큰 수 차감
   없이 마지막 행 추가 직전 `ion_before_last < target <= ion_selected`로 증명한다.
   floor/cap/clamp/jitter/repair/abs-fix/음수 삭제는 없다.
3. mode 2의 common row rank와 candidate cumulative은 mode 1과 같은 전체 sorted
   sequence의 같은 부분합이다. 두 ROW format 문자열도 동일하다. 별도 union metadata가
   common row를 오염시키지 않으므로 교집합 전체 문자열 byte identity가 적절한 증거다.
4. summarizer는 행 삭제·scaled-emission 섭동·BLOCKED·repair·최소성 위반을 rc=4로
   차단한다. intersection/coverage comparator와 V4 monitor도 row/meta 삭제, 공통행
   1-byte 차이, 이온 누락, undercoverage, source/bundle/reference SHA drift를 fail-closed한다.
5. saturation env 소비자와 selector는 `lumina_plasma.c` 한 곳뿐이며 stderr-only 지역
   진단이다. GPU/physical producer/publication 소비자는 0이고 기존 `=1` 경로는 mode-2
   조기 분기 외에 비접촉이다.

## 독립 실측 재확인

- 새 binary SHA:
  `14b199f12d29246e1b7c7173d0ef9e8a9254ba4e8614cc0a838e85c200ad8825`
- 봉인 guard3 V2 JSON SHA:
  `4de6f38cf721aff5dc267b4e81b6482e874c99002b43a7534820226b820f6e20`
- guard3 stderr: ROW 929, SUMMARY 1, saturation BLOCKED 0, UNION 0,
  candidate_rows 211,887. Legacy 형식에는 selection_mode가 없다.

Mode-2 main summary의 `selected_reaches_target=1`은 per-ion 도달을 뜻하며
`selection_mode=PER_ION_UNION`으로 명시된다. Global fraction의 덧셈 결합순서 차이를
tolerance나 clamp로 숨기지 않고 세 ion metadata/V4 owner coverage만 합격 근거로 쓰는
설계도 승인됐다.

## 판정런 뒤 확인할 다섯 단계

1. `model.rc=1`, 자연 `RADEQ_NO_BRACKET`, flight script/NETRATE SHA 무드리프트 확인.
2. V2에서 `PER_ION_UNION`, BLOCKED 0, union row/meta count와 JSON SHA 봉인.
3. V3 PASS 뒤 V4에서 Fe/Co/Ni 각각 owner 대비 coverage ≥0.9와 prefix minimal 확인.
4. guard3 reference와 공통 row 전수 byte identity 및 세 ion 교집합을 확인.
5. R1/R2/LOWER/UPPER/REQUESTED_TE strict baseline과 owner closure까지 PASS한 뒤에만
   Stage 4 J/O 귀속 평가로 진행.

K-final 전용 `COMPLETED/child.rc=0`은 이번 진단에 요구하지 않으며, 물리 원인 claim은
여전히 0이다.
