# Fable 판정 요청: A2-10 V4 per-ion union과 literal binary SHA 모순

읽기 전용으로만 판정하시오. 파일·코드·프로세스·Slurm 상태를 변경하지 마시오.

## 역할과 절대 제약

- Fable: 총괄·계획, 분석·평가, 물리/구현 감사.
- Codex: Fable 판정 이후 코딩·실행·모니터링·문서·커밋.
- 물리값의 floor/cap/clamp/jitter/repair/abs-fix/음수 삭제/임의 scaling은 절대 금지.
- 음수·nonfinite·물리 모순은 원인을 규명하지 못하면 fail-closed.
- rejected pre-core tau refresh는 복귀 금지, coevolution generation barrier 유지.

## 봉인된 K36 guard3 사실

- run root:
  `/gpfs/kjhan/lumina/a210_line_saturation_a100x2_nonoverlap_sobolev_k36/manual_retry_guard3_20260817T011506Z_f9c2d1b826d5`
- binary SHA:
  `f9c2d1b826d5205fa68f938c2affc2c6d9aa86772257fb95958ab9e65a95526c`
- 자연 종료: `model.rc=1`, `RADEQ_NO_BRACKET`, generation/publication 보존,
  floor/cap/clamp/jitter/repair=0, tripwire 충돌=0.
- R1/R2 exact/R6와 LOWER/UPPER/REQUESTED_TE 공통 baseline은 기존 K36 owner
  정본과 strict bit-exact이다.
- V2 PASS: 211,887 candidates 중 combined-emission 90% global prefix 929개,
  selected fraction `0.90009675901227582`, CMFGEN exact transition match 929,
  parity/cause claim은 state mismatch 때문에 0.
- V3 PASS: 929개 모두 tau>1이며 negative external-continuum component 및
  `Jbar/S < 1-beta`가 arithmetic proof bound로 인증됐다. 이 bound는 물리 tolerance가 아니다.
- V4는 `UNDERCOVERED`:
  - Fe IV: `55.810176826045606 / 75.583805875304174493 = 0.738388020816516...`
  - Co IV: `559.758405420775605 / 598.177940701955151 = 0.935772397029394...`
  - Ni IV: `34.054848474162863 / 47.9644725392837613 = 0.710001521360864...`
  - V4 report SHA:
    `1854b00176c6b416381411cd9f3a83eed565397125bd9cc8be1535cc8fc3cb53`
  - coverage report SHA:
    `e0625f4709086130b27b46dad4e6734739a5d3099aa3aa5853323f6cf6f6d997`

## 발견된 구현 사실과 모순

다음 파일과 산출물을 직접 읽어 확인하시오.

- 계획: `docs/FABLE_PLAN_ANALYSIS_TRANSFER_A210_2026-08-17.md` Stage 3
- 정정: `docs/FABLE_PLAN_ANALYSIS_TRANSFER_CORRECTION_A210_2026-08-17.md`
- 구현: `src/lumina_plasma.c`의 `A210LineSaturationDiagnostic` 및
  `a210_line_saturation_log_complete`
- V4 JSON:
  `/gpfs/kjhan/lumina/a210_line_saturation_a100x2_nonoverlap_sobolev_k36/manual_retry_guard3_20260817T011506Z_f9c2d1b826d5/a210_line_saturation_per_ion_coverage_v4.json`
- 봉인 stderr:
  `/gpfs/kjhan/lumina/a210_line_saturation_a100x2_nonoverlap_sobolev_k36/manual_retry_guard3_20260817T011506Z_f9c2d1b826d5/stderr.log`

현재 구현은 모든 Fe/Co/Ni IV candidate를 메모리 안에 모으지만, 종료 시 세 ion을
합친 scaled emission 내림차순의 단일 90% prefix만 stderr에 기록한다. 실행 중 저장된
211,887 candidate 가운데 929개만 로그에 남았으므로, 누락된 Fe/Ni row는 봉인 로그에서
offline 복원할 수 없다. per-ion 선택 mode나 환경변수도 없다.

그런데 기존 Stage 3 계획은 `UNDERCOVERED`이면 “같은 sealed 진단의 ion별 minimal 90%
prefix union만 재실행, 동일 binary SHA·동일 state 필수”라고 규정한다. 현재 바이너리는
그 union을 낼 경로가 없으므로 **literal 동일 binary SHA**와 **per-ion union**을 동시에
만족할 수 없다.

## Fable이 내려야 할 단일 판정

아래 중 하나를 명시적으로 선택하고 근거와 정확한 다음 5단계를 제시하시오.

1. **새 diagnostic-only 바이너리 허용**: per-ion minimal 90% prefix union 선택/기록만
   추가하여 새 SHA로 같은 입력/state를 재실행한다. 이 경우 “동일 binary SHA”를
   “동일 물리 baseline/source state; 변경은 read-only 진단 선택/기록에만 한정”으로
   정정할 수 있는지 판정하시오.
2. **현 증거로 union 생략 허용**: 생략이 물리/통계적으로 정당한 경우에만 정확한 증명을
   제시하시오.
3. **다른 read-only 복원법**: 현재 sealed 산출물만으로 211,887 candidate의 ion별 union을
   정확히 복원할 수 있다면, 사용 가능한 실제 파일/record와 절차를 명시하시오.

새 바이너리를 허용한다면 다음 증거가 충분한지도 판정하시오.

- diff가 selection/logging과 strict diagnostic option에만 국한되고 물리 producer/publication에는 영향 0.
- 기존 `LUMINA_A210_LINE_SATURATION_DIAG=1` 동작은 회귀 테스트에서 바이트 동일.
- 새 mode는 ion별로 scaled emission 내림차순, 동률은 line id 오름차순, 각 ion 총량의
  90%에 처음 도달하는 최소 prefix를 취하고 세 prefix의 union만 기록.
- 동일 deck/input/Sigma/state 계보, pre-core=0, generation barrier/publication 보존.
- 새 실행 R1/R2/LOWER/UPPER/REQUESTED_TE 공통 baseline을 기존 guard3와 strict 비교.
- 새 target row만 증가하며 physical_values_modified=0, 모든 repair counter=0.
- V4 PASS 뒤에만 J/O 귀속 평가로 진행하고, 이 재실행 자체에는 물리 원인 claim=0.

마지막으로, 새 diagnostic-only 바이너리가 필요하다면 약 2.5시간의 A100x2 전체 재실행이
반드시 필요한지, 아니면 동일 state를 물리적으로 정당하게 재현하는 더 좁은 실행 경로가
있는지도 판정하시오. 추측하지 말고 코드와 봉인 산출물에 근거하시오.
