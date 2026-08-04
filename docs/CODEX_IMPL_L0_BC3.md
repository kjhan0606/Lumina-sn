완료했습니다.

- 계약 3건 전문 작성:
  - NE-NAMING: placeholder 명시·fail-closed 및 참값 경로 규정. 개명이나 즉시 덱 재생성을 처분으로 삼지 않음.
  - DECK-FOSSIL: generation 원자성·재현성 계약.
  - CONFIG-PREC: 단일 경계온도와 입력 우선순위 계약.
- `4.005038` 출처는 재현 가능한 producer를 찾지 못해 **`UNRESOLVED`**로 유지했습니다.
- CONFIG-PREC만 구현했습니다.
  - 우선순위: `argv > env > config.json > compiled default`
  - `plasma_state.csv`는 override가 아닌 consistency witness
  - gate: `LUMINA_CONFIG_PREC`
  - 기본 OFF: 현재 덱은 명시적 WARN 후 계속
  - ON: 10020 K 대 14172.549 K 불일치로 FATAL, 프로그램 rc=1
  - 잘못된 gate/env 값도 fail-closed
- `/tmp` 사본을 쓰는 음성 대조 4건을 추가했습니다.
- `LUMINA_TRAD_COLOR_FIX`, 덱, GPFS는 수정하지 않았습니다.
- 기존 dirty 변경을 보존했고 commit/push하지 않았습니다.
- 정적 `git diff --check`는 통과했습니다.

전체 계약·조사 근거·A-2 의존표·복사 가능한 운전석 명령은 [CODEX_L0_NFP_CONFIG_PREC.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_L0_NFP_CONFIG_PREC.md)에 있습니다.

주요 구현:

- [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:515)
- [lumina_main.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_main.c:210)
- [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7343)
- [run_config_prec_negative_controls.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_config_prec_negative_controls.py:1)

로그인 노드 연산 금지에 따라 빌드·음성 대조·D 19/19·K 7/7·Z-INERT는 실행하지 않았으며 현재 `PENDING_DRIVER_EXECUTION`입니다. 문서 §8의 grammar-debug 명령이 인수 절차입니다.