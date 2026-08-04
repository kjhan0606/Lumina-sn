발주 D만 구현했습니다. 기존 dirty worktree의 사용자 변경은 보존했고, 게이트·하니스·GPU 바이너리는 실행하지 않았습니다.

변경 파일:

- [src/lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:818) — 지정 조성 판독부와 필요한 표준 헤더
- [composition_d_harness.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/composition_d_harness.c:1) — 실제 `lumina_atomic.c` 링크 CPU 하니스
- [generate_composition_d_fixtures.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/generate_composition_d_fixtures.py:21) — 18개 최소 오버레이 덱 생성·사전검증
- [run_composition_d_gate.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_composition_d_gate.py:12) — PASS/FAIL 및 비영 종료 러너

FATAL 구현 위치 — 정확히 16개:

- D1: 869
- D2: 1051, 1061
- D3: 1121
- D4: 1146, 1158
- D7a: 1192
- D7b: 1197
- D7c: 1202
- D8: 908, 934, 1054, 1111, 1140, 1185
- D9: 991, 1164
- D10: 901, 929, 1047, 1106
- D12: 1083
- D13: 1235
- D14: 872–1034
- D15: 1242
- D16: 1208
- D17: 953, 972, 1132, 1177

WARN 구현 위치:

- D5: 1254 — 집계 이벤트 한 줄 + 누락 Z 목록 한 줄, stdout
- D6: 1272 — 집계와 셸별 합계 목록, stdout

판독 루프는 EOF까지 진행하며 clamp·정규화·보정과 행수 등식 검사를 넣지 않았습니다. D11의 고정 `shell29` 주석도 제거했습니다.

픽스처는 D1, D2, D3, D4, D7a, D7b, D7c, D8, D9, D10, D12, D13, D14, D15, D16, D17, D5, D6의 18개입니다. 사전검증 결과 18/18 표현 가능했으며 D10 행은 8,227바이트, D8에는 실제 NUL 바이트가 들어 있음을 확인했습니다.

운전석 실행 명령:

```bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn

gcc -O2 -Wall -Wextra -std=c11 \
  -ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
  -o /tmp/composition_d_harness \
  scripts/composition_d_harness.c src/lumina_atomic.c -lm

python3 scripts/generate_composition_d_fixtures.py \
  --base data/tardis_reference_toy06_19p48d \
  --output /tmp/lumina_composition_d_fixtures

python3 scripts/run_composition_d_gate.py \
  --harness /tmp/composition_d_harness \
  --fixtures /tmp/lumina_composition_d_fixtures \
  --canonical data/tardis_reference_toy06_19p48d
```

빌드 결과:

- C 객체 컴파일: 성공
- NVCC C++14 객체 컴파일: 성공
- CPU 하니스 링크: 성공
- 전체 CPU 바이너리 `/tmp/lumina_order_d_cpu` 링크: 성공
- Python 스크립트 구문 컴파일: 성공
- 전체 CPU 빌드에서 기존 `src/lumina_cmfgen.c`의 `setenv` 선언 경고 1건만 발생

남은 위험은 발주 지시상 실제 18개 게이트와 정본 대조를 실행하지 않았다는 점입니다. 러너에는 정본의 D5 1건, 누락 Z `12,13,21,22,23,24,25`, D6 없음 조건이 명시돼 있습니다.