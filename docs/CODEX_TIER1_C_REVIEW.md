## 판정

정상 종료 경로 기준 세 부위 모두 PASS입니다. 계기 추가로 물리 수식·분기·인덱스가 바뀐 흔적은 없습니다.

### 1. D6 배너 — PASS

- 배너: [src/lumina_cuda.cu:6172](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6172)
- 실제 BF 흡수 경로는 입사 패킷 전체를 activation level에 전달하며, `nu_edge/nu` 에너지 분할이 없습니다: [src/lumina_cuda.cu:5510](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5510), [src/lumina_cuda.cu:5546](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5546)
- BF 흡수 경로에 stimulated-recombination 보정도 없습니다. 다른 곳의 `stim` 계산은 line/free-free용이며 D6 구현이 아닙니다.
- `nu_edge`를 쓰는 RADRECOMB 방출 경로 [src/lumina_cuda.cu:5685](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5685)도 “흡수 시 `nu_edge/nu` 분할”과는 별개입니다.
- D6 주변 변경은 `printf` 배너뿐이며 실행 수식·조건·인덱스 변경은 섞이지 않았습니다.

따라서 “D6는 아직 미구현”이라는 현 문구는 정직합니다.

### 2. `[FB-EDGE]` — PASS

정상 종료별 호출은 정확합니다.

- pure-CMFGEN: `!cmfgen_then_mc`일 때 출력 후 반환: [src/lumina_cuda.cu:8825](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:8825), [src/lumina_cuda.cu:8834](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:8834)
- 일반 MC 및 THEN-MC: 공통 MC 에필로그에서 출력: [src/lumina_cuda.cu:9817](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:9817)
- 함수 내부 `static printed`가 재호출도 차단: [src/lumina_cuda.cu:2239](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:2239)

카운터 회수 사슬도 물리 상태를 건드리지 않습니다.

- 호스트 카운터/getter: [src/lumina_plasma.c:3050](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:3050)
- 기존 dominant-edge 계산 뒤 진단값만 증가: [src/lumina_plasma.c:4620](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4620)
- 장치 getter: [src/lumina_cuda.cu:2229](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:2229)
- 네 퇴화 지점 모두 카운트: [src/lumina_cuda.cu:4879](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4879), [src/lumina_cuda.cu:4901](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4901), [src/lumina_cuda.cu:5649](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5649), [src/lumina_cuda.cu:5668](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5668)

카운트 전후의 fallback/no-op 제어흐름과 RNG 소비는 그대로입니다. 정상 시 이중 인쇄·미인쇄 경로는 없습니다.

### 3. `[FORMAL-CONS]` 확장 — PASS

- 기존 첫 `[FORMAL-CONS] integral ...` 줄은 그대로 유지됩니다: [src/lumina_plasma.c:16905](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16905)
- 새 줄만 뒤에 추가됩니다: [src/lumina_plasma.c:16913](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16913)
- 산식은 `L_total_in = L_inj + L_dep`, `ratio_total = Lint/L_total_in`으로 정확합니다.
- 로더는 CSV 값을 `heating_rate`에 그대로 저장합니다: [src/lumina_cuda.cu:6882](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6882)
- 단위는 `erg/s/cm³`; 기존 셸 체적과 곱해 적분합니다: [src/lumina_cuda.cu:2252](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:2252), [src/lumina_cuda.cu:6605](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6605)
- formal flux 계산이 끝난 뒤 출력에만 사용되므로 spectrum/opacity/plasma에는 역류하지 않습니다.

직접 적분 결과:

```text
Σ heating_rate_i × (4π/3)(r_outer_i³-r_inner_i³)
= 7.787639096650e42 erg/s
```

입력은 [deposition_cmfgen.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/deposition_cmfgen.csv:1), 체적은 [geometry.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/geometry.csv:1)을 사용했습니다.

따라서:

- 진짜 deposition 합: **`7.787639e42 erg/s`**
- `1.088e43 erg/s`: deposition이 아니라 약 `3.09e42`의 inner-boundary luminosity를 더한 **`L_total_in`**

### 남은 위험

- selftest, fatal error, `CUDA_CHECK` abort처럼 정상 에필로그에 도달하지 않는 조기 종료에서는 `[FB-EDGE]`가 출력되지 않습니다. “모든 프로세스 종료”까지 요구하면 별도 `atexit` 처리가 필요합니다.
- 일반 MC의 `iter>0` 경로는 외부 deposition 파일이 있어도 `compute_gamma_deposition()`을 다시 호출합니다: [src/lumina_cuda.cu:9395](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:9395). 따라서 다중-iteration MC의 최종 meter는 CSV 원본이 아니라 그 시점의 현재 `heating_rate` 배열을 보고할 수 있습니다. 이는 계기 확장 자체의 단위 오류는 아니지만, 외부 deposition을 끝까지 고정하려는 실행에는 기존 위험입니다.
- 작업 트리에 광범위한 다른 미커밋 변경이 있어 배치 전체의 변경 귀속은 신뢰 가능한 직전 스냅샷 없이는 증명할 수 없습니다. 다만 위 세 계기의 직접 인접 코드에서는 물리 수식·조건·인덱스 혼입을 발견하지 못했습니다.