수리를 완료했습니다. topology·numerics gate는 이제 실제 solve와 유효 `p_elem`까지 통과합니다. 다만 I/V boundary active-set 문턱은 그대로 초과하므로 Stage 2A 전체 PASS를 허위로 선언하지 않고 `EW_VALID_P_ELEM_SCOPE_FAIL`로 분리했습니다.

주요 변경:

- 유효 σ identity가 있는 continuum만 target topology에 포함하고, 합성 Kramers fallback 없이 S `574/574`, Fe `4076/4076`을 완결했습니다.
- atomic identity에 level·route·σ grid·line rate·ionization threshold·Fe III collision table을 포함하고 조립 전후 checksum을 실제 비교합니다.
- 7개 plane의 활성/비활성 사유와 실제 event count를 기록하며, diagonal 재봉합 없이 원자적 inflow/outflow 배치 자체를 검사합니다.
- bf rate를 upper-target별 threshold·통계중량·`n*`·Milne/thermal inverse로 계산하고, NT도 모든 lower level/CSR target을 순회하도록 변경했습니다.
- fallback은 EW 추가 슬롯을 제외한 원래 순서의 16개 pair-wise 호출로 복원했습니다.
- 실제 fallback/cap/nonfinite 발화 계측을 연결했습니다.
- Golub–Reinsch SVD, 독립 LU 1-norm `rcond`, 안정적 equilibration, `n_i/t_ref` 기반 scaled residual을 적용했습니다.
- C48은 숨은 runtime guard가 아니라 checksum에 포함되는 입력 projection으로 명시·계수화했습니다. C65는 EW+STAGE4 충돌을 거부하고 downstream cap도 EW 대상에서 우회합니다.

구현 위치: [element-wide assembler](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:245), [baseline fallback 및 통합](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:8117), [capture API](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:935), [frozen oracle fixture](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:621).

검증 결과:

- s8 S: rank `303/303`, κ₂ `3.8971e7`, residual `1.1171e-15`, conservation `0`, `p_elem_valid=1`
- s8 Fe: rank `303/303`, κ₂ `1.7859e7`, residual `1.1378e-15`, conservation `6.72e-16`, `p_elem_valid=1`
- s8 2회 실행: oracle 포함 산출물 CSV 17개 전부 byte-identical
- s0 Fe: `4076/4076`, solve 수행, κ₂ `2.5573e4`, residual `1.3683e-15`, `p_elem_valid=1`
- OFF SHA 불변:
  - s8: `2ee175c6be4a0ab6ae07034722180c6312f0a6e55e1009fcdda23b72310693eb`
  - s0: `7a79f4f345d5c8500b48530a7f859bbe05a57771f43e52f983639ef9a0839381`
- `make cuda`: 성공. 기존 미사용 변수 경고만 발생
- GPU 실행 및 git 명령: 수행하지 않음
- s20: 요청대로 건드리지 않음

남은 scope gate는 s8 boundary fraction S `1.958e-5`, Fe `7.923e-5`와 boundary process producer 부재입니다. 따라서 유효 `p_elem`은 생성됐지만 해당 해를 acceptance/commit PASS로 승격하지는 않습니다.