벤치 구현은 완료했지만, 최종 물리 판독은 **어느 가지도 선택할 수 없는 `UNRESOLVED-EXTRAP-POSITIVITY`**입니다.

- 입력 인증 PASS: `iter=10`, `gen=10`, `post_damp=1`
- SHA-256: `94d75988034454f55fb6b130f04521f01c56f875cb22ef3a711850d7382ffa2f`
- 첫 sweep이 0.062초 만에 외곽 face의 음의 χ(`−1.4236e−20`, bin 331)로 fail-closed
- 외곽 외삽에서 χ 음수 40빈(판별 대역 내 36빈), η 음수 37빈(전부 판별 대역)
- clamp·floor·boundary-shell 복제 없이 중단

따라서 J_det와 Γ D-lane은 대체값 없이 미산출했습니다. 확인 가능한 baseline은 다음과 같습니다.

- J_producer/CMFGEN: 대역별 `6.91–33.76×`, 전체 600–3000 Å `11.98×`
- Fe III idx 201: Γ_MC/Γ_CMFGEN = `15.55×` (`+1.191601 dex`)
- S II SL4: `69.71×` (`+1.843330 dex`)

산출물:

- [전체 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_BENCH.md)
- [수치 JSON](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/s31_results/stage31_bench_round7.json)
- [결정론 C 드라이버](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/stage31_cmf_field_driver.c)
- [오프라인 벤치 runner](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/stage31_cmf_field_bench.py)

strict C11/Python 검증과 missing-sidecar 음성시험도 통과했습니다. 기존 `src` 수정, acceptance 변경, 모델/GPU 실행, 커밋은 없습니다.

재개하려면 boundary half-cell을 현재 KA본의 one-sided extrapolation으로 유지할지, 원설계의 constant extension으로 승인할지 결정이 필요합니다.