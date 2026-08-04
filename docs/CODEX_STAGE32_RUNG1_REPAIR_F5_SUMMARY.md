완료했습니다.

- v3 패치: [stage32_rung1_readonly_lambda_v3.patch](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/stage32_rung1_readonly_lambda_v3.patch)
- 전체 보고서: [CODEX_STAGE32_RUNG1_REPAIR_F5.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE32_RUNG1_REPAIR_F5.md)
- v2 SHA-256: `60bc65c172f95da5c10a07f37743b1f51b007ab9645b503aa0a7ff1d028655be`
- v3 SHA-256: `b41e991c5ff947e488c85b678b96bfc1f5ba4f88d78146e3978ab262e1728c30`

검증 결과:

- `git apply --check --whitespace=error-all`: PASS
- 격리 `make selftest_stage32_rung1`: PASS
- 격리 CPU `make -B lumina`: PASS
- 두 필수 음성 대조: 각각 residual `-6.4529279658056094e-07`, `+6.4529279658056094e-07`로 FAIL 확인
- 가중 판독: 분율 `0.660377358490566 → 0.8383233532934131`, 중앙값 `0.99 → 0.99`
- 양쪽에서 동시에 ε를 제거하는 결함은 검출하지 못함을 명시
- 실제 트리에 패치 적용·빌드·commit 없음
- GPU 및 모델 런 없음