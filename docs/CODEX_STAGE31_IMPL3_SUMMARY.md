구현과 KA3 실측을 완료했습니다. 최종 판정은 예상대로 **KA3 FAIL / STOP**이며 KA2·산란에는 진입하지 않았습니다.

| 격자 | L2 | η 음수 plane | 해 최소값 |
|---:|---:|---:|---:|
| 32×128 | `2.41727e-2` | 30 | `-1.98224e-5` |
| 64×256 | `7.54084e-3` | 0 | `+3.93199e-17` |
| 128×512 | `2.99217e-3` | 0 | `+5.94250e-15` |

- `p_obs = 1.3335320199`: 사전등록 창 PASS, 기존 acceptance FAIL
- finest centroid `3.10323e-5`, area `4.66215e-5`, L2 `2.99217e-3`: 사전등록 창 모두 적중
- coarse solution-negative 193회, sign-uncertain은 격자별 `591 / 3462 / 32933`
- 안정 꼬리 평가 후 중·미세 격자의 가짜 음수 η plane은 0
- clamp/non-finite는 전 격자 0
- strict compile, self-test, ASan/UBSan, 패치 순차 재생 및 byte identity PASS
- 커밋 없음

납품물:

- [전체 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_IMPL3.md)
- [rung4 패치](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/s31_rung4.patch)
- [rung5 패치](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/s31_rung5.patch)
- [KA3 수치 JSON](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/s31_results/ka3_rev2.json)
- [KA3 전체 로그](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/s31_logs/rung5_ka3_rev2.log)
- [rung4 회귀 로그](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/s31_logs/rung4_guard_rev2.log)