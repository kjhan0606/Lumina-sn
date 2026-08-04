판정: **D8은 Wave 3 shadow의 pair 기준선을 오염시키지 않았습니다.**

- s8 S/Fe와 s0 Fe 모두 armed/unarmed 전체 oracle CSV가 byte-identical.
- pair ion fraction과 II/III 준위인구도 byte-identical.
- 최대 절대·상대·dex 차이 모두 `0`.
- armed 배너 실측: `33 slots`, `Super-levels: ACTIVE`.
- unarmed도 기존 w31 env의 `LUMINA_SUPER_LEVELS=1` 때문에 `ACTIVE`.
- 따라서 B2의 `p_pair`, `D(pair)`, improvement는 재산정 없이 유효합니다.
- live-run의 전역 33-slot layout 우려만 별도로 잔존합니다.

산출물:

- [D8 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE3_D8_FALSIFIER.md:1)
- [재현·dump helper](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/wave3_d8_pair_dump.py:1)

검증은 parity59 frozen replay 4프로세스만 실행했으며, 신규 모델/GPU 실행과 `src/` 수정은 없었습니다. 모든 실행 exit 0, 최종 `cmp` 6건 모두 성공했습니다.