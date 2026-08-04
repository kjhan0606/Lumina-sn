두 오프라인 분석 스크립트를 작성했습니다.

- [pile_ion_attribution.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/pile_ion_attribution.py)
- [scheme_fracture_s0.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/scheme_fracture_s0.py)

사용법:

```bash
python3 scripts/pile_ion_attribution.py --outdir validation/pile_ion_attribution
python3 scripts/scheme_fracture_s0.py --outdir validation/scheme_fracture_s0
```

자기검사:

```bash
python3 scripts/pile_ion_attribution.py --self-test
python3 scripts/scheme_fracture_s0.py --self-test
```

두 fixture 결과 모두 `status: PASS`, 반복 산출물 byte-identical입니다. A의 Numba 가속 경로와 Python fixture 참조 경로도 byte parity PASS, B의 인자분해 곱과 직접 에너지비도 PASS했습니다.

음성 대조 실제 출력:

```text
FAIL (expected): INJECTED-DEFECT emitted-ion attribution failed fixture oracle
```

```text
FAIL (expected): mapsplit mismatch s0 B3 line_emission_energy_total: 0.0008287419960437943 != 0.004659958548714409
```

남은 미해결:

- 현재 캡처에는 `lumina_events.bin.sha256`과 `lumina_events_lines.bin.sha256`이 없습니다. A는 데이터 통과 전에 다음처럼 fail-closed됩니다.

```text
UNRESOLVED-FAIL-CLOSED: missing SHA-256 sidecar: /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/lumina_events.bin.sha256
```

- event archive는 iteration 11, fluor matrix는 iteration 10입니다. 스크립트는 이를 독립 검증하며 직접 동세대 비교로 취급하지 않습니다.
- 지정 summary의 attempted count `970557187`과 현재 캡처 `stdout.log`의 `970557175` 사이에 12건 차이가 있습니다.
- 실데이터 전체 통과를 실행하지 않았으므로 Co IV 실제 활성화 몫, prefix 이동량, B의 개별 선 순위와 지배 인자 판정은 계산 노드 실행 전까지 미해결입니다.

모델/GPU 실행, 커밋, `uv_mapsplit_offline.py`·`src/` 수정은 하지 않았습니다.