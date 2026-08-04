결론은 “%급 수리 효과 계량용”으로는 조건부 적격입니다. 구현·폐합 보고서는 읽지 않았고, 프로젝트·테스트·오라클 재실행이나 수정·git 없이 현 소스와 CSV를 직접 대조했습니다.

### ① parity59 캡처 소비 경로

- 원장 실물은 49,272,071,814바이트, 기록 manifest 기준 683,984,500행입니다.
- 작업 스크립트에는 `LUMINA_JBAR_DUMP_IONS=14:1,14:2`만 보이지만, Gate-B가 켜지면 생산 소스가 이를 8개 이온 `(14:1,14:2,16:1,16:2,26:1,26:2,26:3,27:2)`으로 강제 대체합니다. 실제 stderr에도 같은 8개가 기록돼 있습니다: [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14134).
- 캡처는 권위 NLTE 행렬 조립의 실제 소비점에서, 외부 반복 번호와 CE pass 0만 기록합니다: [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14606).
- 선별기는 원장을 메모리에 올리지 않고 `getline()`으로 한 번 순차 주사합니다. 정확한 헤더를 요구한 뒤 소비 반복 11, 셸 `{0,8,43,49}`, 위 8개 이온만 파싱합니다. s0/s8/s43은 출력하고 s49는 completeness sentinel로만 씁니다: [gateb_select_jbar_capture.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/gateb_select_jbar_capture.c:87).
- 각 이온·셸의 line-id에 대해 행 수, 최초/최종 ID, 합, FNV-1a, 엄격 증가 여부를 s49와 비교합니다. 실물 manifest 결과는 `malformed=0`, `complete=1`, 선택 4,103,907행·298,840,263바이트입니다: [jbar_capture_manifest.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/parity59/jbar_capture_manifest.csv:1).
- s0/s8/s43 각각 1,367,969행이며, oracle loader가 실제로 읽었다고 기록한 수도 각각 1,367,969행입니다. 따라서 후반 6개 필드 파싱 실패로 조용히 빠진 행은 현 산출물에서는 없습니다.
- 대표 전이 재대조 14행은 raw J 전부 exact, beta 전부 허용 정밀도 내입니다. 직접 J 분기 11행은 생산 J까지 exact이고, mode 0인 3행은 “raw J가 직접 소비 J가 아님”으로 정확히 구분됩니다: [production_replay_consistency.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/parity59/production_replay_consistency.csv:1).

누락 위험은 다음처럼 제한됩니다.

- 대상 밖 이온·셸은 의도적으로 버리므로 이 oracle을 전 이온·전 셸 검증으로 확대 해석할 수 없습니다.
- Fe IV `(26:3)`는 네 셸 모두 0행입니다. 선별기의 “0 대 0”도 complete로 인정되므로 양의 캡처 완전성을 증명하지는 못합니다. 다만 downstream에서 `raw_jbar_exact_ions=7`, Fe IV `raw_jbar_ion_recorded=0`으로 명시되어 조용한 누락은 아닙니다.
- s49 signature는 line-id 집합/순서 완전성 검사이지 값 자체의 독립 checksum은 아닙니다.

### ② 비교자 CSV 직접 재집계

[oracle_vs_cmfgen.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/parity59/oracle_vs_cmfgen.csv:1)를 CSV 문법으로 직접 센 결과입니다.

| 범주 | 총행 | strict compared | context-only | Lumina 불가 | CMFGEN 불가 |
|---|---:|---:|---:|---:|---:|
| bb | 144 | 28 | 0 | 60 | 56 |
| bf | 192 | 18 | 0 | 60 | 114 |
| collisional | 48 | 0 | 0 | 20 | 28 |
| ff | 15 | 0 | 3 | 0 | 12 |
| input | 72 | 0 | 0 | 0 | 72 |
| state | 84 | 44 | 0 | 18 | 22 |
| thermal | 27 | 9 | 6 | 3 | 9 |
| 합계 | **582** | **99** | **9** | **161** | **313** |

- 셸별 194행씩, 총 582행입니다.
- strict identical coverage: `99/582 = 17.01%`.
- 명시적 비동일 numeric context 포함: `108/582 = 18.56%`.
- 비-compared 474행 모두 사유가 비어 있지 않습니다.
- [coverage_disposition.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/parity59/coverage_disposition.csv:1)의 count 합도 582이며 상태별 `99/9/161/313`으로 직접 집계와 일치합니다.
- 실제 미폐합 잔여는 `heating_MA_LINE_DESTRUCT` 3행, 즉 셸당 1행입니다. 생산 소스는 파일을 쓰지만 parity59 보존 목록에 해당 파일이 없어 archive에서 소실됐습니다. 현재 원장으로 셸 귀속·packet energy/volume 정규화를 복원할 수 없습니다.
- s43의 CMFGEN PRRR/RVTJ 전자밀도 차이는 0.188%입니다: [cmfgen_snapshot_consistency.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/parity59/cmfgen_snapshot_consistency.csv:18). 따라서 s43의 절대 CMFGEN 일치도를 0.1%급으로 해석하면 안 됩니다.

### ③ 수리 잣대 적격성

조건부 적격입니다.

- Jbar와 beta 캡처는 `%.6e`, 즉 소수점 이하 6자리/유효 7자리입니다. 통상 상대 양자화 바닥은 약 `5×10⁻⁷`; consistency 판정은 보수적으로 `5×10⁻⁶`입니다. 1% 효과는 이 운영 바닥보다 약 2,000배 크므로 동일 행·동일 분기의 %급 전후 변화는 충분히 계량 가능합니다.
- 다만 이후 CSV가 9자리나 17자리로 출력돼도 캡처에서 잃은 정밀도가 복구되는 것은 아닙니다.
- 적격 범위는 strict-compared 행과 명시적으로 허용한 context 행뿐입니다. Fe IV raw-J, 잔여 MA heating 3행, 비교 불가 474행에는 수리 성공 판정을 붙일 수 없습니다.
- s43은 동일 CMFGEN snapshot에 대한 전후 상대효과에는 쓸 수 있지만, 0.188% snapshot 불일치 때문에 sub-percent 절대 일치 주장에는 부적격입니다.
- 따라서 이는 “모든 물리를 포괄하는 oracle”은 아니지만, 등록된 수리 표적에서 1% 이상 효과를 재현성 있게 측정하는 잣대로는 충분합니다.

FIT-FOR-PURPOSE — 동일 셸·동일 행의 %급 효과에는 캡처 정밀도가 충분하고 소비 경로·행 수가 폐합됨; 단 Fe IV 0행, MA-heating 잔여 3행, 낮은 strict coverage 및 s43 0.188% snapshot 차이를 명시적으로 제외하는 조건부 적격.