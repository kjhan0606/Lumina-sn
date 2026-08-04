# UV NLTE map split — 실행 대기

## 상태

요청된 무거운 1,169,145행 집계는 이 작업에서 실행하지 않았다. 따라서 매핑/미매핑
에너지 몫의 숫자는 아직 정의된 결과로 보고할 수 없다. `scripts/uv_mapsplit_offline.py`가
기존 `scripts/uv_t2n9_offline.py::parse_linepop`을 재사용해 아래 계약을 fail-closed로
검사하고, 운전석 실행이 성공하면 이 문서를 실측 보고서로 교체한다.

- schema `LCMFLP01-v1`
- iteration `10`, field_generation `10`
- SHA-256 `84d1849dafd1c796dac77c4037b19683e3ef1d5ddb72dd0e6bf701490b05a1cc`
- rows `1,169,145`, shells `{0,8,16,20,45}`, wavelength `600–3000 Å`

## 계산 규약

production의 선별 조립은 `src/lumina_cmfgen.c:1369-1395`이며, 각 행에서
`eta_l = w * S_l_used`를 직접 계산한다. 에너지 ledger는 이 값을 해당 line bin의
`dnu`와 곱한 `eta_l*dnu`다. bin 총량을 opacity share로 선에 재배분하지 않는다.
얇은 선의 production 분자는 `tau <= 1e-6`일 때 `tau`이고
(`src/lumina_cmfgen.c:1369-1370`), 분석기는 이를 반영해 writer가 기록한 `w`를
그대로 쓴다.

writer의 매핑 술어는 `src/lumina_cmfgen.c:822-824`의
`nlte->nlte_line_map && nlte->nlte_line_map[l] >= 0`이다. 참이면 bit 0
`CMF_LP_F_NLTE_ION`을 세우므로(`src/lumina_cmfgen.c:528-534`), 분석 술어는
`(flags & 1) != 0`이다.

이 프로젝트의 `ion_number`는 0-기반이다. `src/lumina_plasma.c:7672-7684`가 O I,
O II, O III를 각각 원값 0, 1, 2로 명시한다. 이온 순위 CSV는 원값
`ion_number_raw`와 `spectroscopic_stage=Roman(ion_number_raw+1)`을 모두 싣는다.

분모가 0인 분율은 대체하지 않고 JSON `null`, CSV `UNDEFINED`로 기록한다.
clamp/floor/cap/fallback/대체는 없다.

## EPAY 주의

교차표는 artifact의 `src/lumina_cmfgen.c:904-919` disposition과 결합한다. 기존
독립 리뷰 `docs/CODEX_STAGE32_RUNG1_REVIEW.md` F1이 지적했듯 writer의
`rate_shape_replaced` 재구성에는 production의 `acc_w > 0` 조건이 빠져 있으므로,
이는 branch-site 관측이 아니라 payload 기록값과의 교차표로 한정한다.

## 자기검사

경량 synthetic fixture에서 같은 입력의 전체 CSV/JSON/Markdown 직렬화를 두 번
수행해 byte identity를 검사한다. 또한 매핑 bit 0 대신 존재하지 않는 bit 31을 읽는
결함을 주입해 집계가 변해야 통과한다. 이 경량 자기검사는 `PASS`했다
(`repeat_payloads_byte_identical=true`,
`mapping_predicate_negative_control=EXPECTED-CHANGE-OBSERVED`, 이온 표기
`[I,II,III,IV,V]`). 실제 운전석 실행도 전체 입력을 독립적으로 두 번 집계·직렬화한
모든 payload가 byte-identical일 때만 파일을 쓴다.

## 운전석 실행 명령 (한 줄)

```bash
python3 scripts/uv_mapsplit_offline.py --linepop /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/linepop_iter10 --outdir validation/uv_mapsplit --report docs/CODEX_UV_MAPSPLIT.md
```

## 범위 제한

결과는 capture에 포함된 셸 0, 8, 16, 20, 45에만 해당한다. 전 셸로 일반화하지
않는다.
