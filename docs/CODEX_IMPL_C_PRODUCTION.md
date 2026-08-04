C1·C2·C3와 게이트 러너 준비를 완료했습니다. `src/`는 건드리지 않았고 commit·push·PR·게이트 실행도 하지 않았습니다.

## 구현 내용

C1:

- `write_abundances()` 정의와 `main()` 호출 제거
- `DEFAULT_ABUNDANCES`, `N_SHELLS` 삭제
- docstring에서 `abundances.csv` 산출 선언 제거
- `atom_masses.csv` 생산 유지
- 두 상수의 다른 소비처: **없음**. 둘 다 제거된 `write_abundances()`만 사용했습니다.
- 기존 선행 변경은 보존했습니다.

C2 드라이버 위치:

- [deck_quarantine_driver.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_quarantine_driver.py:50): 형상 검사 함수, `_ftos` 입력 검사 378행, 필터 출력 검사 463행
- [deck_regen_fullcov_driver.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_regen_fullcov_driver.py:46): `REBUILT` 제거, 복사 직후 검사 98–99행
- [deck_regen_r1_vintage_driver.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_regen_r1_vintage_driver.py:42): `REBUILT` 제거, 복사 직후 검사 99–100행
- [deck_regen_r4_ftos_driver.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_regen_r4_ftos_driver.py:42): `REBUILT` 제거, 복사 직후 검사 105–106행
- [deck_regen_r4_offcontrol_driver.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_regen_r4_offcontrol_driver.py:30): `REBUILT` 제거, 동반 파일 복사 직후 검사 92행

모든 불일치 메시지는 `abundance_columns`와 `geometry_rows`를 함께 출력하며 비영 종료합니다. 기존 `resolve(strict=True)` 후 `copy2()` 경로를 유지해 심링크는 일반 파일로 해소됩니다.

C3:

- `_fullcov`, `_links`, `_ftos`의 30열 결함 파일을 각각 `abundances.csv.defective_20260803`으로 개명 보존
- 정본 조성을 2,808바이트 일반 파일로 복사
- 세 덱의 `atom_masses.csv` SHA-256은 전후 모두 `216d9df4…f6d8c8`로 동일

## 변경 파일 전수

- [G7_PRE_HASHES_20260803.txt](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/G7_PRE_HASHES_20260803.txt)
- [expand_atomic_data_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:10)
- 위 드라이버 5기
- [run_composition_c_gate.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_composition_c_gate.py:1)
- `_fullcov/abundances.csv`
- `_fullcov/abundances.csv.defective_20260803`
- `_links/abundances.csv`
- `_links/abundances.csv.defective_20260803`
- `_ftos/abundances.csv`
- `_ftos/abundances.csv.defective_20260803`

G7 사전 해시 9개는 C1/C2/C3 수정 전에 봉인했습니다. 변경된 Python 7개 파일은 AST 문법 검사만 통과시켰습니다.

## 운전석 실행 명령

```bash
python3 scripts/run_composition_c_gate.py
```

러너는 G1–G7과 임시 30열 결함 사본의 NEG-G1·NEG-G2를 포함합니다. G6은 전체 확장기가 외부 CMFGEN 원자 트리와 무거운 전체 재생성을 요구하므로, 발주서가 허용한 `write_abundances` 정의 부재 + `main()` 호출/출력 경로 부재 AST 검사로 구현했습니다.

남은 위험은 게이트 및 실제 드라이버 재생성을 실행하지 않았다는 점, G6이 동적 음성대조가 아닌 정적 대체라는 점입니다. 또한 C3는 Lumina 정본 계보 복원이며 CMFGEN 조성과의 동일성은 별건 J로 남습니다. 기존의 광범위한 dirty worktree와 `src/` 변경은 그대로 보존했습니다.