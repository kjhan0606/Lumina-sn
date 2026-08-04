## 결과

검증기 크래시를 수리했고, 기존 덱만 읽어 네 게이트를 모두 재검증했습니다.

- SL 수 정확일치: 27/27 PASS
- SL 멤버 집합 일치: 27/27 exact PASS
- R1 게이트 유지: 네 게이트 모두 PASS
- gate-OFF `_links` 비트동일: 68개 파일, mismatch 0 PASS
- 최종 exit: `0`

### 크래시 원인

[verify_deck_r4_ftos.py:23](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_deck_r4_ftos.py:23)의 동적 로더가 `module_from_spec()` 후 모듈을 `sys.modules`에 등록하지 않고 `exec_module()`을 호출했습니다.

[audit_r4_ftos.py:40](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/audit_r4_ftos.py:40)의 `@dataclass` 처리 중:

```text
sys.modules.get(cls.__module__).__dict__
```

에서 `cls.__module__ == "r4_audit_for_verify"`였지만 해당 모듈 등록이 없어 `sys.modules.get(...)`가 실제 `None`이었습니다.

수리 내용:

- `exec_module()` 전에 `sys.modules[name] = module` 등록
- 실행 후 같은 모듈로 등록됐는지 명시적 검사
- 기존 광범위 `try/except` 제거: 향후 계약 오류는 traceback과 nonzero exit로 드러남
- OFF-control 부재 또는 `_links` 자기대조는 명시적 FAIL 처리

### 음성 대조

`/tmp` 가짜 덱에서 Si II 첫 FL의 `super_level`만 `0 → 999999`로 변경했습니다.

```text
Si II : 157/157 FL, 80/79 SL; membership=DIFF ... FAIL
f_to_s identity: ions=27, count_fail=1, membership_fail=1, provenance_fail=0
R4 VERDICT: FAIL — no adjustment was made
R4_BAD_SL_FULL_EXIT=1
```

정상 OFF-control의 68-file byte gate는 동시에 PASS했으므로, exit 1은 SL 위반을 독립적으로 검출한 결과입니다.

### `n_levels=31792`, `n_ions=14`

결손이 아닙니다.

- `_fullcov`: 36,355 levels
- `_links`: 31,691 levels
- `_ftos`: 31,792 levels
- 따라서 R4는 `_links` 대비 정확히 `+101`

36,355는 `_links`가 아니라 `_fullcov` 수치였습니다.

`MART n_levels`는 [build_ma_radrecomb_target.py:95](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/build_ma_radrecomb_target.py:95)에서 읽은 `levels.csv`의 전체 global-level 축이며, `_ftos`의 31,792행과 정확히 일치합니다.

`n_ions=14`는 덱 이온 수나 27개 f_to_s 이온 수가 아니라 MA-RADRECOMB의 15개 `TARGET_IONS` 중 성공적으로 target route가 매핑된 수입니다. Ni II 하나는 `phot_g=18` 대 ground multiplet `Σg=21` 불일치로 명시적 `FAILCLOSED`, 나머지 14개가 매핑됐습니다.

### 운전석 명령 한 줄

```bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && python3 scripts/verify_deck_r4_ftos.py --new data/tardis_reference_toy06_19p48d_sivcaiv_ftos --links /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/atomic_links.txt --cmf-run /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4 --links-deck data/tardis_reference_toy06_19p48d_sivcaiv_links --off-control /gpfs/kjhan/lumina_runner2/scratch/r4tmp/r4_ftos_offcontrol
```

UNRESOLVED: R4 네 게이트는 없음. 별도 데이터 이슈로 Ni II MART target의 기존 fail-closed 상태는 남아 있으나 R4 SL/멤버십 결손은 아닙니다.

GPU·모델·덱 재생성·commit은 수행하지 않았고 기존 네 덱도 수정하지 않았습니다.